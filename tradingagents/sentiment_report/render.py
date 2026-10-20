"""The report as one self-contained HTML page (light and dark), chart drawn as inline SVG."""

from __future__ import annotations

import math
from html import escape

import pandas as pd
from markdown_it import MarkdownIt

from tradingagents.sentiment_report.signals import Thresholds

PILL = {"buy": "p-buy", "sell": "p-sell", "watch": "p-warn"}

CSS = """
:root{--bg:#f4f6f8;--panel:#fff;--ink:#15202b;--muted:#5a6773;--line:#d9dfe5;--accent:#2b4cc4;
--buy:#1a8a5a;--sell:#c2412d;--warn:#a86a12;--grid:#e6eaee;--buy-bg:#e3f4ec;--sell-bg:#fbe7e3;--warn-bg:#fbf0dc;
--mono:ui-monospace,"JetBrains Mono",Menlo,monospace;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f151b;--panel:#161e26;--ink:#e6ecf1;
--muted:#9aa7b3;--line:#2a3540;--accent:#7d9bff;--buy:#4cc58f;--sell:#f07a63;--warn:#e0a24a;--grid:#222c36;
--buy-bg:#123326;--sell-bg:#3a1d17;--warn-bg:#352812;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:40px 16px 64px;display:flex;flex-direction:column;gap:36px}
h1,h2,h3{margin:0;line-height:1.2}h1{font-size:clamp(28px,5vw,40px)}h2{font-size:22px}
p{margin:0;max-width:70ch}section{display:flex;flex-direction:column;gap:14px}
.eyebrow{font:600 12px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.muted{color:var(--muted)}
.cells{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
.cell{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:14px 16px;display:flex;flex-direction:column;gap:4px}
.cell b{font:700 21px/1.15 var(--mono)}.cell span{font-size:14px;color:var(--muted)}
.box{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:16px;overflow-x:auto}
.box svg{display:block;width:100%;min-width:620px;height:auto}
svg text{font:12px var(--mono);fill:var(--muted)}svg .lbl{font:600 12px system-ui,sans-serif}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:14px;color:var(--muted);margin-top:8px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.tbl{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:15px}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}
th{font:600 12px/1.3 var(--mono);text-transform:uppercase;color:var(--muted);background:var(--bg)}
tr:last-child td{border-bottom:0}td.num{font-family:var(--mono);white-space:nowrap}
.pill{display:inline-block;font:600 12px/1 var(--mono);padding:4px 8px;border-radius:999px;white-space:nowrap}
.p-buy{background:var(--buy-bg);color:var(--buy)}.p-sell{background:var(--sell-bg);color:var(--sell)}
.p-warn{background:var(--warn-bg);color:var(--warn)}
.md h2{font-size:19px;margin-top:8px}.md ul{margin:0;padding-left:22px}.md>*+*{margin-top:10px}
ul{margin:0;padding-left:22px}a{color:var(--accent)}
.note{font-size:14px;color:var(--muted);border-top:1px solid var(--line);padding-top:14px}
"""


def _fmt_price(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f} $".replace(",", ".")
    if value >= 1:
        return f"{value:.2f} $".replace(".", ",")
    return f"{value:.4g} $".replace(".", ",")


def _span(signal) -> str:
    if signal.end_day is not None and signal.end_day != signal.day:
        return f"{signal.day:%d.%m.}–{signal.end_day:%d.%m.}"
    return f"{signal.day:%d.%m.%Y}"


def chart_svg(window: pd.DataFrame, signals) -> str:
    """Log-scale close line, relative volume bars and a marker per signal."""
    width, height, left, right, top = 900, 420, 64, 24, 20
    price_bottom, vol_top, vol_bottom = 290, 308, 380
    close = window["close"]
    n = len(close)
    lo, hi = math.log(close.min() * 0.92), math.log(close.max() * 1.08)

    def x(i):
        return left + (i * (width - left - right) / max(n - 1, 1))

    def y(v):
        return top + (hi - math.log(v)) / (hi - lo) * (price_bottom - top)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Kursverlauf mit Signalen">']
    steps = 5
    for k in range(steps + 1):
        v = math.exp(lo + (hi - lo) * k / steps)
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" style="stroke:var(--grid)"/>')
        parts.append(f'<text x="{left - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{escape(_fmt_price(v))}</text>')
    if "volume" in window and window["volume"].notna().any():
        vmax = window["volume"].max()
        bar = (width - left - right) / max(n - 1, 1) * 0.7
        for i, v in enumerate(window["volume"]):
            if pd.notna(v) and vmax:
                h = v / vmax * (vol_bottom - vol_top)
                parts.append(f'<rect x="{x(i) - bar / 2:.1f}" y="{vol_bottom - h:.1f}" width="{bar:.1f}" '
                             f'height="{h:.1f}" style="fill:var(--accent);opacity:.35"/>')
        parts.append(f'<text x="{left - 8}" y="{vol_top + 10}" text-anchor="end">Vol.</text>')
    parts.append(f'<line x1="{left}" x2="{width - right}" y1="{vol_bottom}" y2="{vol_bottom}" style="stroke:var(--line)"/>')
    label_every = max(1, n // 7)
    for i, day in enumerate(close.index):
        if i % label_every == 0 or i == n - 1:
            parts.append(f'<text x="{x(i):.1f}" y="{vol_bottom + 18}" text-anchor="middle">{day:%d.%m.}</text>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{x(i):.1f} {y(v):.1f}" for i, v in enumerate(close))
    parts.append(f'<path d="{path} L{x(n - 1):.1f} {price_bottom} L{x(0):.1f} {price_bottom} Z" style="fill:var(--accent);opacity:.08"/>')
    parts.append(f'<path d="{path}" style="fill:none;stroke:var(--ink);stroke-width:2;stroke-linejoin:round"/>')
    position = {day: i for i, day in enumerate(close.index)}
    color = {"buy": "buy", "sell": "sell", "watch": "warn"}
    for signal in signals:
        i = position.get(signal.day)
        if i is None:
            continue
        cx, cy, c = x(i), y(close.iloc[i]), color[signal.kind]
        parts.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{cy:.1f}" y2="{price_bottom}" '
                     f'style="stroke:var(--{c});stroke-dasharray:3 3"/>')
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" style="fill:var(--{c});stroke:var(--panel);stroke-width:2"/>')
        anchor = "end" if i > n - 6 else ("start" if i < 4 else "middle")
        ty = cy - 12 if signal.kind == "sell" else cy + 22
        short = signal.label.split(" (")[0]
        parts.append(f'<text class="lbl" x="{cx:.1f}" y="{ty:.1f}" text-anchor="{anchor}" '
                     f'style="fill:var(--{c})">{escape(short)} · {escape(_fmt_price(close.iloc[i]))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_html(report) -> str:
    window = report.window
    close = window["close"]
    first, last = close.iloc[0], close.iloc[-1]
    stance, why = report.stance
    stance_color = {"Abwarten": "warn", "Kaufsignal aktiv": "buy"}.get(stance, "sell")
    buys = [s for s in report.signals if s.kind == "buy"]
    sells = [s for s in report.signals if s.kind == "sell"]
    title = f"{report.coin} Sentiment-Check"

    rows = "".join(
        f'<tr><td class="num">{_span(s)}</td><td><span class="pill {PILL[s.kind]}">{escape(s.label)}</span></td>'
        f'<td>{escape("; ".join(s.reasons))}</td><td class="num">{escape(_fmt_price(s.close))}</td>'
        f"<td>{escape(s.outcome)}</td></tr>"
        for s in report.signals
    ) or '<tr><td colspan="5">Keine Signale im Zeitraum.</td></tr>'

    coverage = "".join(
        f"<tr><td>{escape(k)}</td><td>{escape(v)}</td></tr>" for k, v in report.coverage.items()
    )
    narrative = (
        f'<div class="md">{MarkdownIt().render(report.narrative)}</div>'
        if report.narrative
        else '<p class="muted">Kein KI-Text. Die Signale oben sind regelbasiert berechnet.</p>'
    )
    t = Thresholds()
    summary = (
        f"{len(buys)} Kaufsignal{'e' if len(buys) != 1 else ''} und {len(sells)} Verkaufssignal"
        f"{'e' if len(sells) != 1 else ''} im Zeitraum. Lage am letzten Tag: {stance} ({why})."
    )

    return f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title>
<meta name="report-created" content="{report.created.isoformat()}"><meta name="report-version" content="{escape(report.version)}">
<style>{CSS}</style></head><body><div class="wrap">
<header style="display:flex;flex-direction:column;gap:12px">
<div class="eyebrow">Sentiment-Analyse · {escape(report.coin)} · {report.start:%d.%m.%Y} bis {report.end:%d.%m.%Y}</div>
<h1>{escape(title)}</h1><p style="font-size:18px">{escape(summary)}</p>
<p class="muted">Erstellt am {report.created:%d.%m.%Y um %H:%M} Uhr · Software-Version {escape(report.version)}</p></header>
<div class="cells">
<div class="cell"><span>Schluss {report.end:%d.%m.%Y}</span><b>{escape(_fmt_price(last))}</b><span>{100 * (last / first - 1):+.0f} % im Zeitraum</span></div>
<div class="cell"><span>Hoch ({close.idxmax():%d.%m.})</span><b>{escape(_fmt_price(close.max()))}</b><span>{100 * (last / close.max() - 1):+.0f} % vom Hoch</span></div>
<div class="cell"><span>Tief ({close.idxmin():%d.%m.})</span><b>{escape(_fmt_price(close.min()))}</b><span>Tagesschluss</span></div>
<div class="cell"><span>Aktuelles Signal</span><b style="color:var(--{stance_color})">{escape(stance)}</b><span>{escape(why)}</span></div>
</div>
<section><h2>Kurs und Signale</h2>
<p class="muted">Tagesschlusskurse in USD, logarithmische Skala, Balken: Tagesvolumen relativ.</p>
<div class="box">{chart_svg(window, report.signals)}
<div class="legend"><span><i style="background:var(--buy)"></i>Kaufsignal</span><span><i style="background:var(--sell)"></i>Verkaufssignal</span><span><i style="background:var(--warn)"></i>Beobachten</span></div></div></section>
<section><h2>Die Signale im Detail</h2><div class="tbl"><table>
<thead><tr><th>Datum</th><th>Signal</th><th>Was die Daten zeigten</th><th>Kurs</th><th>Ergebnis</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p class="muted" style="font-size:15px">Renditen rückblickend von Tagesschluss zu Tagesschluss: Käufe bis zum nächsten Verkaufssignal oder bis zum Ende des Zeitraums. Max. Drawdown und max. Gewinn messen ab dem Signal-Schlusskurs den schlechtesten und besten Stand bis zum Ausstieg (Tagestief und -hoch, sonst Schlusskurs), Verkaufssignale gesehen als Short bis Ende Zeitraum, Beobachtungen über 14 Tage. Das Datum nennt den Tag des Drawdowns. Die Werte stehen auch in der JSON-Datei (max_drawdown_pct, max_runup_pct, drawdown_day) für eine spätere SL/TP-Auswertung.</p></section>
<section><h2>Einschätzung</h2>{narrative}<p class="note">{escape(report.narrative_note)}</p></section>
<section><h2>Regeln</h2><ul>
<li><b>Kauf (Ausbruch):</b> Schluss über dem {t.breakout_days}-Tage-Hoch, Volumen ≥ {t.volume_spike:g}× Median, Social Volume unter {t.quiet_social:g}× Schnitt, Funding ≤ {t.neutral_funding:g} %, Long-Anteil der Perp-Konten nicht zugleich ≥ {100 * t.crowded_long:.0f} % und {100 * t.long_shift:.0f} Punkte über seinem 14-Tage-Schnitt, Börsenzufluss unter {t.inflow_jump:g}× Schnitt, Kurs höchstens {t.max_extension_buy:g}× über dem 20-Tage-Schnitt. Netto-Abfluss von Börsen an ≥ {t.outflow_days} der letzten 5 Tage wird als Bestätigung angezeigt.</li>
<li><b>Kauf (früh, On-Chain):</b> aktive Adressen {t.onchain_days} Tage in Folge ≥ {t.onchain_rise:g}× Schnitt, Crowd noch ruhig (nur mit Santiment-Daten).</li>
<li><b>Verkauf (Euphorie):</b> mindestens {t.sell_confirmations} von: Social Volume ≥ {t.hype_social:g}×, Open Interest ≥ {t.oi_multiple:g}× in 7 Tagen, Verhältnis Perp- zu Spot-Volumen ≥ {t.perp_spot_jump:g}× seines 14-Tage-Schnitts, Long-Anteil ≥ {100 * t.crowded_long:.0f} % und {100 * t.long_shift:.0f} Punkte über seinem 14-Tage-Schnitt, Börsenzufluss ≥ {t.inflow_jump:g}× Schnitt, Wal-Transaktionen ≥ {t.whale_jump:g}× Schnitt nach ≥ +{100 * (t.whale_rally - 1):.0f} % in 7 Tagen, Funding ≥ {t.hot_funding:g} %, RSI ≥ {t.rsi_hot:g} bei ≥ {100 * (t.extension_hot - 1):.0f} % über 20-Tage-Schnitt, ≥ +{100 * (t.parabolic_7d - 1):.0f} % in 7 Tagen.</li>
<li><b>Beobachten:</b> Funding ≤ {t.extreme_funding:g} % (viele Shorts). Erst mit neuem Hoch ein Kauf.</li>
</ul><p class="muted">Fehlt eine Quelle, entfällt nur ihre Bedingung. Funding in Prozent je 8 Stunden, Tagesmittel.</p></section>
<section><h2>Datenquellen</h2><div class="tbl"><table><thead><tr><th>Daten</th><th>Quelle</th></tr></thead>
<tbody>{coverage}</tbody></table></div></section>
<p class="note">Erstellt am {report.created:%d.%m.%Y um %H:%M} Uhr mit TradingAgents sentiment_report, Version {escape(report.version)}. Keine Anlageberatung. Signale sind rückblickend berechnet und keine Garantie für die Zukunft.</p>
</div></body></html>"""
