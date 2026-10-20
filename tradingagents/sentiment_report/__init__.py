"""Sentiment report for one coin and period: data, signals, written assessment, HTML.

    tradingagents sentiment QNT --days 56
    python -m tradingagents.sentiment_report QNT --start 2026-08-24 --end 2026-10-19

See ``dataflows.vendors.crypto_sentiment`` for the data, ``signals`` for the rules and ``narrative`` for
which LLM backends may be used with which account.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import cache
from pathlib import Path

import pandas as pd

from tradingagents.dataflows.vendors import crypto_sentiment as sources
from tradingagents.sentiment_report import narrative
from tradingagents.sentiment_report.signals import (
    Signal,
    Thresholds,
    add_features,
    current_stance,
    detect_signals,
    score_outcomes,
)

logger = logging.getLogger(__name__)

# Version of the report logic, raised with every change to data, rules or layout, so the
# web page can tell which reports were made by an older version.
REPORT_VERSION = "1.0"

# Days fetched before the window so the 14- and 20-day averages are warm on day one.
WARMUP_DAYS = 40


@cache
def software_version() -> str:
    """The report version plus the git commit of the checkout when there is one."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).parent, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    return f"{REPORT_VERSION} ({commit})" if commit else REPORT_VERSION


@dataclass
class Report:
    coin: str
    start: date
    end: date
    features: pd.DataFrame
    signals: list[Signal]
    stance: tuple[str, str]
    coverage: dict[str, str]
    snapshot: dict[str, str] = field(default_factory=dict)
    narrative: str | None = None
    narrative_note: str = ""
    created: datetime = field(default_factory=lambda: datetime.now().astimezone().replace(microsecond=0))
    version: str = field(default_factory=software_version)

    @property
    def window(self) -> pd.DataFrame:
        return self.features[self.features.index >= pd.Timestamp(self.start)]


def collect(coin: str, start: date, end: date, santiment_slug: str | None = None) -> tuple[pd.DataFrame, dict]:
    """One daily frame from every reachable source, and a line per source on what it gave."""
    fetch_from = start - timedelta(days=WARMUP_DAYS)
    coverage: dict[str, str] = {}

    price, name = sources.fetch_price(coin, fetch_from, end)  # no price, no report
    coverage["Kurs & Volumen"] = name
    frame = price

    def attach(label, fetch):
        nonlocal frame
        try:
            data, origin = fetch()
        except sources.SourceError as exc:
            coverage[label] = f"fehlt ({exc})"
            return None
        frame = frame.join(data, how="left")
        coverage[label] = origin
        return data

    attach("Funding Rate", lambda: sources.fetch_funding(coin, fetch_from, end))
    attach("Perp-Volumen", lambda: sources.fetch_perp_volume(coin, fetch_from, end))
    attach("Long/Short-Verhältnis", lambda: sources.fetch_long_share(coin, fetch_from, end))
    if attach("Open Interest", lambda: sources.fetch_open_interest(coin, fetch_from, end)) is not None:
        frame["oi_usd"] = frame["oi_coins"] * frame["close"]
    attach("Fear & Greed (Gesamtmarkt)", lambda: sources.fetch_fear_greed(fetch_from, end))

    def santiment():
        data, problems = sources.fetch_santiment(coin, fetch_from, end, santiment_slug)
        note = "Santiment" + (f" (teilweise: {'; '.join(problems)})" if problems else "")
        return data, note

    attach("Social Volume & On-Chain", santiment)

    frame = frame[frame.index <= pd.Timestamp(end)]
    return frame, coverage


def _num(value, digits=4):
    return None if value is None or pd.isna(value) else round(float(value), digits)


def llm_payload(report: Report) -> dict:
    window = report.window
    columns = [c for c in ("close", "volume", "volume_ratio", "perp_volume", "perp_spot", "perp_spot_ratio", "long_share", "funding", "oi_usd", "fng", "social_volume",
                           "social_ratio", "sentiment", "active_addresses", "exchange_inflow",
                           "exchange_outflow", "inflow_ratio", "whale_transactions", "whale_ratio", "rsi") if c in window]
    daily = [
        {"datum": day.date().isoformat(), **{c: _num(row[c]) for c in columns}}
        for day, row in window.iterrows()
    ]
    close = window["close"]
    return {
        "coin": report.coin,
        "erstellt": report.created.isoformat(),
        "software_version": report.version,
        "zeitraum": [report.start.isoformat(), report.end.isoformat()],
        "kennzahlen": {
            "erster_schluss": _num(close.iloc[0]),
            "letzter_schluss": _num(close.iloc[-1]),
            "hoch": _num(close.max()),
            "hoch_datum": close.idxmax().date().isoformat(),
            "tief": _num(close.min()),
            "tief_datum": close.idxmin().date().isoformat(),
        },
        "lage_letzter_tag": {"einschätzung": report.stance[0], "grund": report.stance[1]},
        "signale": [s.as_dict() for s in report.signals],
        "datenquellen": report.coverage,
        "social_snapshot_letzte_7_tage": report.snapshot or "nicht abgerufen (Zeitraum endet nicht in der letzten Woche)",
        "tageswerte": daily,
    }


def build_report(
    coin: str,
    start: date,
    end: date,
    *,
    llm_backend: str = "auto",
    llm_model: str | None = None,
    santiment_slug: str | None = None,
    thresholds: Thresholds | None = None,
    frame: pd.DataFrame | None = None,
    coverage: dict | None = None,
) -> Report:
    """Fetch (unless ``frame`` is given), detect signals and write the narrative."""
    coin = coin.upper().removesuffix("-USD").removesuffix("USDT")
    thresholds = thresholds or Thresholds()
    if frame is None:
        frame, coverage = collect(coin, start, end, santiment_slug)
    features = add_features(frame, thresholds)
    window_start = pd.Timestamp(start)
    if features[features.index >= window_start].empty:
        raise sources.SourceError(f"keine Kursdaten für {coin} im Zeitraum {start} bis {end}")
    signals = detect_signals(features, window_start, thresholds)
    score_outcomes(signals, features)
    report = Report(coin, start, end, features, signals, current_stance(signals, features), coverage or {})
    if narrative.resolve_backend(llm_backend) != "none":
        try:
            report.snapshot = sources.social_snapshot(coin, end)
        except Exception as exc:  # the snapshot is a bonus; never fail the report on it
            logger.warning("Social snapshot failed: %s", exc)
        prompt = narrative.build_prompt(coin, start.isoformat(), end.isoformat(), llm_payload(report))
        report.narrative, report.narrative_note = narrative.write_narrative(llm_backend, prompt, llm_model)
    else:
        report.narrative_note = "ohne KI-Text (Backend none)"
    return report


def save_report(report: Report, out_dir: Path) -> tuple[Path, Path]:
    from tradingagents.sentiment_report.render import render_html

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report.coin}_{report.start:%Y%m%d}-{report.end:%Y%m%d}"
    html_path = out_dir / f"{stem}_sentiment.html"
    json_path = out_dir / f"{stem}_sentiment.json"
    html_path.write_text(render_html(report), encoding="utf-8")
    json_path.write_text(json.dumps(llm_payload(report) | {"narrative": report.narrative}, ensure_ascii=False,
                                    indent=1, default=str), encoding="utf-8")
    return html_path, json_path
