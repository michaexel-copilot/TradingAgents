"""Rule-based buy and sell signals from daily price, derivatives and sentiment data.

The rules follow the pattern the QNT rally of September 2026 showed: the best
entry was "price breaks out while the crowd doubts it" (low social volume,
neutral or negative funding), on-chain activity rose before social media did,
and the exit came with the social and leverage euphoria. Every rule works on
whatever columns are present; a missing source only removes its condition.

Input frame columns (daily, all optional except ``close``): close, volume,
funding (percent per 8h), oi_usd, fng, social_volume, sentiment, active_addresses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Thresholds:
    """Rule parameters. The defaults are a starting point, not an optimum."""

    breakout_days: int = 20          # close above the highest close of this many days
    volume_spike: float = 2.0        # volume vs. its 14-day median
    max_extension_buy: float = 1.5   # no buy above 1.5x the 20-day average (no chasing)
    quiet_social: float = 3.0        # social volume below 3x its 14-day mean = crowd still quiet
    neutral_funding: float = 0.01    # percent per 8h; at or below = no crowded longs
    onchain_rise: float = 1.3        # active addresses vs. 14-day mean
    onchain_days: int = 3            # ... for this many days in a row
    hype_social: float = 4.0         # social volume spike that marks euphoria
    oi_multiple: float = 3.0         # open interest vs. 7 days earlier
    hot_funding: float = 0.05        # percent per 8h
    rsi_hot: float = 80.0
    extension_hot: float = 1.3       # close vs. 20-day average, together with rsi_hot
    parabolic_7d: float = 2.5        # close vs. 7 days earlier
    sell_confirmations: int = 2      # euphoria conditions needed for a sell
    extreme_funding: float = -0.1    # percent per 8h; crowded shorts, contrarian watch


DEFAULT_THRESHOLDS = Thresholds()


@dataclass
class Signal:
    day: pd.Timestamp
    kind: str                        # buy | sell | watch
    label: str
    close: float
    reasons: list[str] = field(default_factory=list)
    end_day: pd.Timestamp | None = None
    outcome: str = ""
    return_pct: float | None = None

    def as_dict(self) -> dict:
        data = asdict(self)
        data["day"] = self.day.date().isoformat()
        data["end_day"] = self.end_day.date().isoformat() if self.end_day is not None else None
        return data


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return (100 - 100 / (1 + rs)).where(loss > 0, 100.0)


def _ratio_to_past(series: pd.Series, window: int, how: str = "mean") -> pd.Series:
    """Today's value vs. the statistic of the ``window`` days before it."""
    past = series.shift(1).rolling(window, min_periods=max(3, window // 2))
    base = past.median() if how == "median" else past.mean()
    return series / base.replace(0, float("nan"))


def add_features(df: pd.DataFrame, t: Thresholds | None = None) -> pd.DataFrame:
    t = t or DEFAULT_THRESHOLDS
    out = df.copy()
    close = out["close"]
    out["sma20"] = close.rolling(20, min_periods=10).mean()
    out["extension"] = close / out["sma20"]
    out["rsi"] = rsi(close)
    out["prior_high"] = close.shift(1).rolling(t.breakout_days, min_periods=10).max()
    out["ret_7d"] = close / close.shift(7)
    if "volume" in out:
        out["volume_ratio"] = _ratio_to_past(out["volume"], 14, "median")
    if "social_volume" in out:
        out["social_ratio"] = _ratio_to_past(out["social_volume"], 14)
    if "active_addresses" in out:
        out["address_ratio"] = _ratio_to_past(out["active_addresses"], 14)
    if "oi_usd" in out:
        out["oi_7d"] = out["oi_usd"] / out["oi_usd"].shift(7)
    return out


def _value(row: pd.Series, column: str):
    value = row.get(column)
    return None if value is None or pd.isna(value) else float(value)


def _crowd_checks(row: pd.Series, t: Thresholds) -> tuple[bool, list[str]]:
    """Social volume quiet and funding not crowded long; missing data passes."""
    reasons, ok = [], True
    social = _value(row, "social_ratio")
    if social is not None:
        ok &= social < t.quiet_social
        reasons.append(f"Social Volume {social:.1f}× Schnitt (Crowd {'noch ruhig' if social < t.quiet_social else 'schon laut'})")
    funding = _value(row, "funding")
    if funding is not None:
        ok &= funding <= t.neutral_funding
        reasons.append(f"Funding {funding:+.3f} % ({'neutral/short' if funding <= t.neutral_funding else 'Longs gedrängt'})")
    return ok, reasons


def detect_day(row: pd.Series, onchain_streak: int, t: Thresholds) -> list[Signal]:
    day, close = row.name, float(row["close"])
    signals = []

    crowd_ok, crowd_reasons = _crowd_checks(row, t)

    # Early on-chain buy: addresses rise for days while the crowd is quiet.
    addresses = _value(row, "address_ratio")
    extension = _value(row, "extension")
    if (
        addresses is not None
        and onchain_streak >= t.onchain_days
        and crowd_ok
        and (extension is None or extension <= t.max_extension_buy)
    ):
        signals.append(Signal(day, "buy", "Kauf (früh, On-Chain)", close, [
            f"Aktive Adressen {addresses:.1f}× Schnitt, {onchain_streak} Tage in Folge", *crowd_reasons,
        ]))

    # Breakout buy: new high on rising volume, crowd not yet on board.
    prior_high = _value(row, "prior_high")
    volume = _value(row, "volume_ratio")
    if (
        prior_high is not None
        and close > prior_high
        and (volume is None or volume >= t.volume_spike)
        and crowd_ok
        and (extension is None or extension <= t.max_extension_buy)
    ):
        reasons = [f"Schluss {close:.4g} $ über dem {t.breakout_days}-Tage-Hoch {prior_high:.4g} $"]
        if volume is not None:
            reasons.append(f"Volumen {volume:.1f}× Median")
        signals.append(Signal(day, "buy", "Kauf (Ausbruch, Crowd skeptisch)", close, reasons + crowd_reasons))

    # Euphoria sell: several hype conditions at once.
    hype = []
    social = _value(row, "social_ratio")
    if social is not None and social >= t.hype_social:
        hype.append(f"Social Volume {social:.1f}× Schnitt")
    oi_7d = _value(row, "oi_7d")
    if oi_7d is not None and oi_7d >= t.oi_multiple:
        hype.append(f"Open Interest {oi_7d:.1f}× in 7 Tagen")
    funding = _value(row, "funding")
    if funding is not None and funding >= t.hot_funding:
        hype.append(f"Funding {funding:+.3f} % (Longs überhitzt)")
    rsi_value = _value(row, "rsi")
    if rsi_value is not None and extension is not None and rsi_value >= t.rsi_hot and extension >= t.extension_hot:
        hype.append(f"RSI {rsi_value:.0f}, Kurs {100 * (extension - 1):.0f} % über 20-Tage-Schnitt")
    ret_7d = _value(row, "ret_7d")
    if ret_7d is not None and ret_7d >= t.parabolic_7d:
        hype.append(f"+{100 * (ret_7d - 1):.0f} % in 7 Tagen")
    if len(hype) >= t.sell_confirmations:
        signals.append(Signal(day, "sell", "Verkauf (Euphorie)", close, hype))

    # Contrarian watch: crowded shorts. Only a watch, it needs price confirmation.
    if funding is not None and funding <= t.extreme_funding:
        signals.append(Signal(day, "watch", "Kontra-Beobachtung (Funding extrem negativ)", close, [
            f"Funding {funding:+.3f} %: viele Shorts, Squeeze möglich; erst kaufen, wenn der Kurs neue Hochs macht",
        ]))
    return signals


def detect_signals(features: pd.DataFrame, start: pd.Timestamp, t: Thresholds | None = None) -> list[Signal]:
    """Signals from ``start`` on, consecutive days of the same label merged into one."""
    t = t or DEFAULT_THRESHOLDS
    raw: list[Signal] = []
    streak = 0
    for day, row in features.iterrows():
        ratio = _value(row, "address_ratio")
        streak = streak + 1 if ratio is not None and ratio >= t.onchain_rise else 0
        if day >= start:
            raw.extend(detect_day(row, streak, t))

    merged: list[Signal] = []
    for signal in raw:
        last = next((s for s in reversed(merged) if s.label == signal.label), None)
        if last is not None and (signal.day - (last.end_day or last.day)).days <= 2:
            last.end_day = signal.day
            continue
        merged.append(signal)
    merged.sort(key=lambda s: s.day)
    return merged


def score_outcomes(signals: list[Signal], features: pd.DataFrame) -> None:
    """What each signal would have returned: buys until the next sell (or the last day)."""
    close = features["close"]
    last_day = close.index[-1]
    for signal in signals:
        if signal.kind == "buy":
            exit_day = next((s.day for s in signals if s.kind == "sell" and s.day > signal.day), last_day)
            signal.return_pct = 100 * (close[exit_day] / signal.close - 1)
            target = "bis Verkaufssignal" if exit_day != last_day else "bis heute"
            signal.outcome = f"{target}: {signal.return_pct:+.0f} %"
        elif signal.kind == "sell":
            signal.return_pct = 100 * (close[last_day] / signal.close - 1)
            signal.outcome = f"danach {signal.return_pct:+.0f} % bis Ende Zeitraum"
        else:
            later = close[signal.day:].iloc[1:15]
            if not later.empty:
                best = 100 * (later.max() / signal.close - 1)
                worst = 100 * (later.min() / signal.close - 1)
                signal.outcome = f"14 Tage danach: max {best:+.0f} %, min {worst:+.0f} %"


def current_stance(signals: list[Signal], features: pd.DataFrame) -> tuple[str, str]:
    """A one-word stance for the last day and the reason for it."""
    last_day = features.index[-1]
    recent = [s for s in signals if ((s.end_day or s.day) >= last_day - pd.Timedelta(days=3))]
    for kind, word in (("sell", "Verkaufen / Gewinne sichern"), ("buy", "Kaufsignal aktiv")):
        hit = next((s for s in reversed(recent) if s.kind == kind), None)
        if hit:
            return word, f"{hit.label} am {hit.day:%d.%m.}"
    last = features.iloc[-1]
    parts = []
    if _value(last, "volume_ratio") is not None:
        parts.append(f"Volumen {last['volume_ratio']:.1f}× Median")
    if _value(last, "funding") is not None:
        parts.append(f"Funding {last['funding']:+.3f} %")
    return "Abwarten", "kein frisches Signal" + (": " + ", ".join(parts) if parts else "")
