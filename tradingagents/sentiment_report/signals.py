"""Rule-based buy and sell signals from daily price, derivatives and sentiment data.

The rules follow the pattern the QNT rally of September 2026 showed: the best
entry was "price breaks out while the crowd doubts it" (low social volume,
neutral or negative funding), on-chain activity rose before social media did,
and the exit came with the social and leverage euphoria. Every rule works on
whatever columns are present; a missing source only removes its condition.

Input frame columns (daily, all optional except ``close``): close, volume,
funding (percent per 8h), oi_usd, perp_volume (USD), long_share (0..1), fng, social_volume, sentiment,
active_addresses, exchange_inflow, exchange_outflow (coins), whale_transactions.
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
    crowded_long: float = 0.65       # share of perp accounts net long at which the crowd is all in
    long_shift: float = 0.08         # ... and it rose this much (share points) above its 14-day mean
    perp_spot_jump: float = 2.0      # perp/spot volume ratio vs. its 14-day mean: leverage takes over
    inflow_jump: float = 3.0         # coins sent to exchanges vs. 14-day mean: holders get ready to sell
    outflow_days: int = 3            # net outflow on this many of the last 5 days confirms a buy
    whale_jump: float = 3.0          # whale transactions vs. 14-day mean ...
    whale_rally: float = 1.3         # ... after this 7-day rise: big holders active into strength
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
    # Worst and best move against the entry close while the trade was open (for SL/TP):
    # for a buy or watch the lowest low and highest high, for a sell taken as a
    # short the highest high and lowest low, each in percent.
    max_drawdown_pct: float | None = None
    max_runup_pct: float | None = None
    drawdown_day: pd.Timestamp | None = None

    def as_dict(self) -> dict:
        data = asdict(self)
        data["day"] = self.day.date().isoformat()
        data["end_day"] = self.end_day.date().isoformat() if self.end_day is not None else None
        data["drawdown_day"] = self.drawdown_day.date().isoformat() if self.drawdown_day is not None else None
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
    if "perp_volume" in out and "volume" in out:
        # Perp volume is one exchange, spot is market-wide; the level differs per
        # coin, so the rule watches the ratio against its own recent mean.
        out["perp_spot"] = out["perp_volume"] / out["volume"].replace(0, float("nan"))
        out["perp_spot_ratio"] = _ratio_to_past(out["perp_spot"], 14)
    if "long_share" in out:
        out["long_shift"] = out["long_share"] - out["long_share"].shift(1).rolling(14, min_periods=7).mean()
    if "exchange_inflow" in out:
        out["inflow_ratio"] = _ratio_to_past(out["exchange_inflow"], 14)
        if "exchange_outflow" in out:
            net = out["exchange_inflow"] - out["exchange_outflow"]
            # Days of the last five on which more coins left exchanges than arrived.
            out["outflow_days"] = (net < 0).astype(float).where(net.notna()).rolling(5, min_periods=3).sum()
    if "whale_transactions" in out:
        out["whale_ratio"] = _ratio_to_past(out["whale_transactions"], 14)
    return out


def _value(row: pd.Series, column: str):
    value = row.get(column)
    return None if value is None or pd.isna(value) else float(value)


def _crowd_checks(row: pd.Series, t: Thresholds) -> tuple[bool, list[str]]:
    """Social volume quiet, funding not crowded long, no rush of coins to the
    exchanges; missing data passes. Steady exchange outflows add a confirmation."""
    reasons, ok = [], True
    inflow = _value(row, "inflow_ratio")
    if inflow is not None and inflow >= t.inflow_jump:
        ok = False
        reasons.append(f"Börsenzufluss {inflow:.1f}× Schnitt (Verkaufsdruck)")
    outflow_days = _value(row, "outflow_days")
    if outflow_days is not None and outflow_days >= t.outflow_days:
        reasons.append(f"bestätigt: Netto-Abfluss von Börsen an {outflow_days:.0f} der letzten 5 Tage")
    social = _value(row, "social_ratio")
    if social is not None:
        ok &= social < t.quiet_social
        reasons.append(f"Social Volume {social:.1f}× Schnitt (Crowd {'noch ruhig' if social < t.quiet_social else 'schon laut'})")
    long_share = _value(row, "long_share")
    if long_share is not None:
        ok &= long_share < t.crowded_long
        reasons.append(f"Long-Anteil {100 * long_share:.0f} % ({'Crowd skeptisch' if long_share <= 0.5 else 'Crowd eher long' if long_share < t.crowded_long else 'Crowd voll long'})")
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
    perp_spot, perp_jump = _value(row, "perp_spot"), _value(row, "perp_spot_ratio")
    if perp_jump is not None and perp_jump >= t.perp_spot_jump:
        hype.append(f"Perp-Volumen {perp_spot:.1f}× Spot ({perp_jump:.1f}× üblich, Hebel treibt)")
    long_share, long_shift = _value(row, "long_share"), _value(row, "long_shift")
    if long_share is not None and long_shift is not None and long_share >= t.crowded_long and long_shift >= t.long_shift:
        hype.append(f"Long-Anteil {100 * long_share:.0f} % (+{100 * long_shift:.0f} Punkte über Schnitt)")
    inflow = _value(row, "inflow_ratio")
    if inflow is not None and inflow >= t.inflow_jump:
        hype.append(f"Börsenzufluss {inflow:.1f}× Schnitt (Halter schicken Coins zum Verkauf)")
    whales = _value(row, "whale_ratio")
    rise = _value(row, "ret_7d")
    if whales is not None and whales >= t.whale_jump and rise is not None and rise >= t.whale_rally:
        hype.append(f"Wal-Transaktionen {whales:.1f}× Schnitt nach +{100 * (rise - 1):.0f} % in 7 Tagen")
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


def _excursions(signal: Signal, features: pd.DataFrame, until: pd.Timestamp, short: bool = False) -> None:
    """Max drawdown and run-up of a position opened at the signal's close, over the days
    after entry up to ``until``, in percent from the position's view (a short loses when
    the price rises). Uses the daily low/high where the source has them, else the close.
    """
    path = features[(features.index > signal.day) & (features.index <= until)]
    if path.empty:
        return
    low = path["low"] if "low" in path else path["close"]
    high = path["high"] if "high" in path else path["close"]
    sign = -1 if short else 1
    adverse, favourable = (high, low) if short else (low, high)
    worst_day = adverse.idxmax() if short else adverse.idxmin()
    signal.max_drawdown_pct = min(0.0, sign * 100 * (adverse[worst_day] / signal.close - 1))
    best = favourable.min() if short else favourable.max()
    signal.max_runup_pct = max(0.0, sign * 100 * (best / signal.close - 1))
    signal.drawdown_day = worst_day


def _excursion_text(signal: Signal, label: str = "max. Drawdown") -> str:
    if signal.max_drawdown_pct is None:
        return ""
    return (f"; {label} {signal.max_drawdown_pct:+.0f} % ({signal.drawdown_day:%d.%m.}), "
            f"max. Gewinn {signal.max_runup_pct:+.0f} %")


def score_outcomes(signals: list[Signal], features: pd.DataFrame) -> None:
    """What each signal would have returned: buys until the next sell (or the last day),
    with the max drawdown on the way for stop-loss and take-profit analysis."""
    close = features["close"]
    last_day = close.index[-1]
    for signal in signals:
        if signal.kind == "buy":
            exit_day = next((s.day for s in signals if s.kind == "sell" and s.day > signal.day), last_day)
            signal.return_pct = 100 * (close[exit_day] / signal.close - 1)
            target = "bis Verkaufssignal" if exit_day != last_day else "bis heute"
            signal.outcome = f"{target}: {signal.return_pct:+.0f} %"
            _excursions(signal, features, exit_day)
            signal.outcome += _excursion_text(signal)
        elif signal.kind == "sell":
            signal.return_pct = 100 * (close[last_day] / signal.close - 1)
            signal.outcome = f"danach {signal.return_pct:+.0f} % bis Ende Zeitraum"
            # Seen as a short: the drawdown is the highest the price went after the sell.
            _excursions(signal, features, last_day, short=True)
            signal.outcome += _excursion_text(signal, "max. Drawdown als Short")
        else:
            later = close[signal.day:].iloc[1:15]
            if not later.empty:
                best = 100 * (later.max() / signal.close - 1)
                worst = 100 * (later.min() / signal.close - 1)
                signal.outcome = f"14 Tage danach: max {best:+.0f} %, min {worst:+.0f} %"
                _excursions(signal, features, later.index[-1])
                if signal.drawdown_day is not None:  # max/min above already are run-up and drawdown
                    signal.outcome += f" (Tief am {signal.drawdown_day:%d.%m.})"


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
