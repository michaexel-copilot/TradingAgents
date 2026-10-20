"""Coin sentiment report: signal rules, outcome scoring, LLM backends and HTML.

Runs on the QNT closes and volumes of Sep/Oct 2026 (CoinLore, USD, volume in
millions), with no network: sources are not called when a frame is passed.
"""

import json
import subprocess
from datetime import date
from unittest import mock

import pandas as pd
import pytest

from tradingagents.sentiment_report import build_report, narrative, save_report
from tradingagents.sentiment_report.__main__ import main, parse_window
from tradingagents.sentiment_report.signals import Thresholds, add_features, detect_day

pytestmark = pytest.mark.unit

_QNT = [
    ("08-31", 61.1, 4.71), ("09-01", 63.68, 7.26), ("09-02", 64.52, 9.2), ("09-03", 65.32, 4.8),
    ("09-04", 63.67, 5.43), ("09-05", 64.89, 3.65), ("09-06", 66.97, 5.26), ("09-07", 66.67, 5.61),
    ("09-08", 67.1, 5.49), ("09-09", 67.13, 6.26), ("09-10", 64.57, 4.75), ("09-11", 63.34, 10.76),
    ("09-12", 63.94, 3.39), ("09-13", 63.07, 3.84), ("09-14", 64.49, 5.16), ("09-15", 61.21, 5.65),
    ("09-16", 61.15, 6.23), ("09-17", 61.13, 4.58), ("09-18", 63.28, 7.5), ("09-19", 65.64, 5.32),
    ("09-20", 64.17, 5.62), ("09-21", 66.91, 9.88), ("09-22", 73.78, 19.07), ("09-23", 70.66, 11.21),
    ("09-24", 90.2, 28.98), ("09-25", 98.65, 78.54), ("09-26", 151.74, 170.43), ("09-27", 286.1, 1070),
    ("09-28", 230.73, 1060), ("09-29", 266.69, 716.4), ("09-30", 287.07, 797.16), ("10-01", 244.88, 445.45),
    ("10-02", 246.62, 369.56), ("10-03", 256.88, 271.78), ("10-04", 252.58, 213.95), ("10-05", 262.84, 200.37),
    ("10-06", 265.5, 195.44), ("10-07", 253.08, 208.72), ("10-08", 236.65, 195.91), ("10-09", 248.37, 128),
    ("10-10", 239.71, 105.73), ("10-11", 238.35, 82.92), ("10-12", 225.94, 97.39), ("10-13", 209.8, 178.29),
    ("10-14", 204.92, 111.8), ("10-15", 204.47, 143.47), ("10-16", 200.75, 81.09), ("10-17", 195.57, 47.47),
    ("10-18", 192.11, 67.51), ("10-19", 202.72, 83.72),
]
START, END = date(2026, 9, 1), date(2026, 10, 19)


def qnt_frame(funding: bool = True) -> pd.DataFrame:
    index = pd.to_datetime([f"2026-{d}" for d, _, _ in _QNT])
    frame = pd.DataFrame({"close": [c for _, c, _ in _QNT], "volume": [v * 1e6 for _, _, v in _QNT]}, index=index)
    if funding:
        # Negative funding through the breakout, the -0.27 % record low on 10-04.
        frame["funding"] = -0.005
        frame.loc["2026-10-04", "funding"] = -0.27
    return frame


def report(**kwargs):
    kwargs.setdefault("llm_backend", "none")
    return build_report("QNT", START, END, frame=qnt_frame(), coverage={"Kurs & Volumen": "Test"}, **kwargs)


def test_qnt_breakout_buy_then_euphoria_sell():
    r = report()
    buys = [s for s in r.signals if s.kind == "buy"]
    sells = [s for s in r.signals if s.kind == "sell"]
    assert [s.day.date().isoformat() for s in buys] == ["2026-09-22"]
    assert buys[0].end_day.date().isoformat() == "2026-09-25"
    assert sells[0].day.date().isoformat() == "2026-09-27"
    # The buy is scored to the sell: 73.78 -> 286.10.
    assert buys[0].return_pct == pytest.approx(100 * (286.1 / 73.78 - 1))


def test_extreme_negative_funding_is_only_a_watch():
    watches = [s for s in report().signals if s.kind == "watch"]
    assert [s.day.date().isoformat() for s in watches] == ["2026-10-04"]
    assert "14 Tage danach" in watches[0].outcome


def test_last_day_stance_waits_without_fresh_signal():
    assert report().stance[0] == "Abwarten"


def test_crowded_longs_block_the_breakout_buy():
    frame = qnt_frame()
    frame["funding"] = 0.08
    r = build_report("QNT", START, END, frame=frame, llm_backend="none")
    assert not [s for s in r.signals if s.kind == "buy"]


def test_loud_social_volume_blocks_the_buy():
    features = add_features(qnt_frame().assign(social_volume=100.0))
    row = features.loc["2026-09-22"].copy()
    row["social_ratio"] = 5.0
    kinds = {s.kind for s in detect_day(row, 0, Thresholds())}
    assert "buy" not in kinds


def test_onchain_streak_gives_early_buy():
    features = add_features(qnt_frame().assign(active_addresses=800.0))
    row = features.loc["2026-09-18"].copy()
    row["address_ratio"] = 1.6
    labels = [s.label for s in detect_day(row, 3, Thresholds())]
    assert "Kauf (früh, On-Chain)" in labels


def test_price_only_still_detects_signals():
    r = build_report("QNT", START, END, frame=qnt_frame(funding=False), llm_backend="none")
    assert {s.kind for s in r.signals} >= {"buy", "sell"}


def test_save_writes_html_and_json(tmp_path):
    html_path, json_path = save_report(report(), tmp_path)
    page = html_path.read_text()
    assert "QNT Sentiment-Check" in page and "<svg" in page and "Verkauf (Euphorie)" in page
    data = json.loads(json_path.read_text())
    assert data["kennzahlen"]["hoch"] == 287.07
    assert data["signale"][0]["day"] == "2026-09-22"


def test_claude_code_backend_uses_plan_not_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    done = subprocess.CompletedProcess([], 0, stdout=json.dumps({"result": "## Fazit\nText", "is_error": False}))
    with mock.patch("shutil.which", return_value="/usr/bin/claude"), \
         mock.patch("subprocess.run", return_value=done) as run:
        r = report(llm_backend="claude-code")
    assert r.narrative == "## Fazit\nText"
    command, env = run.call_args.args[0], run.call_args.kwargs["env"]
    assert command[:2] == ["/usr/bin/claude", "-p"] and "--tools" in command
    assert "ANTHROPIC_API_KEY" not in env
    assert "QNT" in run.call_args.kwargs["input"]
    # Windows' default code page cannot carry the emoji in social posts.
    assert run.call_args.kwargs["encoding"] == "utf-8"


def test_failed_llm_keeps_the_report():
    done = subprocess.CompletedProcess([], 1, stdout="", stderr="not logged in")
    with mock.patch("shutil.which", return_value="/usr/bin/claude"), \
         mock.patch("subprocess.run", return_value=done):
        r = report(llm_backend="claude-code")
    assert r.narrative is None and "not logged in" in r.narrative_note


def test_auto_backend_order(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with mock.patch("shutil.which", return_value=None):
        assert narrative.resolve_backend("auto") == "none"
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert narrative.resolve_backend("auto") == "api"
    with mock.patch("shutil.which", return_value="/usr/bin/claude"):
        assert narrative.resolve_backend("auto") == "claude-code"


def test_window_defaults_to_eight_weeks():
    start, end = parse_window(None, "2026-10-19", 56)
    assert (start, end) == (date(2026, 8, 24), date(2026, 10, 19))
    with pytest.raises(ValueError):
        parse_window("2026-10-19", "2026-10-01", 56)


def test_cli_reports_missing_price_data(capsys):
    from tradingagents.dataflows.vendors.crypto_sentiment import SourceError

    with mock.patch("tradingagents.dataflows.vendors.crypto_sentiment.fetch_price", side_effect=SourceError("offline")):
        assert main(["QNT", "--end", "2026-10-19", "--llm", "none"]) == 1
    assert "offline" in capsys.readouterr().out


# --- sources (mocked HTTP) ---------------------------------------------------

from tradingagents.dataflows.vendors import crypto_sentiment as cs  # noqa: E402


def _ms(day: str, hour: int = 0) -> int:
    return int(pd.Timestamp(f"{day} {hour:02d}:00", tz="UTC").timestamp() * 1000)


def test_bybit_funding_is_a_daily_mean_in_percent():
    rows = [{"fundingRateTimestamp": str(_ms("2026-10-04", h)), "fundingRate": r}
            for h, r in ((0, "-0.003"), (8, "-0.0027"), (16, "-0.0024"))]
    payload = {"retCode": 0, "result": {"list": rows}}
    with mock.patch.object(cs, "_get_json", return_value=payload):
        series = cs.fetch_funding_bybit("QNT", date(2026, 10, 1), date(2026, 10, 5))
    assert series.loc["2026-10-04"] == pytest.approx(-0.27)


def test_fear_greed_trimmed_to_window():
    rows = [{"timestamp": str(_ms(d) // 1000), "value": v}
            for d, v in (("2026-10-18", "40"), ("2026-10-19", "59"), ("2026-10-20", "61"))]
    with mock.patch.object(cs, "_get_json", return_value={"data": rows}):
        series, _ = cs.fetch_fear_greed(date(2026, 10, 18), date(2026, 10, 19))
    assert series.tolist() == [40.0, 59.0]


def test_santiment_without_key_is_reported_missing(monkeypatch):
    monkeypatch.delenv("SANTIMENT_API_KEY", raising=False)
    with pytest.raises(cs.SourceError, match="SANTIMENT_API_KEY"):
        cs.fetch_santiment("QNT", date(2026, 9, 1), date(2026, 10, 1))


def test_missing_sources_listed_in_coverage():
    price = qnt_frame(funding=False)
    fail = mock.Mock(side_effect=cs.SourceError("blocked"))
    with mock.patch.object(cs, "fetch_price", return_value=(price, "Yahoo Finance")), \
         mock.patch.object(cs, "fetch_funding", fail), mock.patch.object(cs, "fetch_open_interest", fail), \
         mock.patch.object(cs, "fetch_perp_volume", fail), mock.patch.object(cs, "fetch_long_share", fail), \
         mock.patch.object(cs, "fetch_fear_greed", fail), mock.patch.object(cs, "fetch_santiment", fail):
        r = build_report("QNT", START, END, llm_backend="none")
    assert r.coverage["Kurs & Volumen"] == "Yahoo Finance"
    assert r.coverage["Funding Rate"].startswith("fehlt")
    assert any(s.kind == "sell" for s in r.signals)


def test_bybit_perp_volume_is_daily_usd_turnover():
    rows = [[str(_ms("2026-09-27")), "99", "374", "95", "286", "4000000", "1100000000"],
            [str(_ms("2026-09-26")), "98", "160", "97", "151", "900000", "120000000"]]
    with mock.patch.object(cs, "_get_json", return_value={"retCode": 0, "result": {"list": rows}}):
        series = cs.fetch_perp_volume_bybit("QNT", date(2026, 9, 26), date(2026, 9, 27))
    assert series.loc["2026-09-27"] == 1.1e9 and series.index.is_monotonic_increasing


def test_perp_volume_jump_counts_toward_the_sell():
    frame = qnt_frame()
    frame["perp_volume"] = frame["volume"] * 2
    features = add_features(frame)
    row = features.loc["2026-10-08"].copy()
    assert row["perp_spot_ratio"] == pytest.approx(1.0)
    row["perp_spot"], row["perp_spot_ratio"] = 6.0, 3.0
    row["rsi"], row["extension"] = 85.0, 1.4
    sells = [s for s in detect_day(row, 0, Thresholds()) if s.kind == "sell"]
    assert sells and any("Perp-Volumen 6.0× Spot" in r for r in sells[0].reasons)


def test_exchange_inflow_blocks_buy_and_counts_toward_sell():
    frame = qnt_frame().assign(exchange_inflow=1000.0, exchange_outflow=1200.0)
    features = add_features(frame)
    row = features.loc["2026-09-22"].copy()
    assert row["outflow_days"] == 5  # outflow beat inflow on every recent day
    buy = [s for s in detect_day(row, 0, Thresholds()) if s.kind == "buy"]
    assert buy and any("Netto-Abfluss" in r for r in buy[0].reasons)
    row["inflow_ratio"] = 4.0
    row["rsi"], row["extension"] = 85.0, 1.4
    signals = detect_day(row, 0, Thresholds())
    assert not [s for s in signals if s.kind == "buy"]
    sells = [s for s in signals if s.kind == "sell"]
    assert sells and any("Börsenzufluss 4.0×" in r for r in sells[0].reasons)


def test_whale_spike_counts_only_after_a_rally():
    row = add_features(qnt_frame().assign(whale_transactions=10.0)).loc["2026-10-08"].copy()
    row["whale_ratio"], row["rsi"], row["extension"] = 4.0, 85.0, 1.4
    row["ret_7d"] = 1.0  # flat week: whales alone are no euphoria
    assert not [s for s in detect_day(row, 0, Thresholds()) if s.kind == "sell"]
    row["ret_7d"] = 1.5
    sells = [s for s in detect_day(row, 0, Thresholds()) if s.kind == "sell"]
    assert sells and any("Wal-Transaktionen 4.0×" in r for r in sells[0].reasons)


def test_bybit_long_share_pages_through_cursor():
    pages = [
        {"retCode": 0, "result": {"list": [{"buyRatio": "0.42", "sellRatio": "0.58", "timestamp": str(_ms("2026-09-22"))}],
                                  "nextPageCursor": "next"}},
        {"retCode": 0, "result": {"list": [{"buyRatio": "0.71", "sellRatio": "0.29", "timestamp": str(_ms("2026-09-27"))}],
                                  "nextPageCursor": ""}},
    ]
    with mock.patch.object(cs, "_get_json", side_effect=pages):
        series = cs.fetch_long_share_bybit("QNT", date(2026, 9, 20), date(2026, 9, 30))
    assert series.tolist() == [0.42, 0.71]


def test_crowd_all_long_blocks_buy_and_counts_toward_sell():
    frame = qnt_frame().assign(long_share=0.45)
    features = add_features(frame)
    row = features.loc["2026-09-22"].copy()
    buy = [s for s in detect_day(row, 0, Thresholds()) if s.kind == "buy"]
    assert buy and any("Long-Anteil 45 % (Crowd skeptisch)" in r for r in buy[0].reasons)
    row["long_share"], row["long_shift"] = 0.74, 0.29
    row["rsi"], row["extension"] = 85.0, 1.4
    signals = detect_day(row, 0, Thresholds())
    assert not [s for s in signals if s.kind == "buy"]
    sells = [s for s in signals if s.kind == "sell"]
    assert sells and any("Long-Anteil 74 %" in r for r in sells[0].reasons)
