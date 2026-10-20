"""Local report page: coin registry, cryptobubbles list, report queue and HTTP routes. No network."""

import io
import json
import time
from unittest import mock

import pytest

from tradingagents.dataflows.vendors import crypto_sentiment as cs
from tradingagents.sentiment_report import web

pytestmark = pytest.mark.unit

BUBBLES = [
    {"symbol": "btc", "name": "Bitcoin", "rank": 1},
    {"symbol": "QNT", "name": "Quant", "rank": 120},
    {"symbol": "ARB", "name": "Arbitrum", "rank": 60, "stable": False},
    {"symbol": "QNT", "name": "Quant Fork", "rank": 900},  # same ticker, smaller coin
    {"symbol": "", "name": "broken", "rank": 5},
    {"symbol": "USDT", "name": "Tether", "rank": 3},  # stable by the known list
    {"symbol": "XYZUSD", "name": "Some Dollar", "rank": 40, "stable": True},  # stable by the flag
]


def test_cryptobubbles_one_entry_per_ticker_by_rank_without_stablecoins():
    coins = cs.parse_cryptobubbles(BUBBLES)
    assert [(c["id"], c["name"], c["rank"]) for c in coins] == [
        ("BTC", "Bitcoin", 1), ("ARB", "Arbitrum", 60), ("QNT", "Quant", 120)]


@pytest.fixture
def library(tmp_path):
    (tmp_path / "QNT_20260824-20261019_sentiment.html").write_text("<p>qnt</p>", encoding="utf-8")
    (tmp_path / "QNT_20260901-20261019_sentiment.html").write_text("<p>qnt2</p>", encoding="utf-8")
    # Written last but ending first: the tree orders by end date, not by file time.
    (tmp_path / "QNT_20260701-20260930_sentiment.html").write_text("<p>old</p>", encoding="utf-8")
    (tmp_path / "notes.html").write_text("x", encoding="utf-8")
    with mock.patch.object(cs, "fetch_coin_universe", return_value=(cs.parse_cryptobubbles(BUBBLES), "test")):
        yield web.Library(tmp_path, llm_backend="none")


def test_state_orders_reports_by_end_date(library):
    library.universe()  # names come from the list for coins only known by their reports
    coins = library.state()["coins"]
    assert [c["id"] for c in coins] == ["QNT"]
    assert coins[0]["name"] == "Quant"
    assert [(r["start"], r["end"]) for r in coins[0]["reports"]] == [
        ("2026-07-01", "2026-09-30"), ("2026-08-24", "2026-10-19"), ("2026-09-01", "2026-10-19")]


def test_universe_leaves_out_analysed_coins(library):
    library.add_coin("arb", "Arbitrum")
    assert [c["id"] for c in library.universe()["coins"]] == ["BTC"]


def test_add_coin_validates_and_refuses_duplicates(library):
    assert library.add_coin(" link ", "Chainlink") == {"id": "LINK", "name": "Chainlink"}
    with pytest.raises(ValueError, match="schon in der Analyse"):
        library.add_coin("QNT", "Quant")
    with pytest.raises(ValueError):
        library.add_coin("../x", "Böse")
    assert json.loads((library.folder / "coins.json").read_text(encoding="utf-8")) == {"LINK": "Chainlink"}


def test_queued_report_runs_and_lands_in_the_folder(library):
    report = mock.Mock(signals=[1, 2], stance=("Abwarten", ""))

    def fake_save(_, folder):
        path = folder / "ARB_20260801-20260926_sentiment.html"
        path.write_text("<p>arb</p>", encoding="utf-8")
        return path, folder / "x.json"

    with mock.patch("tradingagents.sentiment_report.build_report", return_value=report) as build, \
         mock.patch("tradingagents.sentiment_report.save_report", side_effect=fake_save):
        job = library.queue_report("ARB", "2026-08-01", "2026-09-26", 56)
        for _ in range(100):
            if job["status"] in ("fertig", "fehler"):
                break
            time.sleep(0.02)
    assert job["status"] == "fertig" and job["file"] == "ARB_20260801-20260926_sentiment.html"
    assert build.call_args.kwargs["llm_backend"] == "none"


def call(library, method, path, payload=None):
    """One request through the handler, without a socket (the suite blocks them)."""
    handler_class = web.make_handler(library)
    handler = handler_class.__new__(handler_class)
    data = json.dumps(payload).encode() if payload is not None else b""
    handler.rfile, handler.wfile = io.BytesIO(data), io.BytesIO()
    handler.path, handler.command, handler.request_version = path, method, "HTTP/1.1"
    handler.requestline, handler.client_address = f"{method} {path} HTTP/1.1", ("127.0.0.1", 0)
    handler.headers = {"Content-Length": str(len(data))}
    getattr(handler, f"do_{method}")()
    head, _, body = handler.wfile.getvalue().partition(b"\r\n\r\n")
    return int(head.split()[1]), body.decode("utf-8")


def test_routes(library):
    status, page = call(library, "GET", "/")
    assert status == 200 and "Sentiment-Berichte" in page
    assert call(library, "GET", "/reports/QNT_20260824-20261019_sentiment.html") == (200, "<p>qnt</p>")
    assert call(library, "GET", "/reports/notes.html")[0] == 404
    assert call(library, "GET", "/reports/../coins.json")[0] == 404
    status, body = call(library, "POST", "/api/coins", {"id": "ARB", "name": "Arbitrum"})
    assert status == 201 and json.loads(body)["job"] is None
    status, body = call(library, "POST", "/api/coins", {"id": "ARB", "name": "Arbitrum", "run": False})
    assert status == 400 and "schon" in json.loads(body)["error"]
    universe = json.loads(call(library, "GET", "/api/universe")[1])
    assert [c["id"] for c in universe["coins"]] == ["BTC"]
    state = json.loads(call(library, "GET", "/api/state")[1])
    assert [c["id"] for c in state["coins"]] == ["ARB", "QNT"]


def test_unreachable_coin_list_is_an_error_message(tmp_path):
    with mock.patch.object(cs, "fetch_coin_universe", side_effect=cs.SourceError("offline")):
        status, body = call(web.Library(tmp_path), "GET", "/api/universe")
    assert status == 502 and "offline" in json.loads(body)["error"]


def test_report_with_chosen_period(library):
    with mock.patch.object(library, "_queue"):
        status, body = call(library, "POST", "/api/reports",
                            {"id": "QNT", "start": "2026-07-01", "end": "2026-09-30", "days": 60})
        assert status == 201
        assert (json.loads(body)["job"]["start"], json.loads(body)["job"]["end"]) == ("2026-07-01", "2026-09-30")
        status, body = call(library, "POST", "/api/reports", {"id": "QNT", "start": "2026-10-01", "end": "2026-09-30"})
        assert status == 400 and "Start" in json.loads(body)["error"]


def test_state_carries_creation_time_and_version(library):
    from tradingagents.sentiment_report import REPORT_VERSION

    current = library.folder / "QNT_20260101-20260201_sentiment.html"
    current.write_text(f'<head><meta name="report-created" content="2026-10-20T11:46:00+02:00">'
                       f'<meta name="report-version" content="{REPORT_VERSION} (abc1234)"></head>',
                       encoding="utf-8")
    reports = {r["file"]: r for r in library.state()["coins"][0]["reports"]}
    assert reports[current.name]["created"] == "2026-10-20T11:46:00+02:00"
    assert reports[current.name]["version"] == f"{REPORT_VERSION} (abc1234)"
    assert not reports[current.name]["outdated"]
    old = reports["QNT_20260901-20261019_sentiment.html"]  # written before reports were stamped
    assert old["created"] == "" and old["outdated"]
    assert library.state()["version"].startswith(REPORT_VERSION)


def test_coin_overview_scores_signals_up_to_today(library):
    import pandas as pd

    def report_json(name, signals, days):
        (library.folder / name).write_text(json.dumps({
            "signale": signals,
            "tageswerte": [{"datum": d, "close": c} for d, c in days],
        }), encoding="utf-8")

    buy = {"kind": "buy", "label": "Kauf", "day": "2026-09-22", "end_day": None, "close": 100.0, "reasons": ["Ausbruch"]}
    sell = {"kind": "sell", "label": "Verkauf", "day": "2026-09-27", "end_day": "2026-09-28", "close": 200.0, "reasons": []}
    watch = {"kind": "watch", "label": "Beobachten", "day": "2026-10-04", "close": 150.0}
    report_json("QNT_20260824-20261019_sentiment.json", [buy, sell, watch], [("2026-09-22", 100.0)])
    report_json("QNT_20260901-20261019_sentiment.json", [buy], [])  # the same buy again
    prices = pd.DataFrame({"close": [90.0, 200.0, 220.0, 120.0], "high": [95.0, 210.0, 230.0, 125.0],
                           "low": [85.0, 190.0, 200.0, 110.0]},
                          index=pd.to_datetime(["2026-09-23", "2026-09-27", "2026-09-30", "2026-10-20"]))
    with mock.patch.object(cs, "fetch_price", return_value=(prices, "Yahoo Finance")) as fetch:
        status, body = call(library, "GET", "/api/coins/qnt/overview")
        library.overview("QNT")  # cached
    assert status == 200 and fetch.call_count == 1
    data = json.loads(body)
    assert len(data["reports"]) == 3
    assert data["prices"] == {"source": "Yahoo Finance", "until": "2026-10-20", "close": 120.0}
    first, second = data["signals"]  # the watch is left out, the double buy counted once
    assert first["kind"] == "buy" and len(first["reports"]) == 2
    assert first["max_drawdown_pct"] == pytest.approx(-15) and first["drawdown_day"] == "2026-09-23"
    assert first["max_runup_pct"] == pytest.approx(130) and first["current_pct"] == pytest.approx(20)
    # The sell as a short from 200: the 230 high on 30.09 is the drawdown, 110 the best low.
    assert second["max_drawdown_pct"] == pytest.approx(-15) and second["drawdown_day"] == "2026-09-30"
    assert second["max_runup_pct"] == pytest.approx(45) and second["current_pct"] == pytest.approx(40)


def test_coin_overview_falls_back_to_saved_closes(library):
    (library.folder / "QNT_20260824-20261019_sentiment.json").write_text(json.dumps({
        "signale": [{"kind": "buy", "label": "Kauf", "day": "2026-09-22", "close": 100.0}],
        "tageswerte": [{"datum": "2026-09-22", "close": 100.0}, {"datum": "2026-09-23", "close": 80.0},
                       {"datum": "2026-10-19", "close": 150.0}],
    }), encoding="utf-8")
    with mock.patch.object(cs, "fetch_price", side_effect=cs.SourceError("offline")):
        data = library.overview("QNT")
    assert data["prices"]["until"] == "2026-10-19" and "nicht abrufbar" in data["prices"]["source"]
    assert data["signals"][0]["max_drawdown_pct"] == pytest.approx(-20)
    assert data["signals"][0]["current_pct"] == pytest.approx(50)
    assert call(library, "GET", "/api/coins/..%2F/overview")[0] == 400
