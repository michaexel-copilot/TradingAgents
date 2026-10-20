"""python -m tradingagents.sentiment_report QNT --days 56 [--start ... --end ...]"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

from tradingagents.dataflows.vendors.crypto_sentiment import SourceError
from tradingagents.sentiment_report import build_report, save_report
from tradingagents.sentiment_report.narrative import BACKENDS


def parse_window(start: str | None, end: str | None, days: int) -> tuple[date, date]:
    """The report window: ``end`` defaults to today, ``start`` to ``days`` before it."""
    end_day = date.fromisoformat(end) if end else date.today()
    start_day = date.fromisoformat(start) if start else end_day - timedelta(days=days)
    if start_day >= end_day:
        raise ValueError("Der Start muss vor dem Ende liegen")
    if end_day > date.today():
        raise ValueError("Das Ende darf nicht in der Zukunft liegen")
    return start_day, end_day


def default_out_dir() -> Path:
    from tradingagents.default_config import DEFAULT_CONFIG

    return Path(DEFAULT_CONFIG["results_dir"]) / "sentiment"


def run(coin, start, end, days, llm, model, santiment_slug, out_dir, say=print) -> int:
    try:
        start_day, end_day = parse_window(start, end, days)
    except ValueError as exc:
        say(f"Fehler: {exc}")
        return 1
    say(f"{coin.upper()}: {start_day} bis {end_day}, Daten werden geladen ...")
    try:
        report = build_report(coin, start_day, end_day, llm_backend=llm, llm_model=model,
                              santiment_slug=santiment_slug)
    except (SourceError, ValueError) as exc:
        say(f"Fehler: {exc}")
        return 1
    for label, origin in report.coverage.items():
        say(f"  {label}: {origin}")
    say(f"  KI-Text: {report.narrative_note}")
    html_path, json_path = save_report(report, Path(out_dir) if out_dir else default_out_dir())
    say(f"{len(report.signals)} Signale, Lage: {report.stance[0]} ({report.stance[1]})")
    say(f"Bericht: {html_path}\nDaten:   {json_path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tradingagents.sentiment_report",
                                     description="Sentiment-Bericht mit Kauf- und Verkaufssignalen für einen Coin")
    parser.add_argument("coin", help="Ticker, z.B. QNT, ARB, LINK")
    parser.add_argument("--start", help="Erster Tag, YYYY-MM-DD")
    parser.add_argument("--end", help="Letzter Tag, YYYY-MM-DD (Standard: heute)")
    parser.add_argument("--days", type=int, default=56, help="Tage zurück, wenn --start fehlt (Standard: 56 = 8 Wochen)")
    parser.add_argument("--llm", choices=BACKENDS, default=os.environ.get("SENTIMENT_LLM", "auto"),
                        help="KI-Text: claude-code (eigenes Abo), api (ANTHROPIC_API_KEY), none, auto")
    parser.add_argument("--model", help="Modell, z.B. sonnet/opus (claude-code) oder claude-sonnet-5 (api)")
    parser.add_argument("--santiment-slug", help="Santiment-Projektname, falls der Ticker nicht eindeutig ist")
    parser.add_argument("--out", help="Ausgabeordner (Standard: <results_dir>/sentiment)")
    a = parser.parse_args(argv)
    return run(a.coin, a.start, a.end, a.days, a.llm, a.model, a.santiment_slug, a.out)


if __name__ == "__main__":
    sys.exit(main())
