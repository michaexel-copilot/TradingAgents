"""Re-run every report made by an older software version, for the same coin and period.

    uv run tradingagents sentiment-refresh            # all outdated reports
    uv run tradingagents sentiment-refresh --dry-run  # only list them
    uv run tradingagents sentiment-refresh NEAR QNT   # only these coins

Reports run one after another and replace their files; a failed one keeps its old
file and the run goes on. Ctrl+C stops after the report in progress.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

from tradingagents.sentiment_report.web import Library


def outdated_reports(folder: Path, coins: list[str] | None = None) -> list[dict]:
    wanted = {c.upper() for c in coins or []}
    reports = [r for r in Library.scan(folder) if r["outdated"] and (not wanted or r["coin"] in wanted)]
    return sorted(reports, key=lambda r: (r["coin"], r["end"], r["start"]))


def refresh(folder: Path, *, llm_backend: str = "auto", llm_model: str | None = None,
            coins: list[str] | None = None, dry_run: bool = False, pause: float = 2.0, say=print) -> int:
    """Re-run the outdated reports; returns the number that failed."""
    from tradingagents.sentiment_report import build_report, save_report, software_version

    todo = outdated_reports(folder, coins)
    if not todo:
        say(f"Alle Berichte in {folder} sind aktuell (Version {software_version()}).")
        return 0
    say(f"{len(todo)} veraltete Berichte in {folder}, aktuelle Version {software_version()}:")
    if dry_run:
        for r in todo:
            say(f"  {r['coin']} {r['start']} bis {r['end']} (Version {r['version'] or 'ohne Angabe'})")
        return 0
    failed = 0
    for number, r in enumerate(todo, 1):
        say(f"[{number}/{len(todo)}] {r['coin']} {r['start']} bis {r['end']} …")
        try:
            report = build_report(r["coin"], date.fromisoformat(r["start"]), date.fromisoformat(r["end"]),
                                  llm_backend=llm_backend, llm_model=llm_model)
            save_report(report, folder)
            say(f"    fertig: {len(report.signals)} Signale, Lage: {report.stance[0]}")
        except KeyboardInterrupt:
            say("Abgebrochen. Die bisher aktualisierten Berichte sind gespeichert.")
            return failed + len(todo) - number + 1
        except Exception as exc:  # one failing coin must not stop the others
            failed += 1
            say(f"    Fehler, alter Bericht bleibt: {exc}")
        if pause and number < len(todo):
            time.sleep(pause)  # gentle on the free data APIs
    say(f"Fertig: {len(todo) - failed} aktualisiert, {failed} fehlgeschlagen.")
    return failed
