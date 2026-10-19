"""Local web page over the saved sentiment reports.

    uv run tradingagents sentiment-web        # http://127.0.0.1:8765

Left a tree (first letter, coin, report end date), right the selected report,
above it a form that adds a coin from the cryptobubbles.net list. A new coin
opens on an empty page where its first report is started for a chosen period.
Coins are kept in ``coins.json`` next to the reports; reports run one after
another in a background thread. Listens on localhost only.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from datetime import date, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tradingagents.dataflows.vendors import crypto_sentiment as sources

logger = logging.getLogger(__name__)

COIN_ID = re.compile(r"^[A-Z0-9]{1,15}$")
REPORT_FILE = re.compile(r"^([A-Z0-9]{1,15})_(\d{8})-(\d{8})_sentiment\.html$")
UNIVERSE_TTL_SECONDS = 6 * 3600
PAGE = Path(__file__).with_name("static") / "index.html"


def _day(stamp: str) -> str:
    return datetime.strptime(stamp, "%Y%m%d").date().isoformat()


class Library:
    """The coins and reports in one folder, plus the queue of reports to run."""

    def __init__(self, folder: Path, llm_backend: str = "auto", llm_model: str | None = None):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.llm_backend, self.llm_model = llm_backend, llm_model
        self._lock = threading.Lock()
        self._universe: tuple[float, list[dict], str] | None = None
        self.jobs: list[dict] = []
        self._queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._work, daemon=True).start()

    # --- coins -------------------------------------------------------------

    @property
    def _coins_file(self) -> Path:
        return self.folder / "coins.json"

    def _registered(self) -> dict[str, str]:
        try:
            return json.loads(self._coins_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def add_coin(self, coin_id: str, name: str) -> dict:
        coin_id, name = coin_id.strip().upper(), name.strip()
        if not COIN_ID.match(coin_id):
            raise ValueError("Die Coin-ID besteht aus 1 bis 15 Buchstaben oder Ziffern, z.B. QNT")
        if not name or len(name) > 80:
            raise ValueError("Bitte einen Coin-Namen mit höchstens 80 Zeichen angeben")
        with self._lock:
            coins = self._registered()
            if coin_id in coins or coin_id in self._report_coins():
                raise ValueError(f"{coin_id} ist schon in der Analyse")
            coins[coin_id] = name
            self._coins_file.write_text(json.dumps(coins, ensure_ascii=False, indent=1), encoding="utf-8")
        return {"id": coin_id, "name": name}

    def _reports(self) -> list[dict]:
        found = []
        for path in self.folder.glob("*_sentiment.html"):
            match = REPORT_FILE.match(path.name)
            if match:
                found.append({"coin": match[1], "start": _day(match[2]), "end": _day(match[3]),
                              "file": path.name})
        return found

    def _report_coins(self) -> set[str]:
        return {r["coin"] for r in self._reports()}

    def state(self) -> dict:
        registered = self._registered()
        names = {c["id"]: c["name"] for c in (self._universe[1] if self._universe else [])}
        reports = self._reports()
        coins = {}
        for coin_id in set(registered) | {r["coin"] for r in reports}:
            coins[coin_id] = {"id": coin_id, "name": registered.get(coin_id) or names.get(coin_id, ""),
                              "reports": []}
        for report in reports:
            coins[report["coin"]]["reports"].append(report)
        for coin in coins.values():
            coin["reports"].sort(key=lambda r: (r["end"], r["start"]), reverse=True)
        return {"coins": sorted(coins.values(), key=lambda c: c["id"]), "jobs": self.jobs[-20:]}

    def universe(self) -> dict:
        """The cryptobubbles list without the coins already in the analysis."""
        if self._universe is None or time.time() - self._universe[0] > UNIVERSE_TTL_SECONDS:
            coins, origin = sources.fetch_coin_universe()
            self._universe = (time.time(), coins, origin)
        taken = set(self._registered()) | self._report_coins()
        return {"source": self._universe[2], "coins": [c for c in self._universe[1] if c["id"] not in taken]}

    # --- report jobs -------------------------------------------------------

    def queue_report(self, coin_id: str, start: str | None, end: str | None, days: int) -> dict:
        from tradingagents.sentiment_report.__main__ import parse_window

        coin_id = coin_id.strip().upper()
        if not COIN_ID.match(coin_id):
            raise ValueError("Ungültige Coin-ID")
        first, last = parse_window(start or None, end or None, days)
        job = {"id": len(self.jobs) + 1, "coin": coin_id, "start": first.isoformat(), "end": last.isoformat(),
               "status": "wartet", "message": ""}
        self.jobs.append(job)
        self._queue.put(job)
        return job

    def _work(self):
        while True:
            job = self._queue.get()
            job["status"] = "läuft"
            try:
                from tradingagents.sentiment_report import build_report, save_report

                report = build_report(job["coin"], date.fromisoformat(job["start"]), date.fromisoformat(job["end"]),
                                      llm_backend=self.llm_backend, llm_model=self.llm_model)
                html_path, _ = save_report(report, self.folder)
                job.update(status="fertig", file=html_path.name,
                           message=f"{len(report.signals)} Signale, Lage: {report.stance[0]}")
            except Exception as exc:  # a failed report is shown on the page, the server keeps running
                logger.warning("Bericht %s fehlgeschlagen: %s", job["coin"], exc)
                job.update(status="fehler", message=str(exc))


def make_handler(library: Library):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the console quiet
            logger.debug(fmt, *args)

        def _send(self, status, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status=HTTPStatus.OK):
            self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(library.state())
            elif path == "/api/universe":
                try:
                    self._json(library.universe())
                except sources.SourceError as exc:
                    self._json({"error": f"Coin-Liste nicht abrufbar: {exc}"}, HTTPStatus.BAD_GATEWAY)
            elif path.startswith("/reports/") and REPORT_FILE.match(path.removeprefix("/reports/")):
                file = library.folder / path.removeprefix("/reports/")
                if file.is_file():
                    self._send(HTTPStatus.OK, file.read_bytes(), "text/html; charset=utf-8")
                else:
                    self._json({"error": "Bericht nicht gefunden"}, HTTPStatus.NOT_FOUND)
            else:
                self._json({"error": "Nicht gefunden"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(min(length, 10_000)) or b"{}")
                if self.path == "/api/coins":
                    coin = library.add_coin(str(body.get("id", "")), str(body.get("name", "")))
                    job = library.queue_report(coin["id"], None, None, int(body.get("days") or 56)) \
                        if body.get("run") else None
                    self._json({"coin": coin, "job": job}, HTTPStatus.CREATED)
                elif self.path == "/api/reports":
                    job = library.queue_report(str(body.get("id", "")), body.get("start"), body.get("end"),
                                               int(body.get("days") or 56))
                    self._json({"job": job}, HTTPStatus.CREATED)
                else:
                    self._json({"error": "Nicht gefunden"}, HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    return Handler


def serve(folder: Path, port: int = 8765, open_browser: bool = True, llm_backend: str = "auto",
          llm_model: str | None = None, say=print) -> None:
    library = Library(folder, llm_backend, llm_model)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(library))
    url = f"http://127.0.0.1:{port}"
    say(f"Berichte-Webseite: {url}  (Ordner: {folder})  Beenden mit Strg+C")
    if open_browser:
        import webbrowser

        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
