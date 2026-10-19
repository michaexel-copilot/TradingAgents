"""The written assessment in the report, from one of three backends.

  claude-code  The local Claude Code CLI in print mode (``claude -p``), signed in
               with your own Pro/Max plan. Meant for your own, personal runs:
               Anthropic's terms allow scripting Claude Code for yourself, not
               offering your plan's login to other people or products. The
               environment's ANTHROPIC_API_KEY is removed for the call, because
               ``claude -p`` would otherwise bill that key instead of the plan.
  api          The Anthropic API with ANTHROPIC_API_KEY (billed per token). Use
               this when the program runs unattended for others, on a server, or
               is passed on.
  none         No model: the report keeps only the computed signals.

``auto`` picks claude-code when the ``claude`` binary is installed, then api
when a key is set, else none.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

BACKENDS = ("auto", "claude-code", "api", "none")
DEFAULT_API_MODEL = "claude-sonnet-5"
DEFAULT_CLI_MODEL = "sonnet"

SYSTEM = (
    "Du bist ein nüchterner Krypto-Analyst. Du schreibst auf Deutsch, in ganzen Sätzen, "
    "ohne Hype. Du stützt dich nur auf die gelieferten Daten und erfindest keine Zahlen, "
    "News oder Posts. Wo Daten fehlen, sagst du das. Keine Anlageberatung."
)

INSTRUCTIONS = """Unten stehen die berechneten Daten eines Sentiment-Berichts für {coin} ({start} bis {end}).
Schreibe die Einschätzung in Markdown mit genau diesen Abschnitten:

## Fazit
Zwei bis drei Sätze: Gab es im Zeitraum gute Kaufsignale, wann, und wie ist die Lage am letzten Tag?

## Einordnung der Signale
Pro Signal ein kurzer Absatz: was Kurs, Volumen, Funding, Open Interest und Social-Daten zeigten und ob das Signal funktioniert hat. Fasse ähnliche Signale zusammen.

## Worauf jetzt achten
Drei bis fünf Stichpunkte mit konkreten Kursmarken oder Datenschwellen aus den Daten.

## Grenzen der Daten
Ein bis drei Sätze zu fehlenden Quellen und was das für die Aussagekraft heißt.

Daten (JSON):
{payload}
"""


class NarrativeError(Exception):
    pass


def resolve_backend(backend: str) -> str:
    if backend not in BACKENDS:
        raise ValueError(f"Unbekanntes LLM-Backend {backend!r}, erlaubt: {', '.join(BACKENDS)}")
    if backend != "auto":
        return backend
    if shutil.which("claude"):
        return "claude-code"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api"
    return "none"


def build_prompt(coin: str, start: str, end: str, payload: dict) -> str:
    return INSTRUCTIONS.format(
        coin=coin, start=start, end=end, payload=json.dumps(payload, ensure_ascii=False, indent=1, default=str)
    )


def via_claude_code(prompt: str, model: str | None = None, timeout: float = 300.0) -> str:
    binary = shutil.which("claude")
    if not binary:
        raise NarrativeError("Claude Code CLI nicht gefunden (npm install -g @anthropic-ai/claude-code)")
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    command = [
        binary, "-p",
        "--model", model or DEFAULT_CLI_MODEL,
        "--system-prompt", SYSTEM,
        "--tools", "",
        "--max-turns", "1",
        "--output-format", "json",
        "--no-session-persistence",
    ]
    try:
        done = subprocess.run(command, input=prompt, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise NarrativeError("claude -p hat nicht rechtzeitig geantwortet") from None
    try:
        result = json.loads(done.stdout)
    except json.JSONDecodeError:
        detail = (done.stderr or done.stdout).strip()[:300]
        raise NarrativeError(f"claude -p fehlgeschlagen (Exit {done.returncode}): {detail}") from None
    if result.get("is_error") or not result.get("result"):
        raise NarrativeError(f"claude -p: {result.get('result') or result.get('subtype') or 'leere Antwort'}")
    return result["result"].strip()


def via_api(prompt: str, model: str | None = None) -> str:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise NarrativeError("ANTHROPIC_API_KEY ist nicht gesetzt")
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(model=model or DEFAULT_API_MODEL, max_tokens=4000, timeout=300)
    try:
        message = llm.invoke([("system", SYSTEM), ("human", prompt)])
    except Exception as exc:  # the SDK raises many error types; the report goes on without text
        raise NarrativeError(f"Anthropic API: {type(exc).__name__}: {exc}") from None
    content = message.content
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content).strip()


def write_narrative(backend: str, prompt: str, model: str | None = None) -> tuple[str | None, str]:
    """The narrative and a note on how it was made; the narrative is None on 'none' or failure."""
    backend = resolve_backend(backend)
    if backend == "none":
        return None, "ohne KI-Text (Backend none)"
    try:
        if backend == "claude-code":
            return via_claude_code(prompt, model), "Claude Code (eigenes Abo, claude -p)"
        return via_api(prompt, model), f"Anthropic API ({model or DEFAULT_API_MODEL})"
    except NarrativeError as exc:
        return None, f"KI-Text fehlgeschlagen: {exc}"
