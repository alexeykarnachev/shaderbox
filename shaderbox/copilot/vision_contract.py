import re
from dataclasses import dataclass

# The vision eye's ASK contract: the prompt text that DEMANDS the final line, the parser that
# reads it, and the strip that removes it from every model-facing string. A neutral leaf so both
# `llm/openrouter.py` (which writes the prompt) and `backend.py` (which parses + strips) can own
# one definition — backend deliberately imports nothing from `llm/`.

ASK_MET = "met"
ASK_NOT_MET = "not-met"
ASK_UNCLEAR = "unclear"

ASK_CONTRACT_INSTRUCTION = (
    "ONLY when a 'Look for' hint was given (no hint = no such line, end with the baseline read): "
    "THE LAST LINE of your whole reply must be exactly `ASK: met`, `ASK: not-met` or "
    "`ASK: unclear` — an evidence-gated report about the look_for content ONLY, never about "
    "quality or doneness: `met` = you can point at the pixels that show it; `not-met` = you can "
    "see the frame and the named thing is not on it; `unclear` = it is not decidable from a "
    "single still frame — a RELATIVE ask ('darker', 'slower', 'more'), a NON-VISUAL ask (a "
    "uniform value, a rename), or the non-visual part of a compound ask. Anything you cannot "
    "decide from the frame is `unclear`, NEVER `not-met`. Nothing else on that line and nothing "
    "after it."
)

# Loose = any line the eye labelled ASK (a garbled label is stripped too, so no raw done-ness
# wording can reach the model); strict = the demanded format, CRLF-tolerant.
_ASK_LOOSE = re.compile(r"^[ \t]*ASK[ \t]*:.*$", re.MULTILINE | re.IGNORECASE)
_ASK_STRICT = re.compile(
    rf"^[ \t]*ASK[ \t]*:[ \t]*({ASK_NOT_MET}|{ASK_MET}|{ASK_UNCLEAR})[ \t]*\.?[ \t\r]*$",
    re.MULTILINE | re.IGNORECASE,
)
# The gap a mid-body strip leaves behind.
_BLANK_RUN = re.compile(r"\n[ \t\r]*\n([ \t\r]*\n)+")


def _last(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    # The contract puts the line LAST, so a stray earlier mention must never outvote it.
    match: re.Match[str] | None = None
    for match in pattern.finditer(text):
        pass
    return match


@dataclass(frozen=True)
class VisionUsage:
    # A vision call's billed usage. Cost folds into the turn stats; the token counts ride the
    # per-look trace event only (`reply_tokens` means the MAIN model's reply).
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def find_ask_line(text: str) -> str:
    match = _last(_ASK_LOOSE, text)
    return match.group(0).strip() if match is not None else ""


def parse_ask_verdict(text: str) -> str:
    # A missing or garbled line reads as `unclear` — the eye never gets to imply not-met by
    # failing the format.
    match = _last(_ASK_STRICT, text)
    return match.group(1).lower() if match is not None else ASK_UNCLEAR


def strip_ask_line(text: str) -> str:
    return _BLANK_RUN.sub("\n\n", _ASK_LOOSE.sub("", text)).strip()
