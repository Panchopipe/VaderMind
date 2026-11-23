"""Helpers for cleaning assistant text before display and speech."""

from __future__ import annotations

import re
import unicodedata

_STAGE_DIRECTION_PATTERN = re.compile(r"(?<!\*)\*([^*\n]{1,140})\*(?!\*)")
_MARKDOWN_EMPHASIS_PATTERN = re.compile(r"(\*\*|__)(.*?)\1")
_INLINE_CODE_PATTERN = re.compile(r"`([^`\n]+)`")
_EXCESS_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")

# Spoken Chilean rhythm, defined by the user: "Weón, ..." opens a sentence with a
# comma after it, while a weón that closes a phrase goes without a comma before it.
# Piper turns every comma into a pause, so a stray comma makes the word sound staged.
_WEON_WORD = r"(?:we[oó]n|hue[oó]n|huev[oó]n)"
_COMMA_BEFORE_WEON_PATTERN = re.compile(rf",\s+({_WEON_WORD})\b", re.IGNORECASE)
_OPENING_WEON_PATTERN = re.compile(
    rf"(^|[.!?]\s+)({_WEON_WORD})(?=\s+[^\s,.!?])", re.IGNORECASE
)

_STAGE_DIRECTION_KEYWORDS = {
    "acerca",
    "alza",
    "asiente",
    "camina",
    "cruza",
    "extiende",
    "gira",
    "inclina",
    "levanta",
    "mira",
    "observa",
    "pausa",
    "respira",
    "respiracion",
    "rie",
    "sonrie",
    "suspira",
}

def clean_assistant_text(text: str) -> str:
    """Remove stage directions and speech-hostile markup from model output."""

    without_stage_directions = _STAGE_DIRECTION_PATTERN.sub(_remove_stage_direction, text)
    without_emphasis = _MARKDOWN_EMPHASIS_PATTERN.sub(r"\2", without_stage_directions)
    without_inline_code = _INLINE_CODE_PATTERN.sub(r"\1", without_emphasis)
    normalized_spaces = re.sub(r"[ \t]{2,}", " ", without_inline_code)
    normalized_lines = _EXCESS_BLANK_LINES_PATTERN.sub("\n\n", normalized_spaces)
    return _fix_weon_punctuation(normalized_lines.strip())


def _fix_weon_punctuation(text: str) -> str:
    """Apply the spoken Chilean comma rule around weón."""

    without_comma_before = _COMMA_BEFORE_WEON_PATTERN.sub(r" \1", text)
    return _OPENING_WEON_PATTERN.sub(r"\1\2,", without_comma_before)


def _remove_stage_direction(match: re.Match[str]) -> str:
    """Drop action-like italic text while preserving ordinary emphasis."""

    inner_text = match.group(1).strip()
    normalized = _normalize_text(inner_text)
    if any(keyword in normalized for keyword in _STAGE_DIRECTION_KEYWORDS):
        return ""
    return inner_text


def _normalize_text(text: str) -> str:
    """Lowercase and remove accents for robust keyword checks."""

    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(character for character in decomposed if not unicodedata.combining(character))
