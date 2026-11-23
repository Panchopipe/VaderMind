"""Fixed phrases and error metrics for comparing speech-to-text models."""

from __future__ import annotations

from dataclasses import dataclass
import re
from statistics import median
import unicodedata


@dataclass(slots=True, frozen=True)
class SttPhrase:
    number: int
    kind: str
    text: str


STT_PHRASES: tuple[SttPhrase, ...] = (
    SttPhrase(1, "spotify", "Pon música de Los Prisioneros."),
    SttPhrase(2, "spotify", "Sube el volumen a sesenta."),
    SttPhrase(3, "spotify", "Pasa a la siguiente canción."),
    SttPhrase(4, "spotify_tv", "Reproduce mi playlist de trote en el televisor."),
    SttPhrase(5, "tv", "Enciende el televisor y abre Netflix."),
    SttPhrase(6, "tv", "Cambia al canal trece."),
    SttPhrase(7, "tv", "Pon el HDMI dos."),
    SttPhrase(8, "tv", "Apaga la tele en treinta minutos."),
    SttPhrase(9, "informacion", "¿Cómo va a estar el clima mañana en Santiago?"),
    SttPhrase(10, "informacion", "¿Qué hora es en Tokio?"),
    SttPhrase(11, "canon", "Lord Vader, ¿qué opinas de la Alianza Rebelde?"),
    SttPhrase(12, "canon", "¿Por qué te pasaste al lado oscuro?"),
    SttPhrase(13, "chileno", "Oye weón, ¿cachai lo que pasó ayer?"),
    SttPhrase(14, "chileno", "Ya po, ponte algo de Mon Laferte al tiro."),
    SttPhrase(15, "chileno", "Pucha, se cortó la música otra vez."),
    # Added on the Pi (16 Oct 2026): everyday fast Chilean that failed there.
    # The Windows recordings stop at 15, so compare platforms on 1-15.
    SttPhrase(16, "chileno_natural", "¿Qué vai a hacer el fin de semana?"),
    SttPhrase(17, "chileno_natural", "¿Cómo te hai sentido estos días?"),
    SttPhrase(18, "chileno_natural", "Ya po, ¿cuándo vai a poner música?"),
)

# Spellings that Vader's model reads the same way; they are not real errors.
_EQUIVALENT_WORDS = {
    "huevon": "weon",
    "hueon": "weon",
    "guevon": "weon",
    "wn": "weon",
    "cachay": "cachai",
    "pos": "po",
    "poh": "po",
    "pues": "po",
    "tokyo": "tokio",
}

_UNITS = (
    "cero uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece "
    "catorce quince dieciseis diecisiete dieciocho diecinueve veinte veintiuno "
    "veintidos veintitres veinticuatro veinticinco veintiseis veintisiete "
    "veintiocho veintinueve"
).split()
_TENS = {3: "treinta", 4: "cuarenta", 5: "cincuenta", 6: "sesenta",
         7: "setenta", 8: "ochenta", 9: "noventa"}


def _number_words(value: int) -> str:
    """Spell 0-100 in Spanish so '60' and 'sesenta' compare as equal."""

    if value < 30:
        return _UNITS[value]
    if value == 100:
        return "cien"
    tens, unit = divmod(value, 10)
    return _TENS[tens] + (f" y {_UNITS[unit]}" if unit else "")


def normalize_transcript(text: str) -> list[str]:
    """Lowercase, drop accents and punctuation, spell numbers, unify slang."""

    decomposed = unicodedata.normalize("NFD", text.lower())
    plain = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    plain = re.sub(
        r"\d+",
        lambda match: _number_words(int(match.group()))
        if int(match.group()) <= 100
        else match.group(),
        plain,
    )
    words = re.findall(r"[a-z0-9]+", plain)
    return [_EQUIVALENT_WORDS.get(word, word) for word in words]


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return (word edits, reference words) using Levenshtein distance on words."""

    ref = normalize_transcript(reference)
    hyp = normalize_transcript(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    return previous[-1], len(ref)


@dataclass(slots=True, frozen=True)
class SttRow:
    model: str
    phrase_number: int
    reference: str
    hypothesis: str | None
    seconds: float | None
    cost_usd: float | None
    error: str | None = None


def summarize_model(rows: list[SttRow]) -> dict[str, float]:
    """Aggregate word error rate, perfect phrases, latency and cost."""

    done = [row for row in rows if row.hypothesis is not None]
    edits = words = perfect = 0
    for row in done:
        row_edits, row_words = word_errors(row.reference, row.hypothesis or "")
        edits += row_edits
        words += row_words
        perfect += row_edits == 0
    seconds = sorted(row.seconds for row in done if row.seconds is not None)
    costs = [row.cost_usd for row in done if row.cost_usd is not None]
    summary: dict[str, float] = {
        "frases": len(rows),
        "fallidas": len(rows) - len(done),
        "error_palabras_pct": round(100 * edits / words, 1) if words else 0.0,
        "frases_perfectas": perfect,
        "costo_total_usd": round(sum(costs), 6),
    }
    if seconds:
        summary["latencia_mediana_s"] = round(median(seconds), 2)
        summary["latencia_max_s"] = round(seconds[-1], 2)
    return summary
