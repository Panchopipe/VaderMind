"""Fixed questions and text metrics to compare Vader persona prompts."""

from __future__ import annotations

from dataclasses import dataclass
import re
import statistics
import unicodedata

PERSONA_QUESTIONS: tuple[str, ...] = (
    "¿Cómo estás, Vader?",
    "¿Qué opinas de los lunes?",
    "Dame un consejo para una entrevista de trabajo.",
    "¿Qué me recomiendas cenar hoy?",
    "Estoy cansado, fue un día largo.",
    "Cuéntame de tu relación con Obi-Wan.",
    "¿Eres una inteligencia artificial?",
    "Eres un pésimo líder, el Imperio perdió por tu culpa.",
    "Explícame paso a paso cómo preparar un buen café.",
    "Cuéntame un chiste.",
    "¿Qué piensas del amor?",
    "Hoy me ascendieron en el trabajo.",
    "¿Por qué el cielo es azul?",
    "¿Extrañas a Padmé?",
    "What do you think about the Rebellion?",
)

_FILLER_PATTERNS = {
    "imperio": re.compile(r"\bimperi(o|al|ales)\b"),
    "debilidad": re.compile(r"\bdebil(es|idad|idades)?\b"),
}
_PHYSICAL = re.compile(
    r"\b(respir\w*|suspir\w*|exhal\w*|inhal\w*|gesto\w*|se levanta|levanto la mano"
    r"|cierra el puno|aprieta el puno|me acerco|se acerca)\b"
)
_MARKDOWN = re.compile(r"(\*\*|__|^\s*[-*#]\s|\[[^\]]+\]\()", re.MULTILINE)
_BREAKS_CHARACTER = re.compile(
    r"\b(soy (una|un) (ia|inteligencia artificial|asistente|modelo)"
    r"|modelo de lenguaje|asistente virtual)\b"
)
CHILEAN_WORDS: frozenset[str] = frozenset(
    {
        "po", "cachai", "cachay", "cacho", "cachar", "weon", "weona", "weones",
        "huevon", "hueon", "wea", "weas", "huea", "hueas", "wn", "altiro",
        "bacan", "fome", "pega", "luca", "lucas", "filo", "pulento", "cuatico",
        "carrete", "pololo", "polola", "guagua", "cabro", "cabros", "choro",
        "chela", "copete", "pucha", "chucha", "caleta", "penca", "flaite",
        "chanta", "sapo", "quiubo", "sipo", "nopo", "harto", "cuea", "culiao",
        "conchetumare", "ctm", "wena", "weno", "mina", "cabreado", "taco",
        "raja", "pichintun", "achaplinarse", "apitutado", "pituto", "fondo",
    }
)
CHILEAN_SWEARS: frozenset[str] = frozenset(
    {
        "weon", "weona", "weones", "huevon", "hueon", "wea", "weas", "huea",
        "hueas", "wn", "chucha", "culiao", "conchetumare", "ctm", "mierda",
        "puta", "cuea",
    }
)
# Chilean spoken rhythm: "Weón, ..." opening a sentence and "... weón." closing
# one. A comma before a closing weon, or weon wrapped in commas, sounds staged.
_WEON = re.compile(r"\b(weon|hueon|huevon)\b")
_WEON_BAD_PUNCTUATION = (
    re.compile(r",\s*(weon|hueon|huevon)\s*[,.!?]"),
    re.compile(r"(?:^|[.!?]\s+)[¿¡]?(weon|hueon|huevon)\b(?!,)"),
)
_CHILEAN_VOSEO = re.compile(r"\b\w{2,}(ai|ais|is)\b")
_VOSEO_FALSE_POSITIVES = frozenset(
    {"pais", "mais", "jamais", "paris", "crisis", "analisis", "tesis", "dosis",
     "oasis", "gratis", "lapis", "iris", "tenis", "chis", "hawaii", "bonsai",
     "samurai", "brasis", "genesis", "enfasis", "sintesis", "hipotesis"}
)


@dataclass(frozen=True, slots=True)
class AnswerMetrics:
    """Measurable form of one spoken answer."""

    characters: int
    sentences: int
    ends_with_question: bool
    filler_words: int
    physical_descriptions: int
    markdown_marks: int
    breaks_character: bool
    chilean_terms: int
    swears: int
    weon_uses: int = 0
    weon_bad_punctuation: int = 0


def _fold(text: str) -> str:
    """Lowercase and strip accents so word lists stay ASCII."""

    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def count_sentences(text: str) -> int:
    """Count sentences using terminal punctuation, with a floor of one."""

    parts = [part for part in re.split(r"[.!?]+(?:\s+|$)", text.strip()) if part.strip()]
    return max(1, len(parts)) if text.strip() else 0


def measure_answer(text: str) -> AnswerMetrics:
    """Measure length, forbidden habits and Chilean vocabulary in one answer."""

    clean = text.strip()
    folded = _fold(clean)
    words = re.findall(r"[a-z]+", folded)
    voseo = [
        match.group(0)
        for match in _CHILEAN_VOSEO.finditer(folded)
        if match.group(0) not in _VOSEO_FALSE_POSITIVES
    ]
    return AnswerMetrics(
        characters=len(clean),
        sentences=count_sentences(clean),
        ends_with_question=clean.endswith("?"),
        filler_words=sum(len(pattern.findall(folded)) for pattern in _FILLER_PATTERNS.values()),
        physical_descriptions=len(_PHYSICAL.findall(folded)),
        markdown_marks=len(_MARKDOWN.findall(clean)),
        breaks_character=bool(_BREAKS_CHARACTER.search(folded)),
        chilean_terms=sum(1 for word in words if word in CHILEAN_WORDS) + len(voseo),
        swears=sum(1 for word in words if word in CHILEAN_SWEARS),
        weon_uses=len(_WEON.findall(folded)),
        weon_bad_punctuation=sum(
            len(pattern.findall(folded)) for pattern in _WEON_BAD_PUNCTUATION
        ),
    )


def summarize(metrics: list[AnswerMetrics]) -> dict[str, float]:
    """Aggregate metrics for one mode into comparable numbers."""

    if not metrics:
        return {}
    lengths = [item.characters for item in metrics]
    total = len(metrics)
    return {
        "respuestas": total,
        "caracteres_mediana": statistics.median(lengths),
        "caracteres_max": max(lengths),
        "sobre_200_pct": 100 * sum(length > 200 for length in lengths) / total,
        "oraciones_mediana": statistics.median(item.sentences for item in metrics),
        "pregunta_final_pct": 100 * sum(item.ends_with_question for item in metrics) / total,
        "muletillas_por_respuesta": sum(item.filler_words for item in metrics) / total,
        "descripciones_fisicas": sum(item.physical_descriptions for item in metrics),
        "markdown": sum(item.markdown_marks for item in metrics),
        "rompe_personaje": sum(item.breaks_character for item in metrics),
        "chilenismos_por_oracion": (
            sum(item.chilean_terms for item in metrics)
            / max(1, sum(item.sentences for item in metrics))
        ),
        "garabatos_por_respuesta": sum(item.swears for item in metrics) / total,
        "weon_pct_respuestas": 100 * sum(item.weon_uses > 0 for item in metrics) / total,
        "weon_mal_puntuado": sum(item.weon_bad_punctuation for item in metrics),
    }
