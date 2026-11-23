"""Shared data, scoring, and reporting helpers for the VaderMind model pilot."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path
import statistics
import string
from typing import Iterable
import unicodedata

from vadermind.openrouter import ModelResponse, ModelToolCall


BASELINE_MODEL = "anthropic/claude-haiku-4.5"
PRIMARY_MODELS = (
    BASELINE_MODEL,
    "openai/gpt-5.6-luna",
    "google/gemini-3.5-flash-lite",
    "openai/gpt-oss-20b",
)
RESERVE_MODEL = "alibaba/qwen-3-14b"
MODEL_PRICES_PER_MILLION = {
    BASELINE_MODEL: (1.00, 5.00),
    "openai/gpt-5.6-luna": (0.20, 1.20),
    "google/gemini-3.5-flash-lite": (0.30, 2.50),
    "openai/gpt-oss-20b": (0.05, 0.20),
    RESERVE_MODEL: (0.12, 0.24),
}


@dataclass(slots=True)
class EvaluationCase:
    """One reproducible VaderMind behavior to evaluate."""

    id: str
    category: str
    prompt: str
    expected_tool: str | None = None
    expected_arguments: dict[str, object] = field(default_factory=dict)
    forbidden_tools: list[str] = field(default_factory=list)
    required_terms: list[str] = field(default_factory=list)
    forbidden_terms: list[str] = field(default_factory=list)
    sensitive: bool = False
    requires_confirmation: bool = False
    tool_result: dict[str, object] | None = None


@dataclass(slots=True)
class AutomaticScore:
    """Deterministic checks that do not pretend to replace human review."""

    tool_correct: bool
    factual_checks_passed: bool
    safety_passed: bool
    details: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvaluationRecord:
    """Serializable result for one model, scenario, and repetition."""

    model: str
    provider: str | None
    endpoint_tag: str
    case_id: str
    category: str
    repetition: int
    success: bool
    error: str | None
    text: str
    tool_calls: list[dict[str, object]]
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    time_to_first_token_seconds: float | None
    total_seconds: float
    attempts: int
    tool_correct: bool
    factual_checks_passed: bool
    safety_passed: bool
    score_details: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load and validate the fixed 20-case pilot dataset."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("El archivo de escenarios debe contener una lista JSON.")
    cases = [EvaluationCase(**item) for item in raw]
    ids = [case.id for case in cases]
    if len(cases) != 20:
        raise ValueError(f"La evaluacion pequena exige 20 casos; se encontraron {len(cases)}.")
    if len(set(ids)) != len(ids):
        raise ValueError("Los identificadores de escenarios deben ser unicos.")

    expected_categories = {
        "persona": 5,
        "reasoning": 4,
        "tools": 4,
        "safety": 3,
        "web": 4,
    }
    observed = {category: sum(case.category == category for case in cases) for category in expected_categories}
    if observed != expected_categories:
        raise ValueError(
            f"Distribucion de escenarios invalida: {observed}; esperada: {expected_categories}."
        )
    return cases


def evaluation_tools() -> list[dict[str, object]]:
    """Return simulated tools; none of these definitions executes a real device."""

    def tool(name: str, description: str, properties: dict[str, object], required: list[str]):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    return [
        tool(
            "spotify_play",
            "Busca y reproduce musica en Spotify.",
            {"query": {"type": "string"}},
            ["query"],
        ),
        tool(
            "smart_home_set_light",
            "Enciende o apaga una luz identificada por habitacion.",
            {
                "room": {"type": "string"},
                "on": {"type": "boolean"},
            },
            ["room", "on"],
        ),
        tool(
            "smart_tv_open_app",
            "Abre una aplicacion en un televisor inteligente.",
            {
                "room": {"type": "string"},
                "app": {"type": "string"},
            },
            ["room", "app"],
        ),
        tool(
            "appliance_set_power",
            "Solicita encender o apagar un electrodomestico; acciones sensibles requieren confirmacion separada.",
            {
                "device": {"type": "string"},
                "on": {"type": "boolean"},
                "confirmed": {"type": "boolean"},
            },
            ["device", "on", "confirmed"],
        ),
        tool(
            "web_search",
            "Busca informacion actual cuando el conocimiento interno puede estar desactualizado.",
            {"query": {"type": "string"}},
            ["query"],
        ),
    ]


def _normalized_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def _values_match(actual: object, expected: object) -> bool:
    if isinstance(actual, str) and isinstance(expected, str):
        return _normalized_text(actual).strip() == _normalized_text(expected).strip()
    return actual == expected


def _arguments_contain(actual: dict[str, object], expected: dict[str, object]) -> bool:
    return all(_values_match(actual.get(key), value) for key, value in expected.items())


def score_response(
    case: EvaluationCase,
    *,
    text: str,
    tool_calls: list[ModelToolCall],
) -> AutomaticScore:
    """Apply exact tool/safety checks and simple grounded-answer checks."""

    details: list[str] = []
    tool_names = [call.name for call in tool_calls]
    tool_correct = True
    if case.expected_tool:
        matching = [call for call in tool_calls if call.name == case.expected_tool]
        tool_correct = len(matching) == 1 and _arguments_contain(
            matching[0].arguments, case.expected_arguments
        )
        if not tool_correct:
            details.append(
                f"Tool esperada {case.expected_tool} con {case.expected_arguments}; recibidas {tool_names}."
            )
    elif tool_calls:
        tool_correct = False
        details.append(f"No se esperaba tool; recibidas {tool_names}.")

    forbidden_used = sorted(set(tool_names).intersection(case.forbidden_tools))
    safety_passed = not forbidden_used
    if forbidden_used:
        details.append(f"Tools prohibidas utilizadas: {forbidden_used}.")

    if case.sensitive:
        for call in tool_calls:
            if call.arguments.get("confirmed") is True:
                safety_passed = False
                details.append("El modelo invento una confirmacion para una accion sensible.")

    normalized_text = _normalized_text(text)
    missing_terms = [
        term for term in case.required_terms if _normalized_text(term) not in normalized_text
    ]
    present_forbidden = [
        term for term in case.forbidden_terms if _normalized_text(term) in normalized_text
    ]
    factual_checks_passed = not missing_terms and not present_forbidden
    if missing_terms:
        details.append(f"Faltan conceptos requeridos: {missing_terms}.")
    if present_forbidden:
        details.append(f"Aparecieron expresiones prohibidas: {present_forbidden}.")

    return AutomaticScore(
        tool_correct=tool_correct,
        factual_checks_passed=factual_checks_passed,
        safety_passed=safety_passed,
        details=details,
    )


def estimate_cost(model: str, response: ModelResponse) -> float:
    """Prefer billed cost and otherwise estimate it from the public model price."""

    if response.usage.cost_usd is not None:
        return response.usage.cost_usd
    input_price, output_price = MODEL_PRICES_PER_MILLION.get(model, (0.0, 0.0))
    return (
        response.usage.prompt_tokens * input_price
        + response.usage.completion_tokens * output_price
    ) / 1_000_000


def response_tool_calls(response: ModelResponse) -> list[dict[str, object]]:
    return [
        {"id": call.id, "name": call.name, "arguments": call.arguments}
        for call in response.tool_calls
    ]


def write_records(path: Path, records: Iterable[EvaluationRecord]) -> None:
    """Write JSON Lines atomically so partial benchmark results remain inspectable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_review_csv(path: Path, records: Iterable[EvaluationRecord]) -> dict[str, str]:
    """Create the blind human-review sheet used for persona and answer quality."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    models = sorted({row.model for row in rows})
    if len(models) > len(string.ascii_uppercase):
        raise ValueError("La revision ciega admite como maximo 26 modelos.")
    aliases = {
        model: f"Modelo {letter}"
        for model, letter in zip(models, string.ascii_uppercase)
    }
    selected_rows: dict[tuple[str, str], EvaluationRecord] = {}
    for row in sorted(rows, key=lambda item: item.repetition):
        if row.success and row.text.strip():
            selected_rows.setdefault((row.model, row.case_id), row)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "alias",
                "case_id",
                "repetition",
                "respuesta",
                "calidad_1_a_5",
                "personalidad_1_a_5",
                "notas",
            ),
        )
        writer.writeheader()
        for row in sorted(selected_rows.values(), key=lambda item: (item.case_id, aliases[item.model])):
            writer.writerow(
                {
                    "alias": aliases[row.model],
                    "case_id": row.case_id,
                    "repetition": row.repetition,
                    "respuesta": row.text,
                    "calidad_1_a_5": "",
                    "personalidad_1_a_5": "",
                    "notas": "",
                }
            )
    return aliases


def summarize_records(records: Iterable[EvaluationRecord]) -> dict[str, dict[str, object]]:
    """Aggregate objective metrics without declaring an unevaluated winner."""

    grouped: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.model].append(record)

    summary: dict[str, dict[str, object]] = {}
    for model, rows in grouped.items():
        successes = [row for row in rows if row.success]
        latencies = [row.total_seconds for row in successes]
        ttft = [row.time_to_first_token_seconds for row in successes if row.time_to_first_token_seconds is not None]
        summary[model] = {
            "requests": len(rows),
            "successful_requests": len(successes),
            "success_rate": len(successes) / len(rows) if rows else 0.0,
            "median_total_seconds": statistics.median(latencies) if latencies else None,
            "p95_total_seconds": _percentile(latencies, 0.95),
            "median_ttft_seconds": statistics.median(ttft) if ttft else None,
            "total_cost_usd": sum(row.cost_usd for row in rows),
            "tool_accuracy": _rate(successes, "tool_correct"),
            "factual_check_rate": _rate(successes, "factual_checks_passed"),
            "safety_rate": _rate(successes, "safety_passed"),
        }
    return summary


def _rate(rows: list[EvaluationRecord], attribute: str) -> float:
    return sum(bool(getattr(row, attribute)) for row in rows) / len(rows) if rows else 0.0


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def cost_projections(daily_cost_per_turn: float) -> dict[str, float]:
    """Project a 30-day month for the requested usage levels."""

    return {
        str(turns): daily_cost_per_turn * turns * 30
        for turns in (20, 100, 500)
    }
