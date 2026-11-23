"""Ask the fixed persona questions and compare two runs of Vader's prompts.

Usage from the project root:

    .\\.venv\\Scripts\\python.exe -m scripts.evaluate_persona original
    .\\.venv\\Scripts\\python.exe -m scripts.evaluate_persona ajustada
    .\\.venv\\Scripts\\python.exe -m scripts.evaluate_persona --compare DIR_A DIR_B

Each question starts from an empty session. Tool requests are recorded but never
executed, so Spotify and the TV are never touched.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time

from vadermind.brain import BrainError, VercelGatewayChatEngine
from vadermind.config import ConfigurationError, load_dotenv_file, load_settings, validate_settings
from vadermind.contracts import SessionState
from vadermind.persona import SUPPORTED_PERSONA_MODES
from vadermind.persona_evaluation import (
    PERSONA_QUESTIONS,
    AnswerMetrics,
    measure_answer,
    summarize,
)
from vadermind.response_text import clean_assistant_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "evaluations" / "results"

SUMMARY_LABELS = {
    "caracteres_mediana": "Largo mediano (caracteres)",
    "caracteres_max": "Largo maximo",
    "sobre_200_pct": "Respuestas sobre 200 car. (%)",
    "oraciones_mediana": "Oraciones (mediana)",
    "pregunta_final_pct": "Terminan en pregunta (%)",
    "muletillas_por_respuesta": "Imperio/debilidad por respuesta",
    "descripciones_fisicas": "Descripciones fisicas",
    "markdown": "Marcas markdown",
    "rompe_personaje": "Rompe personaje",
    "chilenismos_por_oracion": "Chilenismos por oracion",
    "garabatos_por_respuesta": "Garabatos por respuesta",
    "weon_pct_respuestas": "Respuestas con weon (%)",
    "weon_mal_puntuado": "Weon mal puntuado",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "label", nargs="?", help="Nombre corto de la corrida, por ejemplo 'original'."
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["canon", "chileno"],
        choices=sorted(SUPPORTED_PERSONA_MODES),
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("DIR_A", "DIR_B"))
    return parser


def run(label: str, modes: list[str], repetitions: int) -> Path:
    load_dotenv_file(PROJECT_ROOT / ".env")
    settings = load_settings(PROJECT_ROOT / ".env")
    validate_settings(settings, require_voice=False, require_anthropic=False)
    if settings.llm_provider != "vercel":
        raise ConfigurationError("El banco de personalidad usa LLM_PROVIDER=vercel.")

    output = RESULTS_ROOT / f"persona_{datetime.now():%Y%m%d_%H%M%S}_{label}"
    output.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, float]] = {}

    for mode in modes:
        settings.persona_mode = mode
        engine = VercelGatewayChatEngine(settings)
        metrics: list[AnswerMetrics] = []
        for number, question in enumerate(PERSONA_QUESTIONS, start=1):
            for repetition in range(1, repetitions + 1):
                started = time.perf_counter()
                row: dict[str, object] = {
                    "mode": mode,
                    "question_number": number,
                    "question": question,
                    "repetition": repetition,
                }
                try:
                    turn = engine.respond_with_tools(
                        question, SessionState(language=settings.language)
                    )
                    answer = clean_assistant_text(turn.text or "")
                    row.update(
                        answer=answer,
                        tool_requested=turn.tool_name,
                        tier=engine.last_selected_tier,
                        seconds=round(time.perf_counter() - started, 3),
                    )
                    if answer:
                        measured = measure_answer(answer)
                        metrics.append(measured)
                        row["metrics"] = asdict(measured)
                except BrainError as error:
                    row["error"] = str(error)
                rows.append(row)
                preview = str(row.get("answer", row.get("error", "")))[:90]
                print(f"[{mode}] {number:2d}.{repetition} {preview}")
        summaries[mode] = summarize(metrics)

    with (output / "respuestas.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "resumen.json").write_text(
        json.dumps({"label": label, "summaries": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "respuestas.md").write_text(
        _answers_markdown(label, rows, summaries), encoding="utf-8"
    )
    return output


def _answers_markdown(
    label: str, rows: list[dict[str, object]], summaries: dict[str, dict[str, float]]
) -> str:
    lines = [f"# Banco de personalidad: {label}", ""]
    for mode, summary in summaries.items():
        lines += [f"## {mode}", "", _summary_table({label: summary}), ""]
        current = None
        for row in rows:
            if row["mode"] != mode:
                continue
            if row["question_number"] != current:
                current = row["question_number"]
                lines += ["", f"**{current}. {row['question']}**", ""]
            if "error" in row:
                text = f"ERROR: {row['error']}"
            elif row.get("tool_requested"):
                text = f"(pidio la herramienta {row['tool_requested']}) {row.get('answer', '')}"
            else:
                text = str(row["answer"])
            lines.append(f"- [{len(str(row.get('answer', '')))} car.] {text}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _format(value: float) -> str:
    return f"{value:g}" if float(value).is_integer() else f"{value:.2f}"


def _summary_table(columns: dict[str, dict[str, float]]) -> str:
    names = list(columns)
    lines = [
        "| Medida | " + " | ".join(names) + " |",
        "| --- |" + " ---: |" * len(names),
    ]
    for key, title in SUMMARY_LABELS.items():
        values = [
            _format(columns[name][key]) if key in columns[name] else "-" for name in names
        ]
        lines.append(f"| {title} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def compare(first: Path, second: Path) -> str:
    """Write and return a side-by-side table for two benchmark runs."""

    runs = [
        json.loads((path / "resumen.json").read_text(encoding="utf-8"))
        for path in (first, second)
    ]
    sections = ["# Comparacion de personalidad", ""]
    for mode in runs[1]["summaries"]:
        columns = {run["label"]: run["summaries"].get(mode, {}) for run in runs}
        sections += [f"## {mode}", "", _summary_table(columns), ""]
    report = "\n".join(sections)
    (second / "comparacion.md").write_text(report, encoding="utf-8")
    return report


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    try:
        if args.compare:
            print(compare(*args.compare))
            return 0
        if not args.label:
            print("Indica un nombre para la corrida o usa --compare.", file=sys.stderr)
            return 2
        output = run(args.label, args.modes, args.repetitions)
    except (ConfigurationError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"\nResultados en {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
