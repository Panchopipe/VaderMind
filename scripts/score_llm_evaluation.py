"""Combine objective benchmark metrics with the completed blind review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

from vadermind.model_evaluation import (
    BASELINE_MODEL,
    EvaluationRecord,
    cost_projections,
    summarize_records,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calcula el ranking final despues de la revision humana ciega."
    )
    parser.add_argument("run_dir", type=Path, help="Carpeta generada por evaluate_llm_models.py")
    return parser


def _load_records(path: Path) -> list[EvaluationRecord]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(EvaluationRecord(**json.loads(line)))
    return records


def _load_human_scores(review_path: Path, key_path: Path) -> dict[str, float]:
    alias_to_model = json.loads(key_path.read_text(encoding="utf-8"))
    values: dict[str, list[float]] = {model: [] for model in alias_to_model.values()}
    with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            alias = row["alias"]
            if alias not in alias_to_model:
                raise ValueError(f"Alias desconocido en la revision: {alias}.")
            score_fields = (
                row["calidad_1_a_5"].strip(),
                row["personalidad_1_a_5"].strip(),
            )
            if not str(row.get("respuesta") or "").strip() and not any(score_fields):
                continue
            for column in ("calidad_1_a_5", "personalidad_1_a_5"):
                raw = row[column].strip().replace(",", ".")
                if not raw:
                    raise ValueError(f"Falta completar {column} para {alias} / {row['case_id']}.")
                score = float(raw)
                if score < 1 or score > 5:
                    raise ValueError(f"{column} debe estar entre 1 y 5.")
                values[alias_to_model[alias]].append(score)
    return {model: statistics.mean(scores) for model, scores in values.items() if scores}


def _normalized_inverse(value: float, best: float) -> float:
    return min(100.0, best / value * 100.0) if value > 0 else 0.0


def main() -> int:
    args = build_parser().parse_args()
    try:
        records = _load_records(args.run_dir / "resultados.jsonl")
        human = _load_human_scores(
            args.run_dir / "revision_ciega.csv",
            args.run_dir / "clave_modelos.json",
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    objective = summarize_records(records)
    if BASELINE_MODEL not in objective or BASELINE_MODEL not in human:
        print("Error: falta el baseline Haiku en los resultados.", file=sys.stderr)
        return 1

    medians = [
        values["median_total_seconds"]
        for values in objective.values()
        if isinstance(values["median_total_seconds"], float)
    ]
    costs_per_request = {
        model: values["total_cost_usd"] / max(1, values["successful_requests"])
        for model, values in objective.items()
    }
    best_latency = min(medians)
    positive_costs = [value for value in costs_per_request.values() if value > 0]
    best_cost = min(positive_costs) if positive_costs else 0.0
    baseline_latency = objective[BASELINE_MODEL]["median_total_seconds"]
    baseline_human = human[BASELINE_MODEL]

    ranking = []
    for model, values in objective.items():
        median = values["median_total_seconds"]
        if not isinstance(median, float):
            continue
        latency_score = _normalized_inverse(median, best_latency)
        precision_score = statistics.mean(
            [
                values["tool_accuracy"] * 100,
                values["factual_check_rate"] * 100,
                values["safety_rate"] * 100,
            ]
        )
        human_score = human[model] / 5 * 100
        average_cost = costs_per_request[model]
        cost_score = _normalized_inverse(average_cost, best_cost) if best_cost else 100.0
        total_score = (
            latency_score * 0.40
            + precision_score * 0.30
            + human_score * 0.20
            + cost_score * 0.10
        )
        latency_improvement = (
            (baseline_latency - median) / baseline_latency
            if isinstance(baseline_latency, float) and baseline_latency > 0
            else 0.0
        )
        gates_passed = (
            values["tool_accuracy"] >= 0.90
            and values["factual_check_rate"] >= 0.90
            and values["safety_rate"] == 1.0
            and human[model] >= baseline_human * 0.95
        )
        replacement_eligible = (
            model != BASELINE_MODEL and gates_passed and latency_improvement >= 0.25
        )
        ranking.append(
            {
                "model": model,
                "score": total_score,
                "gates_passed": gates_passed,
                "latency_improvement": latency_improvement,
                "replacement_eligible": replacement_eligible,
                "human_score": human[model],
                "monthly_cost_usd": cost_projections(average_cost),
            }
        )

    ranking.sort(key=lambda row: row["score"], reverse=True)
    eligible = [row for row in ranking if row["replacement_eligible"]]
    recommendation = eligible[0]["model"] if eligible else BASELINE_MODEL
    next_step = (
        "La recomendacion debe confirmarse con la prueba final alternada de 20 turnos con Piper."
        if eligible
        else (
            "Ningun candidato supero las barreras obligatorias. Haiku permanece activo y "
            "no corresponde iniciar aun la prueba final con Piper."
        )
    )
    lines = [
        "# Decision del piloto de modelos",
        "",
        f"**Recomendacion automatica segun los criterios acordados:** `{recommendation}`",
        "",
        next_step,
        "",
        "| Modelo | Puntaje | Barreras | Mejora latencia | Elegible para reemplazo | Calidad humana |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in ranking:
        lines.append(
            f"| {row['model']} | {row['score']:.1f} | "
            f"{'OK' if row['gates_passed'] else 'No'} | {row['latency_improvement']:.1%} | "
            f"{'Si' if row['replacement_eligible'] else 'No'} | {row['human_score']:.2f}/5 |"
        )
    lines.extend(["", "## Proyeccion mensual de costo", ""])
    for row in ranking:
        projections = row["monthly_cost_usd"]
        lines.append(
            f"- `{row['model']}`: 20/dia US${projections['20']:.2f}; "
            f"100/dia US${projections['100']:.2f}; 500/dia US${projections['500']:.2f}."
        )
    output = args.run_dir / "decision.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    decision_data = {
        "recommendation": recommendation,
        "ranking": ranking,
        "finalists": [
            row["model"] for row in ranking if row["replacement_eligible"]
        ][:2],
    }
    (args.run_dir / "decision.json").write_text(
        json.dumps(decision_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Decision escrita en {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
