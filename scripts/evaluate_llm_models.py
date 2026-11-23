"""Run the privacy-constrained VaderMind model evaluation pilot."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys

from vadermind.config import load_dotenv_file
from vadermind.model_evaluation import (
    BASELINE_MODEL,
    PRIMARY_MODELS,
    RESERVE_MODEL,
    EvaluationCase,
    EvaluationRecord,
    cost_projections,
    estimate_cost,
    evaluation_tools,
    load_cases,
    response_tool_calls,
    summarize_records,
    write_records,
    write_review_csv,
)
from vadermind.openrouter import (
    ModelResponse,
    ModelUsage,
    OpenRouterClient,
    OpenRouterError,
    tool_call_to_message,
)
from vadermind.vercel_gateway import VercelGatewayClient
from vadermind.persona import build_system_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIOS = PROJECT_ROOT / "evaluations" / "model_selection_scenarios.json"
DEFAULT_RESULTS = PROJECT_ROOT / "evaluations" / "results"
PREFERRED_BENCHMARK_PROVIDERS = {
    # Novita currently advertises tool support for gpt-oss-20b but rejects the
    # benchmark tool schema with HTTP 400. Groq accepts the same schema and has
    # substantially higher observed output throughput.
    "openai/gpt-oss-20b": "groq",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compara modelos mediante Vercel u OpenRouter con controles de privacidad."
    )
    parser.add_argument(
        "--gateway",
        choices=("vercel", "openrouter"),
        default="vercel",
        help="Gateway usado para la evaluacion (por defecto: vercel).",
    )
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-cost-usd", type=float, default=15.0)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=4519)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=PRIMARY_MODELS,
        help="Ejecuta solo los modelos indicados; el preflight valida todos.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Comprueba endpoints ZDR sin ejecutar el benchmark.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Ejecuta una sola respuesta minima despues del preflight.",
    )
    parser.add_argument(
        "--allow-non-zdr",
        action="store_true",
        help=(
            "Solo Vercel: permite el benchmark sin ZDR, manteniendo no-training. "
            "No afecta el motor normal de VaderMind."
        ),
    )
    return parser


GatewayClient = OpenRouterClient | VercelGatewayClient


def _eligible_endpoint(
    client: GatewayClient,
    model: str,
    *,
    require_zdr: bool = True,
):
    if isinstance(client, VercelGatewayClient):
        endpoints = client.list_model_endpoints(model)
        preferred_provider = PREFERRED_BENCHMARK_PROVIDERS.get(model)
        if preferred_provider:
            preferred = client.select_benchmark_endpoint(
                model,
                (
                    endpoint
                    for endpoint in endpoints
                    if endpoint.provider_name == preferred_provider
                ),
                require_zdr=require_zdr,
            )
            if preferred is not None:
                return preferred
        return client.select_benchmark_endpoint(
            model, endpoints, require_zdr=require_zdr
        )
    endpoints = client.list_zdr_endpoints()
    return client.select_benchmark_endpoint(model, endpoints)


def _select_models(
    client: GatewayClient,
    *,
    require_zdr: bool = True,
) -> tuple[list[str], dict[str, str]]:
    selected_models: list[str] = []
    provider_tags: dict[str, str] = {}
    reserve_used = False

    for model in PRIMARY_MODELS:
        endpoint = _eligible_endpoint(client, model, require_zdr=require_zdr)
        chosen_model = model
        if endpoint is None and model != BASELINE_MODEL and not reserve_used:
            reserve = _eligible_endpoint(
                client, RESERVE_MODEL, require_zdr=require_zdr
            )
            if reserve is not None:
                chosen_model = RESERVE_MODEL
                endpoint = reserve
                reserve_used = True
                print(f"Aviso: {model} no tiene endpoint ZDR elegible; se usara {RESERVE_MODEL}.")
        if endpoint is None:
            privacy_text = "ZDR/no-training" if require_zdr else "no-training"
            raise OpenRouterError(
                f"{model} no tiene un endpoint {privacy_text} con tools, "
                "32K de contexto y 99% de uptime."
            )
        if chosen_model in selected_models:
            raise OpenRouterError(
                "No hay suficientes modelos distintos elegibles para la prueba."
            )
        selected_models.append(chosen_model)
        endpoint_tag = (
            endpoint.provider_name
            if isinstance(client, VercelGatewayClient)
            else endpoint.tag
        )
        provider_tags[chosen_model] = endpoint_tag
        latency = endpoint.latency_p50_seconds
        latency_text = f"{latency:.3f} s" if latency is not None else "sin dato"
        print(
            f"OK {chosen_model}: {endpoint.provider_name} ({endpoint_tag}), "
            f"p50 reciente {latency_text}, uptime {endpoint.uptime_last_1d:.2f}%."
        )

    return selected_models, provider_tags


def _combine_responses(first: ModelResponse, second: ModelResponse) -> ModelResponse:
    first_cost = first.usage.cost_usd
    second_cost = second.usage.cost_usd
    combined_cost = None
    if first_cost is not None or second_cost is not None:
        combined_cost = (first_cost or 0.0) + (second_cost or 0.0)
    return ModelResponse(
        text=second.text,
        tool_calls=first.tool_calls,
        usage=ModelUsage(
            prompt_tokens=first.usage.prompt_tokens + second.usage.prompt_tokens,
            completion_tokens=first.usage.completion_tokens + second.usage.completion_tokens,
            total_tokens=first.usage.total_tokens + second.usage.total_tokens,
            cost_usd=combined_cost,
        ),
        requested_model=first.requested_model,
        effective_model=second.effective_model,
        provider=second.provider or first.provider,
        time_to_first_token_seconds=first.time_to_first_token_seconds,
        total_seconds=first.total_seconds + second.total_seconds,
        attempts=first.attempts + second.attempts,
    )


def _run_case(
    client: GatewayClient,
    *,
    model: str,
    endpoint_tag: str,
    case: EvaluationCase,
) -> ModelResponse:
    system = (
        build_system_prompt(language="es", mode="canon")
        + " Las herramientas entregadas son simuladas para una evaluacion: decide con precision, "
        "no afirmes que una accion se ejecuto y nunca inventes una confirmacion del usuario."
    )
    messages: list[dict[str, object]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": case.prompt},
    ]
    first = client.complete(
        model=model,
        messages=messages,
        tools=evaluation_tools(),
        pinned_provider=endpoint_tag,
        allow_fallbacks=False,
    )
    if case.tool_result is None or not first.tool_calls:
        return first

    matching = [call for call in first.tool_calls if call.name == case.expected_tool]
    if not matching:
        return first

    messages.append(
        {
            "role": "assistant",
            "content": first.text or None,
            "tool_calls": [tool_call_to_message(call) for call in first.tool_calls],
        }
    )
    for call in first.tool_calls:
        result = case.tool_result if call.name == case.expected_tool else {"error": "tool no esperada"}
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )
    second = client.complete(
        model=model,
        messages=messages,
        tools=evaluation_tools(),
        pinned_provider=endpoint_tag,
        allow_fallbacks=False,
    )
    return _combine_responses(first, second)


def _record_success(
    model: str,
    endpoint_tag: str,
    case: EvaluationCase,
    repetition: int,
    response: ModelResponse,
) -> EvaluationRecord:
    from vadermind.model_evaluation import score_response

    score = score_response(case, text=response.text, tool_calls=response.tool_calls)
    return EvaluationRecord(
        model=model,
        provider=response.provider,
        endpoint_tag=endpoint_tag,
        case_id=case.id,
        category=case.category,
        repetition=repetition,
        success=True,
        error=None,
        text=response.text,
        tool_calls=response_tool_calls(response),
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        cost_usd=estimate_cost(model, response),
        time_to_first_token_seconds=response.time_to_first_token_seconds,
        total_seconds=response.total_seconds,
        attempts=response.attempts,
        tool_correct=score.tool_correct,
        factual_checks_passed=score.factual_checks_passed,
        safety_passed=score.safety_passed,
        score_details=score.details,
    )


def _record_error(
    model: str,
    endpoint_tag: str,
    case: EvaluationCase,
    repetition: int,
    error: Exception,
) -> EvaluationRecord:
    return EvaluationRecord(
        model=model,
        provider=None,
        endpoint_tag=endpoint_tag,
        case_id=case.id,
        category=case.category,
        repetition=repetition,
        success=False,
        error=str(error),
        text="",
        tool_calls=[],
        prompt_tokens=0,
        completion_tokens=0,
        cost_usd=0.0,
        time_to_first_token_seconds=None,
        total_seconds=0.0,
        attempts=1,
        tool_correct=False,
        factual_checks_passed=False,
        safety_passed=False,
        score_details=[str(error)],
    )


def _write_summary(path: Path, records: list[EvaluationRecord]) -> None:
    summary = summarize_records(records)
    lines = [
        "# Resultados objetivos del piloto de modelos",
        "",
        "Este informe no declara un ganador hasta completar la revision humana ciega.",
        "",
        "| Modelo | Exito | Mediana total | p95 total | Mediana TTFT | Tools | Hechos | Seguridad | Costo |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, values in summary.items():
        median = values["median_total_seconds"]
        p95 = values["p95_total_seconds"]
        ttft = values["median_ttft_seconds"]
        median_text = f"{median:.3f}s" if isinstance(median, float) else "n/d"
        p95_text = f"{p95:.3f}s" if isinstance(p95, float) else "n/d"
        ttft_text = f"{ttft:.3f}s" if isinstance(ttft, float) else "n/d"
        lines.append(
            f"| {model} | {values['success_rate']:.1%} | "
            f"{median_text} | {p95_text} | {ttft_text} | "
            f"{values['tool_accuracy']:.1%} | {values['factual_check_rate']:.1%} | "
            f"{values['safety_rate']:.1%} | US${values['total_cost_usd']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Proyeccion mensual basada en el consumo observado",
            "",
            "| Modelo | Costo medio/turno | 20/dia | 100/dia | 500/dia |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model, values in summary.items():
        requests = int(values["requests"])
        average_turn_cost = (
            float(values["total_cost_usd"]) / requests if requests else 0.0
        )
        projections = cost_projections(average_turn_cost)
        lines.append(
            f"| {model} | US${average_turn_cost:.6f} | "
            f"US${projections['20']:.4f} | US${projections['100']:.4f} | "
            f"US${projections['500']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Las proyecciones usan 30 dias y el consumo de esta corrida. Los errores que "
            "no devolvieron usage pueden generar una diferencia frente a la factura real.",
            "",
            "## Siguiente paso",
            "",
            "Completar `revision_ciega.csv` y ejecutar `scripts/score_llm_evaluation.py`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if args.repetitions <= 0:
        print("--repetitions debe ser mayor que cero.", file=sys.stderr)
        return 2
    if args.max_cost_usd <= 0:
        print("--max-cost-usd debe ser mayor que cero.", file=sys.stderr)
        return 2
    if args.allow_non_zdr and args.gateway != "vercel":
        print("--allow-non-zdr solo se admite con --gateway vercel.", file=sys.stderr)
        return 2

    load_dotenv_file(PROJECT_ROOT / ".env")
    api_key_variable = (
        "AI_GATEWAY_API_KEY" if args.gateway == "vercel" else "OPENROUTER_API_KEY"
    )
    api_key = os.getenv(api_key_variable, "")
    client_class = VercelGatewayClient if args.gateway == "vercel" else OpenRouterClient
    try:
        cases = load_cases(args.scenarios)
        client_options = {
            "timeout_seconds": args.timeout_seconds,
            "max_retries": 1,
        }
        if args.gateway == "vercel":
            client_options["require_zdr"] = not args.allow_non_zdr
        with client_class(api_key, **client_options) as client:
            models, provider_tags = _select_models(
                client,
                require_zdr=not args.allow_non_zdr,
            )
            if args.models:
                requested_models = set(args.models)
                models = [model for model in models if model in requested_models]
            if args.preflight_only:
                return 0
            if args.smoke_only:
                response = client.complete(
                    model=BASELINE_MODEL,
                    messages=[{"role": "user", "content": "Responde solo OK."}],
                    max_tokens=8,
                    pinned_provider=provider_tags[BASELINE_MODEL],
                    allow_fallbacks=False,
                )
                print(
                    "Smoke test OK: "
                    f"modelo={response.effective_model}, proveedor={response.provider or provider_tags[BASELINE_MODEL]}, "
                    f"TTFT={response.time_to_first_token_seconds or 0.0:.3f}s, total={response.total_seconds:.3f}s."
                )
                return 0

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = args.output_dir / timestamp
            results_path = run_dir / "resultados.jsonl"
            review_path = run_dir / "revision_ciega.csv"
            key_path = run_dir / "clave_modelos.json"
            summary_path = run_dir / "resumen_objetivo.md"

            jobs = [
                (model, case, repetition)
                for model in models
                for case in cases
                for repetition in range(1, args.repetitions + 1)
            ]
            random.Random(args.seed).shuffle(jobs)
            records: list[EvaluationRecord] = []
            total_cost = 0.0
            for index, (model, case, repetition) in enumerate(jobs, start=1):
                if total_cost >= args.max_cost_usd:
                    print(f"Tope de US${args.max_cost_usd:.2f} alcanzado; se detiene la prueba.")
                    break
                tag = provider_tags[model]
                print(f"[{index}/{len(jobs)}] {model} | {case.id} | repeticion {repetition}")
                try:
                    response = _run_case(
                        client,
                        model=model,
                        endpoint_tag=tag,
                        case=case,
                    )
                    record = _record_success(model, tag, case, repetition, response)
                except OpenRouterError as error:
                    record = _record_error(model, tag, case, repetition, error)
                records.append(record)
                total_cost += record.cost_usd
                write_records(results_path, records)

            aliases = write_review_csv(review_path, records)
            key_path.write_text(
                json.dumps({alias: model for model, alias in aliases.items()}, indent=2),
                encoding="utf-8",
            )
            _write_summary(summary_path, records)
            print(f"Resultados: {run_dir}")
            print(f"Costo registrado/estimado: US${total_cost:.4f}")
            return 0 if len(records) == len(jobs) else 1
    except (OpenRouterError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
