"""Run the final balanced, blind 20-turn comparison with local Piper voice."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
import time

from vadermind.config import ConfigurationError, load_dotenv_file, load_settings, validate_settings
from vadermind.model_evaluation import estimate_cost
from vadermind.openrouter import OpenRouterClient, OpenRouterError
from vadermind.vercel_gateway import VercelGatewayClient
from vadermind.persona import build_system_prompt
from vadermind.response_text import clean_assistant_text
from vadermind.voice_output import PiperTextToSpeech, VoiceError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compara a ciegas los dos finalistas con 20 respuestas habladas por Piper."
    )
    parser.add_argument("run_dir", type=Path, help="Carpeta del piloto que contiene decision.json")
    parser.add_argument(
        "--gateway",
        choices=("vercel", "openrouter"),
        default="vercel",
        help="Gateway del piloto (por defecto: vercel).",
    )
    parser.add_argument("--seed", type=int, default=1977)
    return parser


def _load_finalists(run_dir: Path) -> tuple[list[str], dict[str, str]]:
    decision = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
    finalists = decision.get("finalists", [])
    if not isinstance(finalists, list) or len(finalists) != 2:
        raise ValueError("decision.json debe contener exactamente dos finalistas.")

    endpoint_counts: dict[str, Counter[str]] = {str(model): Counter() for model in finalists}
    for line in (run_dir / "resultados.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        model = str(record.get("model", ""))
        tag = str(record.get("endpoint_tag", ""))
        if model in endpoint_counts and tag:
            endpoint_counts[model][tag] += 1
    tags = {}
    for model, counts in endpoint_counts.items():
        if not counts:
            raise ValueError(f"No encontre el endpoint usado para {model}.")
        tags[model] = counts.most_common(1)[0][0]
    return [str(model) for model in finalists], tags


def _save(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _read_rating() -> int:
    while True:
        raw = input("Calificacion general de esta respuesta (1-5): ").strip()
        if raw in {"1", "2", "3", "4", "5"}:
            return int(raw)
        print("Escribe un numero entero entre 1 y 5.")


def main() -> int:
    args = build_parser().parse_args()
    load_dotenv_file(PROJECT_ROOT / ".env")
    try:
        finalists, tags = _load_finalists(args.run_dir)
        settings = load_settings(PROJECT_ROOT / ".env")
        validate_settings(settings, require_voice=True, require_anthropic=False)
        if settings.tts_provider != "piper":
            raise ConfigurationError("Esta prueba exige VADER_TTS_PROVIDER=piper.")
        api_key = (
            settings.ai_gateway_api_key
            if args.gateway == "vercel"
            else settings.openrouter_api_key
        )
        if not api_key:
            variable = (
                "AI_GATEWAY_API_KEY"
                if args.gateway == "vercel"
                else "OPENROUTER_API_KEY"
            )
            raise ConfigurationError(f"Falta {variable} en .env.")
        tts = PiperTextToSpeech(settings)
    except (OSError, ValueError, json.JSONDecodeError, ConfigurationError, VoiceError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    slots = finalists * 10
    rng.shuffle(slots)
    aliases = finalists.copy()
    rng.shuffle(aliases)
    alias_by_model = {model: f"Modelo {letter}" for model, letter in zip(aliases, "AB")}
    output = args.run_dir / f"prueba_voz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    rows: list[dict[str, object]] = []
    system = build_system_prompt(language=settings.language, mode=settings.persona_mode)

    print("Prueba a ciegas lista. Los modelos apareceran solamente como A o B.")
    print("Usa preguntas reales. Escribe 'salir' si necesitas terminar antes.")
    client_class = VercelGatewayClient if args.gateway == "vercel" else OpenRouterClient
    timeout_seconds = (
        settings.ai_gateway_timeout_seconds
        if args.gateway == "vercel"
        else settings.openrouter_timeout_seconds
    )
    max_retries = (
        settings.ai_gateway_max_retries
        if args.gateway == "vercel"
        else settings.openrouter_max_retries
    )
    with client_class(
        api_key,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    ) as client:
        for turn, model in enumerate(slots, start=1):
            prompt = input(f"\nTurno {turn}/20 - Tu > ").strip()
            if prompt.casefold() in {"salir", "exit", "quit"}:
                break
            if not prompt:
                print("Escribe una pregunta; este turno no se enviara vacio.")
                while not prompt:
                    prompt = input(f"Turno {turn}/20 - Tu > ").strip()
                    if prompt.casefold() in {"salir", "exit", "quit"}:
                        break
                if prompt.casefold() in {"salir", "exit", "quit"}:
                    break
            alias = alias_by_model[model]
            try:
                response = client.complete(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=settings.anthropic_max_output_tokens,
                    pinned_provider=tags[model],
                    allow_fallbacks=False,
                )
                answer = clean_assistant_text(response.text)
                voice_start = time.perf_counter()
                tts.speak(answer)
                voice_total = time.perf_counter() - voice_start
                rating = _read_rating()
                print(f"Registrado: {alias}.")
                row = {
                    "turn": turn,
                    "alias": alias,
                    "model": model,
                    "provider": response.provider,
                    "endpoint_tag": tags[model],
                    "prompt": prompt,
                    "answer": answer,
                    "rating": rating,
                    "model_ttft_seconds": response.time_to_first_token_seconds,
                    "model_total_seconds": response.total_seconds,
                    "piper_speak_total_seconds": voice_total,
                    "cost_usd": estimate_cost(model, response),
                }
            except (OpenRouterError, VoiceError) as error:
                print(f"El turno fallo y no se sustituira por el otro modelo: {error}")
                row = {
                    "turn": turn,
                    "alias": alias,
                    "model": model,
                    "endpoint_tag": tags[model],
                    "prompt": prompt,
                    "error": str(error),
                }
            rows.append(row)
            _save(output, rows)

    print(f"Resultados guardados en {output}")
    print("La clave de modelos queda dentro del archivo local para analizarla al terminar.")
    return 0 if len(rows) == 20 and not any("error" in row for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
