"""Summarize Vader's timing lines into medians and worst cases.

Vader (hands-free) writes one `Tiempos [...]` line per spoken turn and one
`Arranque ...` line at start-up; journalctl keeps them. Save them and read them:

    En la Pi, sin escribir horas (scripts/medicion.sh guarda la de inicio):
        sh ~/VaderMind/scripts/medicion.sh inicio
        ... guion fijo ...
        sh ~/VaderMind/scripts/medicion.sh fin medicion

    Tras cada encendido en frio (junta varios en ~/arranques.log):
        sh ~/VaderMind/scripts/medicion.sh arranque 31,2

    O a mano:
        journalctl -u vadermind --since "30 min ago" -o cat > ~/medicion.log
        .venv/bin/python scripts/latency_report.py ~/medicion.log

    En el laptop (despues de copiar el archivo con scp):
        .venv\\Scripts\\python.exe scripts\\latency_report.py medicion.log

The first turn of each start-up (full breath, cold connections) is listed but
left out of the medians; `--incluir-primero` keeps it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import re
import statistics
import sys

TIMING = re.compile(r"Tiempos \[(?P<kind>[^\]]+)\]\s*(?P<body>.*)$")
STARTUP = re.compile(r"Arranque (?P<body>.*)$")
USER = re.compile(r"Tu \(voz\) > (?P<text>.*)$")
NUMBER = re.compile(r"([a-z_0-9]+)=(-?\d+(?:\.\d+)?)")
NOTE = re.compile(r"([a-z_0-9]+)=(\S+)")

CATEGORIES = ("charla", "hora", "clima", "spotify", "tv", "local", "otros")

# Rows of each table, in the order a turn happens.
METRICS = (
    ("despertar", "fin de 'Oye Veider' -> escucha"),
    ("vosk", "  Vosk confirma la frase"),
    ("abrir_micro", "  cerrar y abrir el micro"),
    ("reabrir", "fin del turno -> escucha (seguimiento)"),
    ("sonido_inicio", "  sonido de inicio"),
    ("espera_usuario", "escucha -> tu primera palabra"),
    ("calla_suena", "TE CALLAS -> PRIMER SONIDO"),
    ("silencio", "  espera de silencio (fin de habla)"),
    ("stt", "  transcripcion"),
    ("pensar", "  pensar (modelo y tools antes de hablar)"),
    ("voz", "  voz (Piper primer trozo y luces)"),
    ("completa", "te callas -> turno terminado"),
    ("hablar", "  Vader habla (y actua)"),
    ("ttft", "modelo: primera palabra (TTFT)"),
    ("modelo", "modelo: respuesta completa"),
    ("herramienta", "tool (Spotify, TV, hora, clima)"),
    ("voz_previa", "confirmacion de voz previa (Spotify)"),
    ("stt_especulativa", "espera de la transcripcion especulativa"),
    ("stt_completa", "transcripcion sin especular"),
    ("stt_conexion", "streaming: abrir la conexion"),
    ("stt_streaming", "streaming: 'audio listo' -> texto"),
    ("sintesis1", "Piper: primer trozo"),
    ("luces", "luces: envolvente de la voz"),
    ("aplay_extra", "aplay: arranque y cierre"),
    ("espera_trozos", "silencio entre trozos"),
)

# Pieces of "te callas -> primer sonido" plus the wake-up, for the ranking.
RANKED = ("silencio", "stt", "pensar", "voz", "vosk", "abrir_micro", "sonido_inicio", "reabrir")


@dataclass
class Turn:
    kind: str
    values: dict[str, float]
    notes: dict[str, str] = field(default_factory=dict)
    text: str = ""

    @property
    def category(self) -> str:
        return category(self.kind)

    @property
    def first(self) -> bool:
        return self.notes.get("turno") == "1"


def category(kind: str) -> str:
    if kind == "charla":
        return "charla"
    if kind == "information.time":
        return "hora"
    if kind == "information.weather":
        return "clima"
    if kind.startswith(("spotify.", "media.")):
        return "spotify"
    if kind.startswith("lg_tv."):
        return "tv"
    if kind.startswith("local_"):
        return "local"
    return "otros"


def parse(lines: list[str]) -> tuple[list[Turn], list[dict[str, float]]]:
    """Read turns and start-ups from log lines (any prefix before the message is fine)."""

    turns: list[Turn] = []
    startups: list[dict[str, float]] = []
    last_text = ""
    for line in lines:
        line = line.rstrip("\n")
        if match := USER.search(line):
            last_text = match["text"].strip()
            continue
        if match := TIMING.search(line):
            body, _, notes = match["body"].partition(" | ")
            values = {name: float(value) for name, value in NUMBER.findall(body)}
            turns.append(Turn(match["kind"], values, dict(NOTE.findall(notes)), last_text))
            last_text = ""
            continue
        if match := STARTUP.search(line):
            body = match["body"].partition(" | ")[0]
            values = {name: float(value) for name, value in NUMBER.findall(body)}
            if "proceso_a_reposo" in values or not startups:
                startups.append(values)
            else:  # a late "piper_tras_reposo" belongs to the last start-up
                startups[-1].update(values)
    return turns, startups


def _fmt(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def _stats(values: list[float]) -> str:
    return (
        f"{len(values):>3}  {_fmt(statistics.median(values)):>7}  "
        f"{_fmt(min(values)):>6}  {_fmt(max(values)):>6}"
    )


def report(turns: list[Turn], startups: list[dict[str, float]], include_first: bool = False) -> str:
    out: list[str] = []
    if startups:
        out.append("== Arranque (segundos) ==")
        keys = list(dict.fromkeys(key for startup in startups for key in startup))
        out.append(f"{'':28}{'n':>3}  {'mediana':>7}  {'min':>6}  {'max':>6}")
        for key in keys:
            values = [startup[key] for startup in startups if key in startup]
            out.append(f"{key:28}{_stats(values)}")
        out.append("(kernel_a_*: desde que arranca Linux; suma el firmware con el cronometro)")
        out.append("")

    measured = [turn for turn in turns if include_first or not turn.first]
    skipped = len(turns) - len(measured)
    header = f"{'':42}{'n':>3}  {'mediana':>7}  {'min':>6}  {'max':>6}"
    for name in CATEGORIES:
        group = [turn for turn in measured if turn.category == name]
        if not group:
            continue
        kinds = ", ".join(sorted({turn.kind for turn in group}))
        out.append(f"== {name} ({len(group)} turnos: {kinds}) ==")
        out.append(header)
        for key, label in METRICS:
            values = [turn.values[key] for turn in group if key in turn.values]
            if values:
                out.append(f"{label:42}{_stats(values)}")
        out.append("")

    ranked = []
    for key in RANKED:
        values = [turn.values[key] for turn in measured if key in turn.values]
        if values:
            ranked.append((statistics.median(values), max(values), len(values), key))
    if ranked:
        out.append("== Tramos ordenados por mediana (todas las categorias) ==")
        for median, worst, count, key in sorted(ranked, reverse=True):
            out.append(f"{key:16} mediana {_fmt(median):>6} s   peor {_fmt(worst):>6} s   (n={count})")
        out.append("")

    modes = [turn.notes["transcriptor"] for turn in measured if "transcriptor" in turn.notes]
    if modes:
        counts = ", ".join(f"{mode} {modes.count(mode)}" for mode in sorted(set(modes)))
        out.append(f"== Transcriptor (turnos con voz): {counts} ==")
        out.append("")

    out.append("== Turnos ==")
    for turn in turns:
        mark = " (primer turno, fuera de las medianas)" if turn.first and not include_first else ""
        main = turn.values.get("calla_suena")
        whole = turn.values.get("completa")
        figures = "  ".join(
            f"{label} {_fmt(value)}"
            for label, value in (("calla->suena", main), ("completa", whole))
            if value is not None
        )
        text = f' "{turn.text}"' if turn.text else ""
        out.append(f"{turn.notes.get('turno', '?'):>3} [{turn.kind}]{text}  {figures}{mark}")
    if skipped:
        out.append(f"({skipped} primer(os) turno(s) fuera de las medianas; usa --incluir-primero)")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archivos", nargs="*", help="logs de journalctl (sin archivo: entrada estandar)")
    parser.add_argument("--incluir-primero", action="store_true", help="contar el primer turno")
    args = parser.parse_args(argv)
    lines: list[str] = []
    if args.archivos:
        for name in args.archivos:
            lines.extend(Path(name).read_text(encoding="utf-8", errors="replace").splitlines())
    else:
        lines = sys.stdin.read().splitlines()
    turns, startups = parse(lines)
    if not turns and not startups:
        print("No encontre lineas 'Tiempos [...]' ni 'Arranque ...' en el log.")
        return 1
    print(report(turns, startups, include_first=args.incluir_primero))
    return 0


if __name__ == "__main__":
    sys.exit(main())
