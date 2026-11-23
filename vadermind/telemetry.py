"""Small logging and timing utilities."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import sys
import threading
import time


def configure_logging(debug: bool = False) -> None:
    """Configure standard logging for the application."""

    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # Connection-level chatter from the HTTP stack hides VaderMind's own timings.
    for noisy in ("httpcore", "httpx", "hpack"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # aiowebostv dumps the whole channel list and programme guide on connect.
    for noisy in ("aiowebostv", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.INFO)


def _seconds(value: float) -> str:
    return f"{value:.2f}"


class TurnTimeline:
    """What happened during one spoken turn, to measure where the time goes.

    - marks: first time each milestone happened (perf_counter);
    - durations: seconds spent in each step, summed when it repeats (every
      `timed_operation` lands here too);
    - notes: short labels such as the model tier or the CPU temperature.

    Recording never changes what Vader does: it only reads the clock.
    """

    def __init__(self) -> None:
        self.marks: dict[str, float] = {}
        self.durations: dict[str, float] = {}
        self.notes: dict[str, str] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.marks.clear()
            self.durations.clear()
            self.notes.clear()

    def mark(self, name: str, at: float | None = None) -> None:
        with self._lock:
            self.marks.setdefault(name, time.perf_counter() if at is None else at)

    def add_duration(self, name: str, seconds: float) -> None:
        with self._lock:
            self.durations[name] = self.durations.get(name, 0.0) + seconds

    def note(self, name: str, value: object) -> None:
        with self._lock:
            self.notes[name] = str(value)

    def _span(self, first: str, second: str) -> float | None:
        if first in self.marks and second in self.marks:
            return self.marks[second] - self.marks[first]
        return None

    def summary(self) -> str | None:
        """Describe end of user speech -> Vader's first sound, or None."""

        marks = self.marks
        start = marks.get("fin_habla")
        if start is None:
            return None
        parts = []
        steps = (
            ("silencio", "fin_habla", "grabacion_cerrada"),
            ("stt", "grabacion_cerrada", "transcripcion_lista"),
            ("modelo", "transcripcion_lista", "sintesis_inicio"),
            ("voz", "sintesis_inicio", "primer_sonido"),
        )
        for label, first, second in steps:
            span = self._span(first, second)
            if span is not None:
                parts.append(f"{label} {span:.2f}")
        if "primer_sonido" in marks:
            head = f"fin_habla->primer_sonido {marks['primer_sonido'] - start:.2f} s"
        else:
            head = "fin_habla->sin voz de Vader"
        return f"{head} ({' · '.join(parts)})" if parts else head

    def tool_name(self) -> str | None:
        """The tool this turn ran (every tool call is timed as `tool_<name>`)."""

        for name in self.durations:
            if name.startswith("tool_"):
                return name[len("tool_") :]
        return None

    # Spans of the turn line: (label, from milestone, to milestone).
    _SPANS = (
        # Woken by "Oye Vader": from the end of the phrase until Vader listens.
        ("despertar", "fin_palabra", "escucha_lista"),
        ("vosk", "fin_palabra", "palabra_detectada"),
        ("abrir_micro", "palabra_detectada", "micro_abierto"),
        # Follow-up window: from the end of the previous turn until he listens again.
        ("reabrir", "turno_anterior_fin", "micro_abierto"),
        ("sonido_inicio", "micro_abierto", "escucha_lista"),
        ("espera_usuario", "escucha_lista", "inicio_habla"),
        # The main figure: the user goes quiet -> Vader's first sound.
        ("calla_suena", "fin_habla", "primer_sonido"),
        ("silencio", "fin_habla", "grabacion_cerrada"),
        ("stt", "grabacion_cerrada", "transcripcion_lista"),
        ("pensar", "transcripcion_lista", "sintesis_inicio"),
        ("voz", "sintesis_inicio", "primer_sonido"),
        # Until Vader is done (voice and action) and the turn closes.
        ("completa", "fin_habla", "fin_turno"),
        ("hablar", "primer_sonido", "fin_turno"),
    )

    # Step durations worth showing: (label, recorded name).
    _STEPS = (
        ("ttft", "ttft"),
        ("modelo", "enrutamiento_capacidades_modelo"),
        ("modelo_charla", "respuesta_modelo"),
        ("stt_especulativa", "espera_transcripcion_especulativa"),
        ("stt_completa", "transcripcion"),
        ("stt_conexion", "conexion_stt_streaming"),
        ("stt_streaming", "respuesta_stt_streaming"),
        ("voz_previa", "salida_voz_previa_spotify"),
        ("espera_prefetch", "espera_prefetch_spotify"),
        ("sintesis1", "sintesis_voz"),
        ("luces", "envolvente_luces"),
        ("espera_trozos", "espera_trozos"),
        ("audio_vader", "audio_vader"),
    )

    def report(self, kind: str | None = None) -> str:
        """One line per turn: `Tiempos [kind] name=seconds ... | notes`.

        Every value is `name=number` so `scripts/latency_report.py` can read it
        back from the journal; spans that did not happen are left out.
        """

        with self._lock:
            label = kind or self.tool_name() or "charla"
            parts = []
            for name, first, second in self._SPANS:
                span = self._span(first, second)
                if span is not None:
                    parts.append(f"{name}={_seconds(span)}")
            tool = self.tool_name()
            if tool:
                parts.append(f"herramienta={_seconds(self.durations['tool_' + tool])}")
            for name, key in self._STEPS:
                if key in self.durations:
                    parts.append(f"{name}={_seconds(self.durations[key])}")
            if "reproduccion" in self.durations and "audio_vader" in self.durations:
                # aplay's own start-up and drain, beyond the audio itself.
                extra = self.durations["reproduccion"] - self.durations["audio_vader"]
                parts.append(f"aplay_extra={_seconds(extra)}")
            notes = " ".join(f"{name}={value}" for name, value in self.notes.items())
        line = f"Tiempos [{label}] {' '.join(parts)}"
        return f"{line} | {notes}" if notes else line


TURN_TIMELINE = TurnTimeline()


def cpu_temperature() -> str | None:
    """CPU temperature in °C on Linux (the Pi), None elsewhere."""

    try:
        raw = Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()
        return f"{int(raw) / 1000:.1f}"
    except (OSError, ValueError):
        return None


def cpu_governor() -> str | None:
    """How Linux picks the CPU speed ("ondemand", "performance"...), None elsewhere."""

    try:
        return Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        return None


def _seconds_since_boot() -> float | None:
    clock = getattr(time, "CLOCK_BOOTTIME", None)
    if clock is None or not sys.platform.startswith("linux"):
        return None
    return time.clock_gettime(clock)


def _process_started_since_boot() -> float | None:
    """When this Python process started, in seconds since the kernel booted."""

    try:
        stat = Path(f"/proc/{os.getpid()}/stat").read_text()
        # Field 22 (starttime) counts clock ticks; the name in field 2 may hold spaces.
        fields = stat[stat.rindex(")") + 2 :].split()
        return int(fields[19]) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, AttributeError):
        return None


class StartupTimeline:
    """Milestones from power-on to the dim pupils (Vader ready), logged once.

    On the Pi the kernel clock gives "seconds since boot", so the line shows
    how long the system took before VaderMind even started. The firmware time
    before the kernel (plug -> kernel) is not visible from software.
    """

    # (label, milestone): each step counts from the previous milestone present.
    _STEPS = (
        ("python", "main"),
        ("imports", "imports"),
        ("config", "config"),
        ("motor_voz", "motor_voz"),
        ("escucha", "escucha"),
        ("herramientas", "herramientas"),
        ("luces", "luces"),
        ("vosk", "vosk"),
        ("red", "red"),  # only with --esperar-red: waiting for the Wi-Fi after loading
        ("reposo", "reposo"),
    )

    def __init__(self) -> None:
        self.marks: dict[str, float] = {}
        self.reported = False
        self._lock = threading.Lock()
        # perf_counter -> seconds since boot (None outside Linux).
        boot_now = _seconds_since_boot()
        self._boot_offset = None if boot_now is None else boot_now - time.perf_counter()
        started = _process_started_since_boot()
        if started is not None and self._boot_offset is not None:
            self.marks["proceso"] = started - self._boot_offset

    def mark(self, name: str, at: float | None = None) -> None:
        with self._lock:
            self.marks.setdefault(name, time.perf_counter() if at is None else at)

    def report(self) -> str:
        marks = self.marks
        parts = []
        if self._boot_offset is not None and "reposo" in marks:
            parts.append(f"kernel_a_reposo={_seconds(marks['reposo'] + self._boot_offset)}")
            if "proceso" in marks:
                parts.append(f"kernel_a_proceso={_seconds(marks['proceso'] + self._boot_offset)}")
        if "reposo" in marks:
            first = marks.get("proceso", marks.get("main"))
            if first is not None:
                parts.append(f"proceso_a_reposo={_seconds(marks['reposo'] - first)}")
        previous = marks.get("proceso")
        for label, name in self._STEPS:
            if name not in marks:
                continue
            if previous is not None:
                parts.append(f"{label}={_seconds(marks[name] - previous)}")
            previous = marks[name]
        if "piper_listo" in marks and "reposo" in marks:
            # Positive: Piper finished warming up after the pupils were already dim.
            parts.append(f"piper_tras_reposo={_seconds(marks['piper_listo'] - marks['reposo'])}")
        notes = [f"{name}={value}" for name, value in (
            ("gobernador", cpu_governor()), ("temp", cpu_temperature())
        ) if value]
        line = f"Arranque {' '.join(parts)}"
        return f"{line} | {' '.join(notes)}" if notes else line

    def log_ready(self) -> None:
        """Log the startup line the first time Vader reaches the dim pupils."""

        with self._lock:
            if self.reported:
                return
            self.reported = True
        self.mark("reposo")
        logging.getLogger("vadermind.telemetry").info(self.report())

    def piper_ready(self) -> None:
        """Piper warms up in the background: log it alone if the pupils were already dim."""

        self.mark("piper_listo")
        with self._lock:
            late = self.reported and "reposo" in self.marks
        if late:
            logging.getLogger("vadermind.telemetry").info(
                "Arranque piper_tras_reposo=%s",
                _seconds(self.marks["piper_listo"] - self.marks["reposo"]),
            )


STARTUP = StartupTimeline()


@contextmanager
def timed_operation(operation_name: str):
    """Measure and log the execution time of a code block."""

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        TURN_TIMELINE.add_duration(operation_name, elapsed)
        logging.getLogger("vadermind.telemetry").debug(
            "Operacion '%s' completada en %.2f segundos.",
            operation_name,
            elapsed,
        )
