#!/bin/sh
# Mark the start of a speed measurement on the Pi and save its log later,
# without typing any time by hand (an error-prone step with --since "HH:MM").
#
# En la Pi (vader@vader), desde cualquier carpeta:
#   sh ~/VaderMind/scripts/medicion.sh inicio
#       guarda la hora actual y reinicia Vader (el servicio vadermind)
#   sh ~/VaderMind/scripts/medicion.sh fin rapido_192
#       guarda en ~/rapido_192.log todo lo que Vader escribio desde "inicio"
#       y muestra el informe de scripts/latency_report.py
#   sh ~/VaderMind/scripts/medicion.sh arranque 31,2
#       tras un encendido en frio, con las pupilas tenues: agrega a
#       ~/arranques.log lo que tardo este encendido (systemd y la linea
#       "Arranque" de Vader) junto al tiempo del cronometro (31,2 s)
#
# The start time lives in a file, so it survives closing the SSH session.

set -e
cd "$(dirname "$0")/.."
MARCA="$HOME/.vadermind_inicio_medicion"
ARRANQUES="$HOME/arranques.log"
# Changes on every power-on; overridable only to try the script off the Pi.
BOOT_ID_FILE="${BOOT_ID_FILE:-/proc/sys/kernel/random/boot_id}"

case "$1" in
  inicio)
    date '+%Y-%m-%d %H:%M:%S' > "$MARCA"
    sudo systemctl restart vadermind
    echo "Medicion iniciada: $(cat "$MARCA"). Espera las pupilas tenues antes de hablar."
    ;;
  fin)
    nombre="${2:-medicion}"
    if [ ! -f "$MARCA" ]; then
      echo "No hay inicio guardado: usa primero 'sh ~/VaderMind/scripts/medicion.sh inicio'."
      exit 1
    fi
    journalctl -u vadermind --since "$(cat "$MARCA")" -o cat > "$HOME/$nombre.log"
    turnos=$(grep -c 'Tiempos \[' "$HOME/$nombre.log" || true)
    echo "Guardado ~/$nombre.log: $turnos turnos desde $(cat "$MARCA")."
    .venv/bin/python scripts/latency_report.py "$HOME/$nombre.log"
    ;;
  arranque)
    boot=$(cat "$BOOT_ID_FILE")
    if [ -f "$ARRANQUES" ] && grep -q "$boot" "$ARRANQUES"; then
      echo "Este encendido ya esta guardado en ~/arranques.log."
      exit 0
    fi
    lineas=$(journalctl -u vadermind -b -o cat | grep 'Arranque' || true)
    if [ -z "$lineas" ]; then
      echo "Vader aun no termina de arrancar: espera las pupilas tenues y repite."
      exit 1
    fi
    resumen=$(systemd-analyze 2>&1 || true)
    case "$resumen" in
      *"not yet finished"*)
        echo "Linux aun no termina de arrancar: espera 10 s y repite."
        exit 1
        ;;
    esac
    n=$(grep -c '^== Encendido' "$ARRANQUES" 2>/dev/null || true)
    n=$(( ${n:-0} + 1 ))
    cronometro="sin anotar"
    [ -n "$2" ] && cronometro="$2 s"
    {
      echo "== Encendido $n: $(date '+%Y-%m-%d %H:%M:%S') | cronometro: $cronometro | boot $boot =="
      echo "-- systemd-analyze"
      echo "$resumen"
      echo "-- systemd-analyze critical-chain vadermind.service"
      systemd-analyze critical-chain vadermind.service 2>&1 || true
      echo "-- systemd-analyze blame (los 15 mas lentos)"
      systemd-analyze blame 2>&1 | head -15
      echo "-- lineas de Vader"
      echo "$lineas"
      echo
    } >> "$ARRANQUES"
    sed -n "/^== Encendido $n:/,\$p" "$ARRANQUES"
    echo "Guardado como encendido $n en ~/arranques.log."
    ;;
  *)
    echo "Uso: sh ~/VaderMind/scripts/medicion.sh inicio | fin NOMBRE | arranque SEGUNDOS"
    exit 1
    ;;
esac
