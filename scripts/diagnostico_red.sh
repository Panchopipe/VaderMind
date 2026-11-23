#!/bin/sh
# Read-only look at how the Wi-Fi came up during this boot (option A4 of the
# speed stage: why NetworkManager takes ~9 s to start). Changes nothing and
# never prints Wi-Fi passwords: only logs, timings and connection names.
#
# Desde PowerShell del laptop (lo deja en ~/diagnostico_red.log de la Pi):
#   ssh vader@vader.local "sh ~/VaderMind/scripts/diagnostico_red.sh > ~/diagnostico_red.log 2>&1"
# Sirve despues de un encendido en frio, antes del siguiente reinicio.

echo "== Diagnostico de red: $(date '+%Y-%m-%d %H:%M:%S') | boot $(cat /proc/sys/kernel/random/boot_id) =="
echo "-- systemd-analyze"
systemd-analyze 2>&1
echo
echo "-- cadena critica hasta la red (network-online.target)"
systemd-analyze critical-chain network-online.target 2>&1
echo
echo "-- cadena critica de NetworkManager.service"
systemd-analyze critical-chain NetworkManager.service 2>&1
echo
echo "-- cuando partio y cuando quedo listo cada servicio (microsegundos desde el kernel)"
for unidad in NetworkManager.service NetworkManager-wait-online.service \
  wpa_supplicant.service dbus.service cloud-init-local.service \
  cloud-init-network.service cloud-init-main.service vadermind.service; do
  echo "[$unidad]"
  systemctl show "$unidad" -p InactiveExitTimestampMonotonic \
    -p ActiveEnterTimestampMonotonic -p Type 2>&1
done
echo
echo "-- definicion de los servicios de red (incluye agregados)"
systemctl cat NetworkManager.service NetworkManager-wait-online.service 2>&1
echo
echo "-- configuracion de NetworkManager (sin conexiones ni claves)"
cat /etc/NetworkManager/NetworkManager.conf 2>&1
for archivo in /etc/NetworkManager/conf.d/*.conf /usr/lib/NetworkManager/conf.d/*.conf /run/NetworkManager/conf.d/*.conf; do
  [ -f "$archivo" ] && { echo "[$archivo]"; cat "$archivo"; }
done
echo
echo "-- conexiones guardadas (solo nombres y archivos, nunca su contenido)"
nmcli -t -f NAME,TYPE,AUTOCONNECT,AUTOCONNECT-PRIORITY,FILENAME connection show 2>&1
ls /etc/netplan 2>&1
echo
echo "-- region del Wi-Fi"
iw reg get 2>&1 | head -4
echo
echo "-- driver del Wi-Fi (kernel)"
journalctl -b -k -o short-monotonic --no-pager 2>&1 | grep -i -E 'brcmfmac|wlan0|cfg80211|regulatory' | head -40
echo
echo "-- NetworkManager, wait-online y wpa_supplicant (este encendido)"
journalctl -b -o short-monotonic --no-pager -u NetworkManager \
  -u NetworkManager-wait-online -u wpa_supplicant 2>&1 | head -250
echo
echo "-- quien lanza netplan 'generate' y con que orden"
journalctl -b _COMM=generate -o verbose --no-pager \
  --output-fields=_PID,_CMDLINE,_EXE,_SYSTEMD_UNIT,MESSAGE 2>&1 | head -60
echo
echo "-- netplan y NetworkManager: versiones, enlace y archivos (sin contenido)"
dpkg-query -W network-manager netplan.io libnetplan1 2>&1
ldd /usr/sbin/NetworkManager 2>&1 | grep -i netplan
ls -l /etc/netplan /lib/netplan /run/netplan /run/NetworkManager/system-connections 2>&1
echo
echo "-- todo el sistema mientras NetworkManager partia (de 'Starting' a 'Started')"
desde=$(systemctl show -p InactiveExitTimestampMonotonic --value NetworkManager.service)
hasta=$(systemctl show -p ActiveEnterTimestampMonotonic --value NetworkManager.service)
journalctl -b -o short-monotonic --no-pager 2>&1 | awk -v a="$desde" -v b="$hasta" '
  { s = $0; sub(/^\[ */, "", s); t = (s + 0) * 1000000
    if (t >= a && t <= b) print }' | head -400
