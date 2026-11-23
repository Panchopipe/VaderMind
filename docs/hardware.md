# Hardware

[Español](es/hardware.md) · [Back to the README](../README.md)

Everything lives inside a two-piece helmet: the Raspberry Pi and the power
bank in the dome, the speakers and the eyes in the mask, the microphone behind
the chin grille. A cable with a connector halfway lets the two pieces come
apart.

## Parts

| Part | Details |
| --- | --- |
| Raspberry Pi 4 Model B, 4 GB | Raspberry Pi OS Lite 64-bit (Debian 13), Python 3.13, heatsinks and a fan |
| USB microphone | Behind the chin grille. Gain saved with `alsactl store` |
| Small speakers | Audio through the 3.5 mm jack; the USB cable only powers them |
| 2 x WS2812 modules, 7 LEDs, 22 mm | One per eye: a center LED (the "pupil") and a ring of 6 |
| 330 Ω resistor, 1000 µF 25 V capacitor | Protection for the LED data line and power |
| Power bank | USB-A output to the Pi's USB-C |

## The eyes

The eye openings are 4 cm tall and 7.5 cm wide. Each one has a single 7-LED
module behind the lens, inside a tube of black cardstock so the light does not
spill. The two modules are chained on one data line (DOUT of the first to DIN
of the second).

The effect that made it feel alive was the **pupil**: a test with a single
point of light in the center of each eye gave the impression of a person
inside the helmet. So the center LED is the pupil and the ring around it lights
up with the voice. A diffuser is optional, because it erases the point.

| State | What the eyes do (all red) |
| --- | --- |
| Booting | Dark, until Vader is really ready |
| Idle | Dim pupils |
| Listening | Ring on |
| Thinking | The ring breathes |
| Speaking | The ring follows the volume of each sentence |
| Error | Double blink |

### Wiring to the Pi

| Function | Physical pin | Note |
| --- | --- | --- |
| LED 5 V | 2 | The fan already uses pins 4 (5 V), 6 (GND) and 8 (GPIO14) |
| LED ground | 20 | |
| Data (DIN) | 19 (GPIO10, SPI MOSI) | Through the 330 Ω resistor, close to the first LED |

- The 1000 µF capacitor goes between 5 V and ground next to the first module,
  with the stripe (negative) to ground.
- **Not GPIO18.** It is the usual pin for WS2812, but on the Pi 4 its PWM is the
  same one that drives the 3.5 mm audio jack, where Vader's voice comes out.
- The Pi's 3.3 V data signal was enough without a level shifter, with a short
  data cable.

### Driving WS2812 over SPI

WS2812 LEDs expect precise pulses. Instead of a library, `lights.py` encodes the
signal itself and writes it to `/dev/spidev0.0` with plain `ioctl` calls (no
`spidev` package, which the virtual environment could not see):

- SPI at 2.4 MHz, so each LED bit becomes 3 SPI bits (`100` for 0, `110` for 1).
- Colors in GRB order, brightness 10 % by default and capped at 25 % in code.
- A thread animates both eyes at about 30 frames per second.
- In `/boot/firmware/config.txt`: `dtparam=spi=on`, plus `core_freq=500` and
  `core_freq_min=500` to fix the core clock, so the SPI timing does not drift
  with load.

`scripts/ws2812_spi_test.py` tests a single LED; `scripts/lights_demo.py` walks
through every state and helps find which LED index is the pupil.

## Power

- The Pi runs from a **power bank**. The wall adapter of the kit put a 50 Hz
  hum into the microphone; the power bank does not.
- It must be the power bank's **USB-A output**. Through its USB-C output, the
  power bank's protection cut the power at boot, because of the inrush into
  the eyes' capacitor.
- `vcgencmd get_throttled` stayed at `0x0` (no undervoltage) with the eyes
  animated and the voice playing.
- Inside the closed helmet the Pi went from 31.6 to 42.3 °C after five minutes
  of nonstop speech, without slowing down.

## Audio

- Speakers on the 3.5 mm jack (`aplay` uses the default ALSA output). Card
  numbers can change when other USB devices are plugged in.
- The USB microphone only records at 44.1 or 48 kHz. Vader records at 48 kHz,
  resamples to 16 kHz and removes everything below ~90 Hz (hum and handling
  noise).
- With the microphone behind the chin grille, the speech recognition word
  error rate was 2.6 % at 1 m and 0.9 % at 2 m, over 15 fixed phrases
  (`scripts/benchmark_stt.py`). `scripts/measure_mic_noise.py` measures the
  noise floor.

## Software on the Pi

- Piper `2023.11.14-2` for `aarch64`, running as a persistent process.
- The custom voice model is copied separately and checked with `sha256sum -c`.
- `deploy/vadermind.service` starts Vader at boot; logs with
  `journalctl -u vadermind -f`.
- `scripts/medicion.sh` and `scripts/latency_report.py` measure boot and turn
  times; `scripts/diagnostico_red.sh` explains a slow network at boot (read
  only).
- *"Vader, apágate"* shuts the Pi down cleanly after a spoken confirmation, so
  the SD card is not corrupted by pulling the power.
