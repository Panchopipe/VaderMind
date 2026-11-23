# Hardware

[English](../hardware.md) · [Volver al README](../../README.es.md)

Todo vive dentro de un casco de dos piezas: la Raspberry Pi y el power bank en
la cúpula, los parlantes y los ojos en la máscara, y el micrófono detrás de la
rejilla de la barbilla. Un cable con conector a mitad de camino permite
separar las dos piezas.

## Piezas

| Pieza | Detalle |
| --- | --- |
| Raspberry Pi 4 Model B, 4 GB | Raspberry Pi OS Lite 64 bits (Debian 13), Python 3.13, disipadores y ventilador |
| Micrófono USB | Detrás de la rejilla de la barbilla. Ganancia guardada con `alsactl store` |
| Parlantes pequeños | Audio por el jack de 3,5 mm; el cable USB solo los alimenta |
| 2 módulos WS2812 de 7 LEDs, 22 mm | Uno por ojo: un LED central (la "pupila") y un anillo de 6 |
| Resistencia de 330 Ω y condensador de 1000 µF 25 V | Protección de la línea de datos y de la alimentación de los LEDs |
| Power bank | De su salida USB-A al USB-C de la Pi |

## Los ojos

Las aberturas de los ojos miden 4 cm de alto por 7,5 cm de ancho. Cada una
tiene un solo módulo de 7 LEDs detrás del lente, dentro de un tubo de
cartulina negra para que la luz no se escape. Los dos módulos van encadenados
en una sola línea de datos (DOUT del primero al DIN del segundo).

El efecto que lo hizo sentirse vivo fue la **pupila**: una prueba con un solo
punto de luz al centro de cada ojo dio la sensación de una persona dentro del
casco. Por eso el LED central hace de pupila y el anillo se enciende con la
voz. El difusor es opcional, porque borra el punto.

| Estado | Qué hacen los ojos (todo en rojo) |
| --- | --- |
| Arrancando | Oscuros, hasta que Vader está realmente listo |
| Reposo | Pupilas tenues |
| Escuchando | Anillo encendido |
| Pensando | El anillo respira |
| Hablando | El anillo sigue el volumen de cada oración |
| Error | Doble parpadeo |

### Conexión a la Pi

| Función | Pin físico | Nota |
| --- | --- | --- |
| 5 V de los LEDs | 2 | El ventilador ya usa los pines 4 (5 V), 6 (GND) y 8 (GPIO14) |
| GND de los LEDs | 20 | |
| Datos (DIN) | 19 (GPIO10, SPI MOSI) | A través de la resistencia de 330 Ω, cerca del primer LED |

- El condensador de 1000 µF va entre 5 V y GND junto al primer módulo, con la
  franja (el negativo) hacia GND.
- **GPIO18 no.** Es el pin habitual para WS2812, pero en la Pi 4 su PWM es el
  mismo que maneja el jack de audio de 3,5 mm, por donde sale la voz de Vader.
- La señal de datos de 3,3 V de la Pi bastó sin conversor de nivel, con un
  cable de datos corto.

### WS2812 por SPI

Los LEDs WS2812 esperan pulsos precisos. En vez de una librería, `lights.py`
codifica la señal y la escribe en `/dev/spidev0.0` con llamadas `ioctl`
directas (sin el paquete `spidev`, que el entorno virtual no veía):

- SPI a 2,4 MHz, así cada bit del LED se vuelve 3 bits SPI (`100` para 0, `110`
  para 1).
- Colores en orden GRB, brillo de 10 % por defecto y tope de 25 % en el código.
- Un hilo anima ambos ojos a unos 30 cuadros por segundo.
- En `/boot/firmware/config.txt`: `dtparam=spi=on`, más `core_freq=500` y
  `core_freq_min=500` para fijar el reloj del núcleo, así los tiempos del SPI
  no cambian con la carga.

`scripts/ws2812_spi_test.py` prueba un solo LED; `scripts/lights_demo.py`
recorre todos los estados y ayuda a encontrar qué LED es la pupila.

## Energía

- La Pi funciona con un **power bank**. La fuente de pared del kit metía un
  zumbido de 50 Hz al micrófono; el power bank no.
- Tiene que ser la **salida USB-A** del power bank. Por su salida USB-C, la
  protección del power bank cortaba la energía al arrancar, por la corriente
  inicial que pide el condensador de los ojos.
- `vcgencmd get_throttled` se mantuvo en `0x0` (sin bajo voltaje) con los ojos
  animados y la voz sonando.
- Con el casco cerrado, la Pi pasó de 31,6 a 42,3 °C tras cinco minutos de
  habla continua, sin bajar la velocidad.

## Audio

- Parlantes en el jack de 3,5 mm (`aplay` usa la salida ALSA por defecto). El
  número de cada tarjeta puede cambiar al conectar otros dispositivos USB.
- El micrófono USB solo graba a 44,1 o 48 kHz. Vader graba a 48 kHz, convierte
  a 16 kHz y quita lo que está bajo ~90 Hz (zumbido y ruido de manipulación).
- Con el micrófono detrás de la rejilla de la barbilla, el error por palabra de
  la transcripción fue de 2,6 % a 1 m y 0,9 % a 2 m, con 15 frases fijas
  (`scripts/benchmark_stt.py`). `scripts/measure_mic_noise.py` mide el ruido de
  fondo.

## Software en la Pi

- Piper `2023.11.14-2` para `aarch64`, como proceso persistente.
- El modelo de la voz propia se copia aparte y se comprueba con `sha256sum -c`.
- `deploy/vadermind.service` arranca Vader al encender; los registros con
  `journalctl -u vadermind -f`.
- `scripts/medicion.sh` y `scripts/latency_report.py` miden los tiempos de
  arranque y de cada turno; `scripts/diagnostico_red.sh` explica una red lenta
  al arrancar (solo lectura).
- *"Vader, apágate"* apaga la Pi de forma ordenada tras una confirmación
  hablada, para no dañar la tarjeta SD al desenchufar.
