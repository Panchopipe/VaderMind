# Velocidad

[English](../performance.md) · [Volver al README](../../README.es.md)

Un asistente de voz que se demora parece roto. La última gran etapa del
proyecto fue la velocidad: medir todo en la Pi y recortar cada décima de
segundo que se pudiera **sin perder calidad**. La calidad se juzgó a oído: una
respuesta más rápida que suena peor no cuenta.

## Primero medir

Nada de esto cambia el comportamiento; solo lee el reloj.

- `telemetry.TURN_TIMELINE` registra hitos de cada turno hablado: fin de la
  palabra de activación, micrófono abierto, inicio y fin del habla,
  transcripción lista, primera palabra del modelo, inicio de la síntesis y
  primer sonido.
- Cada turno escribe una línea `Tiempos [tipo] ...` en el log y cada arranque
  una línea `Arranque ...`. `scripts/latency_report.py` las resume.
- Un **guion fijo** de 19 frases cubre cada tipo de turno (charla, respuesta
  larga, hora, hora en otra ciudad, clima, Spotify a través del modelo y por
  regla local, estado y acciones del TV, cambios de modo, "silencio"), repetido
  tres veces.

## Dónde se iba el tiempo (línea base en la Pi, 18/11/2026)

Desde que callas hasta el primer sonido de Vader, por tramo (medianas):

| Tramo | Mediana | Notas |
| --- | ---: | --- |
| Pensar | 1,01 s | Casi todo es la primera palabra del modelo: 0,98 s con ~3.000 tokens de entrada |
| Silencio | 0,80 s | Fijo: la espera que decide que terminaste |
| Primer trozo de voz | 0,77 s | Piper: ~0,3 s más ~0,02 s por carácter de la primera oración |
| Transcripción | 0,42 s | La especulación se usó en 58 de 58 turnos |
| Micro, luces, pausas | ≤ 0,03 s | Ya resueltos |

Antes, en el laptop (octubre de 2026), una primera ronda ya había bajado un
turno de conversación de 4,2 s a unos 2,5 s y una orden al TV de unos 4,4 s a
0,05 s, con un proceso Piper persistente, conexiones precalentadas,
transcripción especulativa, caché del clima y una conexión persistente al TV.

## Lo que se cambió

| Cambio | Efecto en la Pi |
| --- | --- |
| Transcripción especulativa a 192 ms en vez de 320 ms (misma precisión con las grabaciones del autor) | Transcripción 0,42 → 0,25 s |
| Precalentar la conexión al TV en cada "Oye Vader", en un hilo | Primera orden al TV 7,29 → 3,10 s |
| Transcripción en streaming con `grok-stt` y palabras clave en español | Transcripción 0,25 → 0,00 s en 19 de 19 turnos; unos 0,3 s menos hasta el primer sonido |

En total: **unos 0,45 s menos por turno con voz** frente a la línea base. Con
streaming, el texto suele estar listo antes de que terminen los 800 ms de
silencio. Con las grabaciones del laptop, su error por palabra fue de 5,8 %,
frente a 7,1 % del camino REST.

## Lo que se probó y se descartó

- **Acortar el primer trozo**, para que Piper empiece antes. Probado a oído:
  los cortes le quitaban naturalidad a la entonación.
- **Pedirle al modelo una primera oración corta.** Sonó mejor que cortar, pero
  Vader perdía chispa.

Las dos son más rápidas en el papel. Ninguna se quedó.

## Arranque

| | Enchufar → listo |
| --- | ---: |
| Primera medición | ~42 s |
| Cargar mientras conecta el Wi-Fi | **36 s** |

El servicio de systemd esperaba la red antes de lanzar Python. Ahora Vader
carga Vosk, Silero y Piper mientras el Wi-Fi conecta, y recién después espera
la red: de Linux a listo bajó de 31,5 a 25,5 s.

Una medición también pilló a los ojos mintiendo: las pupilas tenues se
encendían 11,5 s antes de que Vader escuchara de verdad, y un "Oye Vader" dicho
en ese rato se perdía. Ahora los ojos quedan oscuros hasta que Vader está
listo, y la palabra dicha apenas se encienden funciona al primer intento.

Dónde se van los 36 s que quedan: unos 11 s de firmware, y luego Linux y la
red. Un diagnóstico de solo lectura (`scripts/diagnostico_red.sh`) encontró
que NetworkManager pasa unos 8 s regenerando su configuración cuatro veces y
2,6 s esperando un cable de red que no está. Las opciones para recuperar ese
tiempo están documentadas pero no aplicadas: 36 s se consideró suficiente.
