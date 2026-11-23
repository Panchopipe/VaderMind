# Voz

[English](../voice.md) · [Volver al README](../../README.es.md)

Vader está hecho para escucharse, no para leerse. Esta página cuenta cómo se
eligió la voz, cómo se entrenó la voz propia y cómo se produce y se oye la voz
durante el uso.

## Elegir el motor de voz

| Motor | Probado | Resultado |
| --- | --- | --- |
| OpenAI TTS (`gpt-4o-mini-tts`, voz `onyx`) | mayo de 2026 | Funcionó desde el primer día; lejos del personaje |
| ElevenLabs Flash, con streaming | junio de 2026 | El primer sonido más temprano; solo en la nube y con costo por carácter |
| **Piper, voz propia** | septiembre de 2026 | Elegido: corre en la Pi, sin internet y con tiempos estables |

Un benchmark de cinco repeticiones en el laptop, con el mismo texto de 168
caracteres:

| | Piper (local) | ElevenLabs Flash (streaming) |
| --- | ---: | ---: |
| Primer audio disponible | 1,19 s | 0,53 s |
| Audio completo generado | 1,19 s | 1,67 s |
| Variación entre repeticiones | 7 ms | 458 ms hasta el primer audio |

ElevenLabs empieza a hablar antes, pero Piper termina antes, nunca depende de
la red y corre en una Raspberry Pi. OpenAI y ElevenLabs siguen en el código
como alternativas (`VADER_TTS_PROVIDER`).

## Entrenar la voz propia

Las voces de [Piper](https://github.com/rhasspy/piper) son modelos VITS
pequeños exportados a ONNX. Entrenar una desde cero necesita muchas horas de
audio; afinar un checkpoint existente necesita mucho menos.

1. **Frases.** 400 frases escritas para esto: 340 frases cotidianas de
   asistente (confirmaciones, preguntas, explicaciones largas) y 60 en
   español chileno coloquial.
2. **Clips sintéticos.** Los clips se generaron con un modelo de texto a voz
   de código abierto en español latinoamericano, se ralentizaron al 85 %
   (`atempo=0.85`) para darles peso y gravedad, y se revisaron a oído uno por
   uno. Pasaron 398, que suman 35,9 minutos de audio (WAV mono, 16 bits,
   22,05 kHz).
3. **Checkpoint base.** `es_MX-ald-medium` de
   [rhasspy/piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints):
   la misma arquitectura `medium`, la misma frecuencia de muestreo y fonemas
   latinoamericanos (`es-419`).
4. **Ajuste fino** en una RTX 3090 arrendada (RunPod, unos US$0,50 por hora),
   con el entrenador histórico `rhasspy/piper` fijado en un commit: primero una
   prueba corta con 50 clips y 10 épocas, y después 1.000 épocas más sobre el
   checkpoint base (de la época 9.999 a la 10.999), con lotes de 32.
5. **Exportación** a `es_CL-vader-medium.onnx` y su configuración JSON,
   verificados con hashes SHA-256 antes de cada copia a la Pi.

La receta completa, con sus comprobaciones previas, está en
[`cloud_training/runpod_piper_v1/`](../../cloud_training/runpod_piper_v1/). Dos
lecciones de ahí: el primer preprocesamiento se quedó sin procesos e hilos (se
arregló limitando los workers), y el entrenador no debe reemplazarse por el
fork GPL más nuevo sin volver a validar, porque el formato del checkpoint
cambió.

**No se publica:** los clips de entrenamiento, el material con que se
generaron ni el modelo de voz resultante.

**El resultado.** Conserva el timbre buscado y se entiende claramente, pero
pierde algo de naturalidad y de entonación. Más entrenamiento podría corregir
palabras puntuales, pero Piper tiene un techo expresivo más bajo que modelos
más pesados, y un dataset pequeño se sobreajusta con facilidad. Se aceptó ese
costo a cambio de una voz rápida y local.

## Hablar durante el uso

- **Un proceso persistente.** Piper parte una vez con el modelo cargado y
  recibe una línea de texto por respuesta en `stdin`. Lanzarlo en cada
  respuesta obligaba a recargar el modelo cada vez.
- **Oración por oración.** La respuesta se divide en oraciones (y las largas en
  las comas). Un hilo las sintetiza en orden y la primera suena apenas está
  lista. En la Pi, una respuesta larga tardaba hasta 4 s en empezar; su primera
  oración tarda alrededor de 1 s.
- **Reproducción.** `aplay` en Linux, `winsound` en Windows. La envolvente de
  volumen de cada WAV mueve el anillo de los ojos mientras suena.
- **Solo texto limpio.** `response_text.py` quita markdown, listas y
  acotaciones (Vader nunca narra su respiración). También aplica una regla del
  chileno hablado: *"weón"* abre una oración con coma (*"Weón, ..."*) pero la
  cierra sin ella (*"... weón."*).
- **Respiración.** El primer turno de escucha de cada sesión parte con la
  respiración de un respirador. Nada se tomó de las películas: los sonidos son
  ruido moldeado que genera `scripts/generate_breathing_cues.py`.

## Oír

- **Palabra de activación en el equipo** con Vosk (`vosk-model-small-es-0.42`)
  y una gramática cerrada, así nada va a la nube antes de *"Oye Vader"*.
  `scripts/wake_word_test.py` cuenta aciertos y falsas activaciones.
- **Fin del habla** con Silero VAD: 800 ms de silencio cierran el turno.
- **La transcripción** se eligió con un benchmark de 15 frases propias del
  autor, grabadas con el micrófono real y puntuadas por error por palabra,
  latencia y costo (`scripts/benchmark_stt.py`). Hoy: primero `grok-stt` en
  streaming, luego `gpt-4o-transcribe` con contexto chileno y luego `grok-stt`
  por REST. Cada solicitud le pide al gateway no entrenar con el audio.

El detalle de cada paso está en [Cómo funciona](como-funciona.md).
