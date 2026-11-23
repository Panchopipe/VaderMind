# Checkpoint seleccionado

## `es_MX-ald-medium`

- Fuente: repositorio oficial `rhasspy/piper-checkpoints` en Hugging Face.
- Calidad: `medium`.
- Idioma del modelo: español de México.
- Voz de eSpeak: `es-419`, español latinoamericano.
- Hablantes: uno.
- Frecuencia: 22.050 Hz.
- Versión declarada en su configuración: Piper 1.0.0.
- Archivo: `epoch=9999-step=1753600.ckpt`.
- Tamaño verificado: `845898456` bytes.
- SHA-256: `77ee30f9a382594123ec4fb3ca53b437612819a44b3e3dc490f39f5a2dfefcce`.

Este checkpoint fue afinado originalmente desde la voz española `davefx-medium`.
Se eligió porque coincide simultáneamente con la arquitectura medium, la frecuencia
del dataset Vader y la fonetización latinoamericana.

El entorno utiliza el commit fijado `73c04d81d5590ecc46e522de3601ce7fb29fc2be`
del entrenador histórico `rhasspy/piper`, cuyo paquete `piper_train` declara la
versión 1.0.0. No debe sustituirse por `OHF-Voice/piper1-gpl` sin volver a validar
la carga del checkpoint, porque el formato de hiperparámetros cambió.
