# Entrenamiento Vader Piper v1 en RunPod

## Estado final

Ejecucion completada el 2 de septiembre de 2026.

- GPU usada: RTX 3090 de 24 GB, On-Demand a USD 0,50/h segun la consola.
- Preflight: correcto.
- Setup: correcto, PyTorch 1.13.1 con CUDA disponible.
- Dataset procesado: 398 clips.
- Smoke test: correcto y revisado auditivamente.
- Entrenamiento final: termino en `epoch=10999-step=1777600.ckpt`.
- Exportacion ONNX: correcta.
- Archivo descargado: `es_CL-vader-medium.tar.gz`, 58 MB mostrados por RunPod.
- SHA-256 del archivo descargado:
  `0b1b98203c00ac84238b06a63bffe1e07e89f5c874137597aec2503973d5b891`.
- Modelo extraido en `models/piper/es_CL-vader-medium/`.
- Inferencia local Windows y prueba completa VaderMind: correctas.

Los warnings de tracing ONNX durante la exportacion no impidieron crear ni usar
el modelo. La corrida requirio limitar el preprocesamiento a dos trabajadores y
los hilos numericos a uno para evitar `Resource temporarily unavailable`.

Este paquete entrena una voz Piper `medium` con 398 clips aprobados y 35,933
minutos de audio. El entrenamiento parte del checkpoint latinoamericano
`es_MX-ald-medium` y produce un modelo ONNX para VaderMind y Raspberry Pi.

## Pod recomendado

- GPU: RTX 3090 de 24 GB.
- Modalidad: Pod bajo demanda, no Spot/interrumpible.
- Imagen: `runpod/pytorch:1.13.0-py3.10-cuda11.7.1-devel`.
- Volumen persistente: 50 GB montado en `/workspace`.
- Disco del contenedor: 30 GB.
- RAM recomendada: 24 GB o más.

La imagen antigua es intencional: conserva PyTorch 1.x, compatible con el
checkpoint Piper 1.0.0. No selecciones la plantilla PyTorch más nueva para esta
ejecución.

## Secuencia segura

Sube el ZIP a `/workspace`, abre el terminal de RunPod y ejecuta:

```bash
cd /workspace
unzip vader_piper_runpod_398_v1.zip
cd vader_piper_runpod_398_v1

bash scripts/00_preflight.sh
bash scripts/01_setup.sh
bash scripts/02_download_checkpoint.sh
bash scripts/03_preprocess.sh
bash scripts/04_smoke_test.sh
```

El smoke test usa 50 clips y solo diez épocas adicionales. Su objetivo no es
producir la voz final: confirma que checkpoint, GPU, preprocesamiento, guardado,
inferencia y exportación ONNX funcionan juntos.

Antes de continuar, descarga o escucha los WAV ubicados en:

```text
/workspace/vader_piper_run/training_smoke/samples
```

Si los archivos existen y contienen voz inteligible, inicia la corrida completa:

```bash
bash scripts/05_train_full.sh
bash scripts/06_export.sh
```

El artefacto que debes descargar al finalizar será:

```text
/workspace/vader_piper_run/es_CL-vader-medium.tar.gz
```

## Controles de gasto y recuperación

- `05_train_full.sh` tiene un límite máximo de 36 horas.
- Se guarda un checkpoint cada 25 épocas.
- Datos, checkpoints y exportación viven bajo `/workspace`, no en el disco
  efímero del contenedor.
- El entrenamiento completo siempre parte del checkpoint base; no continúa
  desde el smoke test de 50 audios.
- Si el entrenamiento falla, no borres el volumen persistente: conserva logs y
  checkpoints para diagnosticar sin repetir descargas ni preprocesamiento.
- Después de descargar el modelo, detén el Pod. El almacenamiento persistente
  puede seguir facturándose hasta eliminar el volumen.

## Parámetros principales

- `batch_size=32`
- precisión de 32 bits
- preprocesamiento limitado a 2 trabajadores para evitar agotar hilos del sistema
- validación: 5 %
- máximo de 400 identificadores fonéticos
- checkpoint cada 25 épocas
- semilla interna de Piper: 1234
- tope absoluto: epoch 11000, equivalente a unas 1000 épocas adicionales

Todos los valores modificables y las identidades verificadas están en
`config.env`. No cambies arquitectura, frecuencia, idioma fonético o checkpoint
sin repetir primero el smoke test.
