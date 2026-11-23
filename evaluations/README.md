# Evaluacion de modelos conversacionales

Este documento conserva el procedimiento del piloto historico que comparo cuatro
modelos. El proveedor operativo elegido despues del piloto es Vercel AI Gateway,
con Gemini 3.5 Flash-Lite para el uso cotidiano y Gemini 3.7 Flash para
complejidad explicita y fallback. Gemini 3.7 Flash es la ruta prevista para web
cuando se reactive; actualmente solo hora y clima estan habilitados.

Los candidatos del piloto fueron:

- `anthropic/claude-haiku-4.5` como baseline
- `openai/gpt-5.6-luna`
- `google/gemini-3.5-flash-lite`
- `openai/gpt-oss-20b`, candidato de pesos abiertos

Vercel AI Gateway es el gateway principal del piloto. OpenRouter se conserva como
alternativa tecnica, pero no es necesario para ejecutar la evaluacion. Si un
candidato no tiene una ruta Zero Data Retention elegible, el evaluador intenta
usar `alibaba/qwen-3-14b` como reserva. Las tools de luces, television,
electrodomesticos y web son simulaciones: el benchmark nunca controla dispositivos
reales.

## Privacidad

Por defecto, cada solicitud del evaluador enviada por Vercel fuerza:

- `zeroDataRetention=true`
- `disallowPromptTraining=true`
- un unico proveedor durante el benchmark, sin fallback silencioso

El preflight descarta endpoints que no publiquen tanto ZDR como no-training. Como
defensa adicional, Vercel permite activar esas mismas restricciones para todo el
equipo desde AI Gateway Settings.

Para repetir el piloto en Hobby, `--allow-non-zdr` omite exclusivamente
`zeroDataRetention`; `disallowPromptTraining` permanece activo. Esta opcion afecta
al evaluador. La configuracion operativa actual tambien usa
`AI_GATEWAY_REQUIRE_ZDR=false`, decision que debe reevaluarse antes de tratar datos
sensibles.

## Preparacion para repetir el piloto historico

1. Crea una API key en Vercel AI Gateway.
2. Para repetir el benchmark con ZDR, usa un equipo Vercel Pro o Enterprise.
   En Hobby, el evaluador necesita `--allow-non-zdr` para omitir ZDR; no-training
   permanece activo.
3. Copia la clave a `.env`:

```env
AI_GATEWAY_API_KEY=tu-clave-de-vercel
```

No cambies el `LLM_PROVIDER=vercel` operativo para repetir el benchmark: el
evaluador selecciona su gateway con `--gateway`.

## Paso 1: preflight sin gasto de modelos

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_llm_models --preflight-only
```

El preflight consulta endpoints ZDR, exige soporte de tools, al menos 32K de
contexto y 99% de uptime diario, y fija para cada modelo el proveedor elegible con
menor latencia p50 reciente.

Importante: que el preflight pase confirma el catalogo, no el plan de la cuenta.
El smoke test siguiente confirma autenticacion, facturacion y permiso ZDR.

Luego valida autenticacion y facturacion con una sola respuesta minima:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_llm_models --smoke-only
```

## Paso 2: piloto completo

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_llm_models
```

El comando ejecuta 20 casos, tres veces por modelo, en orden aleatorio. Detiene la
prueba al llegar al tope de US$15. Los resultados quedan localmente bajo
`evaluations/results/`, carpeta ignorada por Git para no publicar respuestas.

El piloto reducido del 7 de septiembre de 2026 se ejecuto con una repeticion y un
tope de US$5:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_llm_models --gateway vercel --allow-non-zdr --repetitions 1 --max-cost-usd 5
```

`gpt-oss-20b` queda fijado a Groq durante este benchmark porque Novita rechazaba el
esquema de tools anunciado por el catalogo. Esto evita confundir una incompatibilidad
del endpoint con la capacidad del modelo.

Cada corrida genera:

- `resultados.jsonl`: datos técnicos de cada solicitud
- `resumen_objetivo.md`: latencia, tools, seguridad y costo
- `revision_ciega.csv`: formulario sin nombres de modelos
- `clave_modelos.json`: equivalencia entre alias y modelo

## Paso 3: revision humana

Sin abrir `clave_modelos.json`, califica cada fila de `revision_ciega.csv` de 1 a
5 en calidad y personalidad. Luego ejecuta:

```powershell
.\.venv\Scripts\python.exe -m scripts.score_llm_evaluation evaluations\results\AAAAMMDD_HHMMSS
```

Se generara `decision.md` con el puntaje 40% velocidad, 30% precision/seguridad,
20% revision humana y 10% costo. Un reemplazo necesita superar todas las barreras
y reducir al menos 25% la mediana de latencia frente a Haiku.

La prueba alternada de 20 turnos reales con Piper sigue pendiente como validacion
adicional de la seleccion operativa ya realizada:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_blind_voice_final evaluations\results\AAAAMMDD_HHMMSS
```

## Activacion elegida despues del benchmark

El benchmark no produjo un ganador automatico, pero el usuario eligio una
arquitectura por niveles centralizada en Vercel:

```env
LLM_PROVIDER=vercel
VADER_FAST_MODEL=google/gemini-3.5-flash-lite
VADER_QUALITY_MODEL=google/gemini-3.7-flash
AI_GATEWAY_REQUIRE_ZDR=false
```

Las ordenes Spotify reconocidas siguen pasando primero por el router local; las
ambiguas usan Gemini 3.5 Flash-Lite. Gemini 3.7 Flash atiende solicitudes
explicitamente complejas y el fallback de conversacion mediante Vercel. Su ruta
de busqueda web permanece en el codigo para una futura reactivacion, pero el
catalogo operativo no la habilita: solo hora y clima pueden consultar informacion
actual. Spotify no repite automaticamente acciones externas.

El evaluador conserva OpenRouter como alternativa mediante `--gateway openrouter`
y `OPENROUTER_API_KEY`; no es el proveedor operativo vigente.

## Banco de personalidad

Compara cambios del prompt de personalidad con las mismas 15 preguntas, dos veces
cada una, en canon y chileno. Cada pregunta parte con una sesion vacia; si el
modelo pide una herramienta se registra, pero no se ejecuta.

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_persona NOMBRE
.\.venv\Scripts\python.exe -m scripts.evaluate_persona --compare CARPETA_A CARPETA_B
```

Mide largo, oraciones, preguntas de vuelta, muletillas, descripciones fisicas,
markdown, ruptura de personaje, chilenismos, garabatos y puntuacion de "weon".
Los resultados quedan en `evaluations/results/persona_*`, fuera de Git. Los
ejemplos de tono del prompt no deben repetir preguntas del banco; un test lo vigila.
