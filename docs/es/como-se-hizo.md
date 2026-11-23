# Cómo se hizo

[English](../how-it-was-built.md) · [Volver al README](../../README.es.md)

VaderMind lo construyó una persona entre abril y noviembre de 2026, trabajando
con agentes de programación con IA: primero OpenAI Codex, después también
Claude Code. El autor tomó las decisiones y probó todo en el mundo real; los
agentes implementaron los cambios, explicaron cada costo y beneficio y
mantuvieron viva la documentación. Esta página trata del método que permitió
llevar un proyecto largo con agentes intercambiables.

## Cronología

| Cuándo | Hito |
| --- | --- |
| Abril de 2026 | Chat en la terminal con memoria corta; dos personalidades; el método de trabajo |
| Mayo a junio | Primeras voces (OpenAI TTS, luego ElevenLabs); Spotify |
| Septiembre | Voz Piper propia entrenada e integrada; piloto de modelos y paso a Vercel AI Gateway; TV LG en tres capas; personalidad cerrada |
| Fines de septiembre a octubre | Hablarle a Vader por voz; primer trabajo de latencia; Vader en la Raspberry Pi dentro del casco |
| Noviembre | Aprender a soldar; los ojos; "Oye Vader" manos libres con luces y arranque solo; la etapa de velocidad; perfiles por hogar; una auditoría final del código |

## El método

La idea central: **la fuente de verdad es el repositorio, no la herramienta.**
Cualquier agente que lea los mismos archivos puede retomar el trabajo, y el
autor puede cambiar de agente a mitad del proyecto. Nada importante vive solo
en la memoria de un chat.

### Los archivos

| Archivo | Rol |
| --- | --- |
| `CODEX.md` | El mapa corto del proyecto y las reglas globales. El archivo de reglas de cada agente (`CLAUDE.md` y otros) lo enlaza en vez de copiarlo |
| `ledger.md` | El traspaso operativo rápido: solo se agregan entradas, la más nueva arriba, una por sesión o decisión. Terminó con 95 entradas |
| `docs/plan.md` | El estado consolidado: qué funciona, qué se decidió, qué falta |
| Un traspaso por fase | El punto de partida de cada etapa: entrada por voz, Raspberry Pi, luces, velocidad, perfiles |
| `docs/architecture.md` | El recorrido del código, sus contratos y sus límites |
| `docs/local-artifacts.md` | Lo que Git no guarda (modelo de voz, claves, tokens), con hashes SHA-256 para comprobar las copias |

Cada sesión empieza igual: leer el ledger, luego el plan, luego el traspaso de
la fase activa, y resumir el estado confirmado **antes** de proponer nada.

### Una entrada real del ledger

Esta entrada muestra el formato: qué cambió, por qué, qué se verificó y qué
falta.

> **2026-11-20 - Luces: las pupilas se encienden solo con Vader listo**
>
> - Aprobado por el usuario tras la medicion de A1. `Ws2812Lights` nace en
>   `APAGADO` en vez de `REPOSO`: los ojos quedan oscuros mientras Vader carga y
>   espera la red, y `HandsFreeSession` enciende las pupilas tenues al quedar
>   escuchando (lo hacia desde antes), junto con la marca `reposo`. Antes se
>   encendian 11,5 s antes de tiempo y un "Oye Veider" ahi se perdia.
> - 402 pruebas OK en el laptop. Falta copiar `vadermind/lights.py` a la Pi y
>   confirmar con un encendido en frio: el cronometro ahora mide hasta "listo"
>   (~36,5 s esperados).

### Reglas de trabajo

- **Pausar solo ante decisiones reales.** El agente avanza solo cuando el
  siguiente paso está claro, y se detiene cuando hay una decisión de
  arquitectura, varios caminos razonables o un riesgo escondido. Ahí explica
  las opciones en lenguaje simple y el autor decide.
- **Explicar, no solo cambiar.** El autor quería entender el porqué, así que
  las explicaciones son pedagógicas y no asumen conocimientos previos (tampoco
  de soldadura: los ojos fueron el primer proyecto de soldadura del autor).
- **Informar con fidelidad.** Lo verificado se separa de lo supuesto. Las
  pruebas automatizadas y las comprobaciones en los equipos reales se
  registran por separado, y lo que no se verificó queda marcado para que la
  siguiente sesión no lo dé por probado.
- **Decir dónde corre cada comando.** Con un laptop y una Pi en juego, cada
  comando lleva su etiqueta: en la Pi o en PowerShell del laptop.
- **Commits pequeños y con intención**, y la documentación al día cada vez que
  cambia una decisión o el estado operativo. Sin papeleo por el papeleo.

## Decidir con números

Casi todas las elecciones se hicieron midiendo, no por gusto:

- **El modelo conversacional.** Un piloto en Vercel AI Gateway corrió cuatro
  modelos sobre 20 escenarios, sin tocar equipos reales: 80 casos por
  US$0,047. Después una revisión humana a ciegas puntuó las respuestas: Gemini
  3.5 Flash-Lite 4,16/5, GPT-5.6 Luna 4,12, Claude Haiku 4.5 3,91 y gpt-oss-20b
  3,54. Ningún candidato pasó todas las barreras automáticas, así que el autor
  eligió un esquema por niveles: Gemini 3.5 Flash-Lite para el día a día y
  Gemini 3.7 Flash para lo complejo y como respaldo. (El proyecto había partido
  con Claude a través de la API de Anthropic).
- **La personalidad.** Un banco fijo de 15 preguntas puntúa cada cambio de
  prompt antes de escucharlo.
- **La transcripción.** 15 frases propias del autor, grabadas con el micrófono
  real a distintas distancias, puntuadas por error, latencia y costo.
- **La velocidad.** Un guion fijo de 19 frases, tres rondas, en la Pi. Ver
  [Velocidad](velocidad.md).
- **La voz.** A oído, clip por clip, y en comparaciones a ciegas.

## Cerrar con una auditoría

Antes de cerrar el proyecto, una última sesión auditó todo el código sin
cambiar nada: seguridad, caminos de falla y desorden, cada hallazgo con su
gravedad y su nivel de confianza. Se revisaron los 85 commits en busca de
secretos (nunca entró ninguno) y pasaron las 423 pruebas. Sus hallazgos se
convirtieron en las "Limitaciones conocidas" del
[README](../../README.es.md#limitaciones-conocidas), que se corregirán pronto.

## Algunas lecciones

- Una fuente de pared puede ser lo más ruidoso de un proyecto de voz: metía un
  zumbido de 50 Hz al micrófono. El power bank lo resolvió.
- En la Pi 4, GPIO18 (el pin habitual para WS2812) comparte su PWM con el jack
  de audio. SPI en GPIO10 evita el conflicto.
- El resultado parcial de un detector de palabra de activación es una
  adivinanza. Contar solo los resultados finales evitó que Vader despertara con
  un "oye" suelto.
- Las luces pueden mentir. Medir el arranque mostró a los ojos diciendo
  "listo" 11,5 s antes de tiempo.
- Enviar un paquete Wake-on-LAN no es encender un TV. Solo una respuesta del TV
  lo demuestra.
- Más rápido no siempre es mejor: se descartaron dos trucos de velocidad
  porque hacían sonar peor a Vader.

## Lo que no está en este repositorio

Esta copia pública se cura desde un repositorio privado. El ledger, los
traspasos por fase y el plan quedan privados porque contienen detalles de
hogares y redes reales; esta página resume su método. Tampoco se publican el
modelo de voz ni sus clips de entrenamiento.
