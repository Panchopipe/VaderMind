# How it was built

[Español](es/como-se-hizo.md) · [Back to the README](../README.md)

VaderMind was built by one person between April and November 2026, working
with AI coding agents: first OpenAI Codex, later also Claude Code. The author
made the decisions and tested everything in the real world; the agents
implemented the changes, explained each trade-off and kept the documentation
alive.
This page is about the method that made a long project with interchangeable
agents work.

## Timeline

| When | Milestone |
| --- | --- |
| April 2026 | Terminal chat with short memory; two personalities; the working method |
| May to June | First voices (OpenAI TTS, then ElevenLabs); Spotify |
| September | Custom Piper voice trained and integrated; model pilot and the move to Vercel AI Gateway; LG TV in three layers; personality closed |
| Late September to October | Talking to Vader by voice; first latency work; Vader running on the Raspberry Pi inside the helmet |
| November | Learning to solder; the eyes; "Oye Vader" hands-free with lights and boot by itself; the speed stage; home profiles; a final code audit |

## The method

The core idea: **the repository, not the tool, is the source of truth.** Any
agent that reads the same files can pick up the work, and the author can switch
agents in the middle of the project. Nothing important lives only in the memory
of one chat.

### The files

| File | Role |
| --- | --- |
| `CODEX.md` | The short map of the project and the global rules. Each agent's own rules file (`CLAUDE.md` and so on) links to it instead of copying it |
| `ledger.md` | The quick operational handoff: append-only, newest entry first, one entry per session or decision. It ended with 95 entries |
| `docs/plan.md` | The consolidated state: what works, what was decided, what is pending |
| One handoff per phase | The starting point for each stage: voice input, Raspberry Pi, lights, speed, profiles |
| `docs/architecture.md` | The path through the code, its contracts and its limits |
| `docs/local-artifacts.md` | What Git does not keep (voice model, keys, tokens), with SHA-256 hashes to check copies |

Every session starts the same way: read the ledger, then the plan, then the
handoff of the active phase, and summarize the confirmed state **before**
proposing anything.

### A real ledger entry

This entry (translated from Spanish) shows the format: what changed, why, what
was verified and what is still pending.

> **2026-11-20 - Lights: the pupils light up only when Vader is ready**
>
> - Approved by the user after measuring A1. `Ws2812Lights` starts in `APAGADO`
>   (off) instead of `REPOSO` (idle): the eyes stay dark while Vader loads and
>   waits for the network, and `HandsFreeSession` lights the dim pupils when it
>   starts listening, together with the `reposo` mark. Before, they lit up
>   11.5 s too early and an "Oye Veider" said then was lost.
> - 402 tests OK on the laptop. Still to do: copy `vadermind/lights.py` to the
>   Pi and confirm with a cold boot; the stopwatch now measures until "ready"
>   (~36.5 s expected).

### Working rules

- **Pause only for real decisions.** The agent moves on by itself when the next
  step is clear, and stops when there is an architectural choice, several
  reasonable paths or a hidden risk. Then it explains the options in plain
  language and the author decides.
- **Explain, don't just change.** The author wanted to understand the why, so
  explanations are pedagogical and assume no prior knowledge (nor of
  soldering: the eyes were the author's first soldering project).
- **Report faithfully.** What was verified is kept apart from what was assumed.
  Automated tests and checks on the real devices are recorded separately, and
  anything not verified is marked so the next session does not assume it.
- **Say where each command runs.** With a laptop and a Pi in play, every
  command is labeled: on the Pi or in the laptop's PowerShell.
- **Small commits with intent**, and documentation updated whenever a decision
  or the operating state changes. No paperwork for its own sake.

## Deciding with numbers

Most choices were made by measuring, not by taste:

- **The conversation model.** A pilot on Vercel AI Gateway ran four models over
  20 scenarios, without touching real devices: 80 cases for US$0.047. Then a
  blind human review scored the answers: Gemini 3.5 Flash-Lite 4.16/5, GPT-5.6
  Luna 4.12, Claude Haiku 4.5 3.91 and gpt-oss-20b 3.54. No candidate passed
  every automatic gate, so the author chose a tiered setup: Gemini 3.5
  Flash-Lite for everyday turns and Gemini 3.7 Flash for complex requests and
  fallback. (The project had started on Claude through the Anthropic API.)
- **The personality.** A fixed bank of 15 questions scores every prompt change
  before anyone listens to it.
- **Transcription.** 15 of the author's own phrases, recorded with the real
  microphone at different distances, scored by error rate, latency and cost.
- **Speed.** A fixed 19-phrase script, three rounds, on the Pi. See
  [Performance](performance.md).
- **The voice.** By ear, clip by clip, and in blind comparisons.

## Closing with an audit

Before closing the project, one last session audited the whole code base and
changed nothing: security, failure paths and clutter, each finding with a
severity and a confidence level. All 85 commits were checked for secrets
(none was ever committed) and the 423 tests passed. Its findings became the
"Known limitations" in the [README](../README.md#known-limitations), to be
fixed soon.

## Some lessons

- A wall adapter can be the noisiest part of a voice project: it put a 50 Hz
  hum into the microphone. The power bank fixed it.
- On the Pi 4, GPIO18 (the usual WS2812 pin) shares its PWM with the audio
  jack. SPI on GPIO10 avoids the conflict.
- A wake-word engine's partial result is a guess. Counting only final results
  stopped Vader from waking up on a lone "oye".
- Lights can lie. Measuring the boot showed the eyes saying "ready" 11.5 s too
  early.
- Sending a Wake-on-LAN packet is not turning a TV on. Only an answer from the
  TV proves it.
- Faster is not always better: two speed tricks were dropped because they made
  Vader sound worse.

## What is not in this repository

This public copy is curated from a private repository. The ledger, the phase
handoffs and the plan stay private because they contain details of real homes
and networks; this page summarizes their method. The voice model and its
training clips are not published either.
