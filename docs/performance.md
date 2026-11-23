# Performance

[Español](es/velocidad.md) · [Back to the README](../README.md)

A voice assistant that takes too long feels broken. The last big stage of the
project was speed: measure everything on the Pi and cut every tenth of a
second that could go **without losing quality**. Quality was judged by ear: a
faster answer that sounds worse does not count.

## Measuring first

Nothing changes behavior here; it only reads the clock.

- `telemetry.TURN_TIMELINE` records milestones for every spoken turn: end of
  the wake word, microphone open, start and end of speech, transcription ready,
  first token from the model, synthesis start, first sound.
- Each turn writes one `Tiempos [type] ...` line to the log and each boot one
  `Arranque ...` line. `scripts/latency_report.py` summarizes them.
- A **fixed script** of 19 phrases covers every kind of turn (chat, long
  answer, time, time in another city, weather, Spotify through the model and by
  local rule, TV state and actions, mode changes, "silencio"), repeated three
  times.

## Where the time went (baseline on the Pi, 18 Nov 2026)

From the moment you stop talking to Vader's first sound, by stage (medians):

| Stage | Median | Notes |
| --- | ---: | --- |
| Thinking | 1.01 s | Almost all of it is the model's first token: 0.98 s with ~3,000 input tokens |
| Silence | 0.80 s | Fixed: the wait that decides you finished |
| First voice chunk | 0.77 s | Piper: ~0.3 s plus ~0.02 s per character of the first sentence |
| Transcription | 0.42 s | Speculation was used in 58 of 58 turns |
| Mic, lights, gaps | ≤ 0.03 s | Already solved |

Earlier, on the laptop (October 2026), a first round had already brought a
conversation turn from 4.2 s to about 2.5 s and a TV order from about 4.4 s
to 0.05 s, with a persistent Piper process, pre-warmed connections,
speculative transcription, a weather cache and a persistent TV connection.

## What was changed

| Change | Effect on the Pi |
| --- | --- |
| Speculative transcription at 192 ms instead of 320 ms (same accuracy on the author's recordings) | Transcription 0.42 → 0.25 s |
| Re-warm the TV connection on every "Oye Vader", in a thread | First TV order 7.29 → 3.10 s |
| Streaming transcription with `grok-stt` and Spanish keywords | Transcription 0.25 → 0.00 s in 19 of 19 turns; about 0.3 s less to first sound |

Together: **about 0.45 s less per voice turn** than the baseline. With
streaming, the text is usually ready before the 800 ms of silence are over.
On the laptop recordings its word error rate was 5.8 %, against 7.1 % for the
REST path.

## What was tried and rejected

- **Cutting the first chunk shorter**, so Piper starts sooner. Tested by ear:
  the cuts took the naturalness out of the intonation.
- **Asking the model for a short first sentence.** It sounded better than
  cutting, but Vader lost his spark.

Both are faster on paper. Neither was kept.

## Boot

| | Plug in → ready |
| --- | ---: |
| First measurement | ~42 s |
| Load while the Wi-Fi connects | **36 s** |

The systemd service used to wait for the network before starting Python. Now
Vader loads Vosk, Silero and Piper while the Wi-Fi connects and only then waits
for the network: Linux to ready went from 31.5 to 25.5 s.

A measurement also caught the eyes lying: the dim pupils lit up 11.5 s before
Vader was really listening, and an "Oye Vader" said then was lost. Now the eyes
stay dark until Vader is ready, and a wake word said the moment they light up
works on the first try.

Where the remaining 36 s go: about 11 s of firmware, then Linux and the
network. A read-only diagnosis (`scripts/diagnostico_red.sh`) found that
NetworkManager spends about 8 s regenerating its configuration four times and
2.6 s waiting for an Ethernet cable that is not there. Options to recover that
are documented but not applied: 36 s was judged good enough.
