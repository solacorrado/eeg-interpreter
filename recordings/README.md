# Recordings

Real EEG sessions recorded with the Hands-Free Viewer app at the Neurotech @ NC State BCI
Hackathon (7 Nov 2026), on ANT Neuro's eego amplifier and 24-electrode cap, streamed over LSL.
Both wearers agreed to share their data; names are replaced with `wearer1` / `wearer2`.

| Folder | Length | What it is |
|---|---|---|
| `wearer1/20261107-164627_calibration/` | 7.5 min | full randomized calibration of every input (no spoken cues yet; the first ~90 s are very noisy) |
| `wearer2/20261107-171911_calibration/` | 12.3 min | full randomized calibration of every input, with spoken cues |

Both: **500 Hz, 25 channels, no gaps.** Channel order:

```
Fp1 Fp2 F9 F7 F3 Fz F4 F8 F10 T9 T7 C3 C4 T8 T10 P9 P7 P3 Pz P4 P8 P10 O1 O2 Trigger
```

## Files in each session folder

| File | Contents |
|---|---|
| `eeg.npz` | raw EEG: `data` (samples × channels, float32), `timestamps` (LSL clock, seconds), `channels`, `rate` |
| `cues.csv` | every step: `t_start`, `t_end` (same clock as `timestamps`), `input`, `phase` (settle / rest / action), `cls` (e.g. flicker rate, left/right), `text` shown/spoken |
| `events.csv` | what the live app detected, with times |
| `features.csv` | the app's processed signals, 25 rows per second |
| `results.json` | the scores, the settings in use, app version and the shuffle seed |

## Use them

Load in Python:

```python
import numpy as np
z = np.load("recordings/wearer1/20261107-164627_calibration/eeg.npz")
data, ts, channels, rate = z["data"], z["timestamps"], list(z["channels"]), float(z["rate"])
```

Re-score with the current code (from the repo root):

```bash
uv run python bci_demo/reanalyze.py recordings/wearer1/20261107-164627_calibration
uv run python bci_demo/reanalyze.py recordings/wearer2/20261107-171911_calibration --csv   # also writes eeg.csv
```

## Results (calibration scores, current code)

Balanced accuracy separating "doing it" from "resting": Good ≥ 90%, OK ≥ 75%, Poor below.

| Input | wearer1 | wearer2 |
|---|---|---|
| Jaw clench | 100% | 92% |
| Eyes closed | 100% | 100% |
| Blink | 56%¹ | **100%** |
| Glance left / right | 31% / 25%¹ | **94% / 81%** |
| Eyebrow raise | 8%¹ | 25% |
| Fist right / left | 81% / 44% | 56% / 56% |
| Flicker 8 Hz | 94% | 94% |
| Flicker 9 Hz | 98% | 66%² |
| Flicker 10 Hz | 100% | 77%² |
| Flicker 11 Hz | 92% | 100% |
| Flicker 12 Hz | 88% | 99% |
| Flicker 13 Hz | 100% | 96% |
| Flicker 14 Hz | 96% | 99% |
| Flicker 15 Hz | 95% | 94% |
| Flicker 17 Hz | 98% | 87% |
| Flicker 19 Hz | 93% | 70% |

¹ wearer1's blink, glance and eyebrow blocks fell in the very noisy first ~90 s, before spoken
cues existed, and the wearer couldn't see the cues, so those scores say little about the inputs.
² high even when not staring: the person's natural ~10 Hz alpha rhythm, so these rates may fire falsely.

**Takeaways:** flicker at 11–15 Hz, jaw clench and eyes closed worked for both people. Blink and
glance worked for wearer2 once the headset had settled and cues were spoken. Rates near 10 Hz
(9–10) can be unreliable for people with strong alpha. Fist squeezes were not reliable.
