# How the results slide (slide 5) was scored

The numbers on the presentation's results slide come from `analysis/slide5.py`, run on the
four sessions in [`recordings/`](../recordings/). The script reads only the signals the app
recorded during each session (`features.csv`) and the cues the wearer followed (`cues.csv`).

```bash
uv run python analysis/slide5.py                    # prints the slide table and every session
uv run python analysis/slide5.py --json out.json    # also saves all per-session numbers
```

## The slide's numbers

2 wearers, 4 sessions on the ANT eego (500 Hz), 7–8 Nov 2026.

| Measure | Gaze: which of 4 arrows | Jaw clench |
|---|---|---|
| Correct | 30 of 32 looks (94%) | 23 of 24 clenches (96%) |
| 95% confidence range | 80–98% | 80–99% |
| Median time to a command | 2.3 s | 0.4 s |
| False triggers at rest | 1.7 per min (wearer 1) | 0.3 per min |

## Why not the app's calibration scores

The calibration screen (`bci_demo/calibration.py`) is built to choose settings, not to measure
accuracy, and its percentages run high (often 100%) for two reasons:

| Reason | Effect |
|---|---|
| It picks the threshold that best separates action from rest on the same data, then scores that data | The score is in-sample, so it is optimistic |
| For flicker it counts every 40 ms row of a look; each session has only 2 looks per rate | Thousands of overlapping rows from a few looks make the percentage look more certain than it is |

The script below counts each look or clench once and never tunes on the data it scores.

## Method

**Data used**

| Session | Looks at the viewer's rates (9, 11, 13, 15 Hz) | Jaw clenches | Rest time scored |
|---|---|---|---|
| wearer1 `20261107-164627_calibration` | 8 | 6 | 2.5 min |
| wearer1 `20261107-202648_calibration_quick` | 8 | 6 | 0.8 min |
| wearer1 `20261108-103543_calibration_quick` | 8 | 6 | 0.8 min |
| wearer2 `20261107-171911_calibration` | 8 | 6 | 2.4 min |
| **Total** | **32** | **24** | **6.5 min** |

Each look lasts 6 s; each clench cue lasts 2 s. Rest time is every cue marked `rest` (any input),
minus its first second.

**Gaze**

| Measure | How it is computed |
|---|---|
| Correct | For each look, the arrow (of the four viewer rates) with the highest average flicker score from 2 s into the look (when the 2 s analysis window is full) to the end. No threshold, so nothing to tune. Chance is 25%. |
| Time to a command | The app's own live rule (`bci_demo/gestures.py`, `update_ssvep`) run on the recorded scores at the app's **default** settings: score ≥ 0.42, ahead of the next arrow by 0.08, the same winner on 3 checks 0.25 s apart, then a 1 s pause. Time is from the start of the look to the first correct command, so it includes the 2 s the analysis window needs to fill. |
| False triggers at rest | Commands from the same rule during rest periods. Shown for wearer 1 only: 47 of wearer 2's 48 came from the 9 Hz arrow, which sits near the natural ~10 Hz alpha rhythm (see below). |

**Jaw**

| Measure | How it is computed |
|---|---|
| Threshold | The app's own formula (`calibration._separate`): geometric mean of the 25th percentile of clench peaks and the 90th percentile of rest peaks, clamped to 2.5–200× resting. Always learned on data other than the clenches being scored: for wearer 1, from the other two sessions (leave one session out); for wearer 2 (one session), from half of the trials, tested on the other half, then swapped. |
| Correct | The app's own live rule (`gestures.py`, `_emg`) run on the recorded jaw signal. A clench counts for a cue if it starts between 2 s before and 2.5 s after the cue: wearers sometimes clenched as the cue appeared, before the action period began. |
| Time to a command | From the cue to the start of the detected clench. Early clenches give negative times. |
| False triggers at rest | Detected clenches during rest periods that do not belong to any cue. For wearer 2 this uses the threshold from all six of its trials, since one session cannot be split for this count without halving the rest time. |

**95% confidence range** is the Wilson score interval for the count (for example 30 of 32).

## Checks

The counts barely move when the windows change:

| Change | Gaze correct | Jaw correct |
|---|---|---|
| As used | 30/32 | 23/24 |
| Gaze window starts 1.5 s or 2.5 s into the look | 30/32 | |
| Gaze window starts 3.0 s into the look | 31/32 | |
| Jaw match window starts 1.0 s or 2.5 s before the cue | | 23/24 |

## Limits

| Limit | Detail |
|---|---|
| Small samples | 32 looks and 24 clenches from 2 people, so the confidence ranges are wide |
| Offline | Scored from recorded app signals, not from a wearer using the viewer live |
| Short rest time | 6.5 min in total, so false-trigger rates are rough |
| 9 Hz | Fires during rest for wearer 2 (47 false commands in 2.4 min); a candidate to replace |
| Settings | Gaze uses the app defaults; a profile's calibrated thresholds could do better or worse |
