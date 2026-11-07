# Hands-Free Viewer

Read a PDF, listen to audio and watch video **without touching anything**.
Look at a glowing arrow to move; clench your jaw to act. Both signals come from an
EEG headset streaming over [LSL](https://labstreaminglayer.org/) (Lab Streaming Layer).

Built by Team 7 for the Neurotech @ NC State BCI Hackathon (Nov 2026), ANT Neuro
track: *movement-based neurotech applications* on ANT Neuro's 24-channel eego system.

## How it works

The web version's signal path (the desktop app adds more inputs, see below):

```
eego headset ──LSL──▶ server.py ──▶ look detector  (O1, Oz, O2, POz)  ──┐
 24 ch, 500 Hz                  └─▶ jaw detector   (T7, T8, M1, M2)  ──┴─▶ web page (localhost:8780)
```

- **Look (SSVEP):** each arrow pulses at its own rate (12–15 times a second). Staring at
  one makes the visual cortex echo that rate. Every 0.25 s the app compares the last
  2 s of the back-of-head channels with sine waves at each arrow's rate (canonical
  correlation analysis, CCA). The same winner 3 times in a row counts as a press.
- **Jaw clench:** clenching fires the chewing muscles next to the temporal electrodes,
  causing a burst of 30–100 Hz activity many times above the resting level. The
  detector reports a quick clench, a double clench, or a clench held for 1 second.
- The arrows are **soft Gaussian glows** with a strength slider, which is easier on the
  eyes than hard flashing boxes. Lower strength is gentler but gives a weaker brain signal.

## Controls

| Tab   | Look at an arrow | Jaw | Keyboard backup |
|-------|------------------|-----|-----------------|
| PDF   | ▲ 12 Hz, ▶ 13 Hz, ▼ 14 Hz, ◀ 15 Hz to scroll. Scrolling past the edge turns the page | **Double clench:** arrows on/off. **Hold ~1 s:** zoom in/out | Arrow keys; `J` tap twice / hold |
| Audio | none | **Double clench:** play/pause. **Single clench:** back 15 s | `J` tap twice / tap once |
| Video | ⏪ 12 Hz back 15 s, ⏩ 15 Hz forward 15 s | **Clench:** play/pause | ← →; `J` tap |

`J` behaves like your jaw: tap = clench, two quick taps = double clench, hold 1 s = held clench.
The keyboard always works, so a live demo still runs if the EEG is noisy.

## Setup

Needs [uv](https://docs.astral.sh/uv/) (it installs Python 3.13 and all libraries for you).
**Install before going to the headset station.** The ANT network may have no internet.

```bash
git clone git@github.com:solacorrado/eeg-interpreter.git
cd eeg-interpreter
uv sync                                   # install libraries
uv run python bci_demo/make_media.py      # make the sample PDF, audio and video
```

## Run

### Desktop app (the full version)

Double-click the desktop icon, or:

```bash
uv run python bci_demo/desktop_app.py            # real headset: connect on the start screen
uv run python bci_demo/desktop_app.py --fake     # start with the practice (fake) headset
```

**Start screen**
1. **Profile:** pick who is wearing the headset, or make a new profile. Each wearer gets
   their own calibrated settings.
2. **Headset:** *Real headset* (press **Search** to list LSL streams, or use "first EEG
   stream"), or *Practice headset* (fake signals). Press **Connect**. The green dot shows
   the stream, channel count and rate.
3. Choose **Calibrate & test** or **Use the viewer**.

**Calibrate & test**
- Tick the inputs to include: flicker (12–15 Hz), jaw clench, eyebrow raise, blink,
  glance left/right, eyes closed, fist squeeze right/left. All of them take about 5.5 minutes.
- **Start calibration:** the wearer follows the cues on screen (a beep marks each new step).
  For every input the app finds the threshold that best separates "doing it" from
  "resting" and grades it **Good** (≥90%), **OK** (≥75%) or **Poor**. Good and OK
  inputs update the profile; Poor ones keep their old settings.
- **Run live test:** random cues; the live detectors (using the profile) must fire the
  right event in time. Reports hit rate, reaction time and false alarms per minute.
- **Live signals:** a bar for every signal with the profile's threshold marked, and a
  light for every detection. Good for checking the headset before you start.
- With the practice headset, "Practice person follows the cues" makes the fake signals
  act out each cue, so the whole flow can be tried without hardware. Practice keys:
  J jaw, E eyebrows, B blink, ← → glance, C eyes closed, R / L fist, 1–4 stare at 12–15 Hz.

**Viewer:** the PDF / audio / video controls below, using the profile's settings.

### What gets saved, and where

Everything goes to **Documents/HandsFreeViewer** (outside the code, so updates and
re-clones never touch it):

```
profiles/<name>.json        settings + latest result per input + full history of every session
profiles/<name>.json.bak    the previous version of that file
trials/<name>/<date>_calibration|test/
    eeg.npz                 raw EEG, timestamps, channel names, sample rate
    features.csv            signal features, 25 rows per second
    cues.csv / events.csv   what the wearer was asked to do / what the app detected
    results.json            the scores
trials/index.csv            one row per input per session: the "what works" log
```

Profiles record the app version that made each result and carry a schema version, so
newer app versions can still read them.

### Web app (simple version: flicker + jaw only)

```bash
uv run python bci_demo/server.py --fake     # then open http://localhost:8780
uv run python bci_demo/server.py
```

Options: `--pdf FILE`, `--ssvep-threshold 0.42`, `--clench-factor 5`, `--port 8780`.

## Status

- Working end to end against the **simulated** headset (calibration, live test, viewer).
- Not yet tested on the real eego. Calibrate each wearer on the real headset; the live
  test then shows which inputs actually work for them.

## Safety

The arrows flicker at 12–15 Hz. Flicker in this range can trigger seizures in people with
photosensitive epilepsy. Don't use it if that applies to you or anyone watching.
