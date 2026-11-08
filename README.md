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

- **Look (SSVEP):** each arrow pulses at its own rate (8–19 times a second). Staring at
  one makes the visual cortex echo that rate. Every 0.25 s the app compares the last
  2 s of the back-of-head channels with sine waves at each arrow's rate (canonical
  correlation analysis, CCA). The same winner 3 times in a row counts as a press.
- **Jaw clench:** clenching fires the chewing muscles next to the temporal electrodes,
  causing a burst of 30–100 Hz activity many times above the resting level. The
  detector reports a quick clench, a double clench, or a clench held for 1 second.
- The arrows are **soft Gaussian glows** with a strength slider, which is easier on the
  eyes than hard flashing boxes. Lower strength is gentler but gives a weaker brain signal.

## Results on a real brain

Two real sessions are in [`recordings/`](recordings/) (raw EEG, cues, detections, scores), with a
side-by-side results table for both wearers. First real session (wearer1, ANT eego 24-channel cap,
500 Hz, 7.6 min):

| Signal | Result | Used in the viewer |
|---|---|---|
| **Flicker (SSVEP)** | 88–100% at all 10 rates tested (8–19 Hz); **8/8** correct arrows on the PDF tab and **4/4** on the video tab in replay, **0 wrong arrows**, ~1–1.5 s to trigger | ✅ arrows |
| **Jaw clench** | 100% in calibration; **6/6** in replay, 0.4–1.4 s after the cue | ✅ actions |
| **Eyes closed** | 100% in calibration; 3/4 in replay, 0.3 false alarms/min | candidate "rest" control |
| Fist squeeze | right 81% in calibration, but the live detector never fired on it | ✗ |
| Blink / glance / eyebrows | not testable in that session (recorded in a very noisy first 90 s, before spoken cues existed) | ✗ untested |

Replaying wearer1's recording over LSL into the real app: staring at 11 Hz scrolled left, 13 Hz scrolled
up, the unused rates (10, 17, 19 Hz) fired nothing, and all 6 clenches toggled video play/pause.

Blink findings from the same data: she blinked naturally 12.6×/min and double-blinked naturally
2–3×/min, so single and double blinks can't be controls; natural *triple* blinks: 0 in 6 min.

## Controls

The flicker rates are the best well-spaced rates from the real session; each profile can change them.

| Tab | Look at an arrow | Jaw | Big buttons | Keyboard backup |
|-----|------------------|-----|-------------|-----------------|
| 📄 **Read** (PDF) | ▲ 13 Hz, ▶ 9 Hz, ▼ 15 Hz, ◀ 11 Hz to scroll; past the edge turns the page | **Double clench:** arrows on/off. **Hold ~1 s:** zoom in/out | Previous page · Zoom · Arrows on/off · Next page | Arrow keys; `J` tap twice / hold |
| 🎵 **Listen** (audio) | none | **Double clench:** play/pause. **Single clench:** back 15 s | Back 15 s · Play/Pause · Forward 15 s | `J` tap twice / tap once |
| 🎬 **Watch** (video) | ⏪ 9 Hz back 15 s, ⏩ 13 Hz forward 15 s | **Clench:** play/pause | Back 15 s · Play/Pause · Forward 15 s | ← →; `J` tap |

- **📂 Open file** (or drag a file onto the window) opens your own PDF, audio (mp3, wav, m4a…) or
  video (mp4, mov…). The last file of each kind is remembered per profile.
- The viewer only switches on the inputs it uses (flicker + jaw), so other detectors can't
  steal events. On the real recording, clenching also tensed the forehead, and without this
  every clench was read as an eyebrow raise.
- `J` behaves like your jaw: tap = clench, two quick taps = double clench, hold 1 s = held clench.
  The keyboard and big buttons always work, so a live demo still runs if the EEG is noisy.
- `1` / `2` / `3` switch between **Read**, **Listen** and **Watch**.

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
- **Randomized:** each calibration shuffles the order of the input blocks and of the trials
  inside each block, and varies every rest by -20% to +30%, so the wearer can't anticipate
  cues and slow drift can't line up with one class. The shuffle seed is saved with the trial.
- **Redo Poor inputs only:** recalibrates just the inputs whose last calibration had a Poor result.
- **Run live test:** random cues; the live detectors (using the profile) must fire the
  right event in time. Reports hit rate, reaction time and false alarms per minute.
- **Live signals:** a bar for every signal with the profile's threshold marked, and a
  light for every detection. Good for checking the headset before you start.
- With the practice headset, "Practice person follows the cues" makes the fake signals
  act out each cue, so the whole flow can be tried without hardware. Practice keys:
  J jaw, E eyebrows, B blink, ← → glance, C eyes closed, R / L fist, 1–4 stare at 12–15 Hz.

**Viewer:** the PDF / audio / video controls below, using the profile's settings.

**Plugging in the real headset.** The app reads whatever ANT's eego software streams over
LSL on their network. It copes with streams whose type isn't exactly "EEG", any sample
rate, volts or microvolts (every signal is measured against the wearer's own resting
level), labels like `Fp1-CPz` or `EEG O1`, and extra trigger/counter channels. If a
channel it needs can't be found, the start screen says which one. Needs: LSL streaming
turned on in eego, your laptop on ANT's network, and Python allowed through the Windows firewall.

**If something goes wrong mid-session:** the status line turns red ("NO DATA for N s") if
the headset stops sending (LSL reconnects by itself). **Stop** offers to save what was
recorded as a *partial* trial (scored, but it doesn't change the profile's settings), and
closing the window mid-session saves it automatically.

### Before you go to the station: self-test

```bash
uv run python bci_demo/selftest.py
```

About 2 minutes, no headset. It checks the libraries, the sample files and the data folder,
then runs a short calibration and live test on the practice headset. Ends with PASS or FAIL.

### After a session: re-analyse the recordings

Every trial keeps the raw EEG, so detection can be improved without the headset. Change the
code, then re-score real recordings with it:

```bash
uv run python bci_demo/reanalyze.py                       # list saved trials
uv run python bci_demo/reanalyze.py latest                # re-score the newest one
uv run python bci_demo/reanalyze.py all --profile wearer1   # every trial for one wearer
uv run python bci_demo/reanalyze.py latest --csv          # also export eeg.csv (time + channels)
```

It prints the old score next to the new one for every input. The live app and this tool
share the same processing code and cut the data into identical 40 ms steps, so a re-run of
unchanged code reproduces the original scores exactly.

### What gets saved, and where

Everything goes to **Documents/HandsFreeViewer** (outside the code, so updates and
re-clones never touch it):

```
profiles/<name>.json        settings + latest result per input + full history of every session
profiles/<name>.json.bak    the previous version of that file
trials/<name>/<date>_calibration|test[_partial]/
trials/index.csv            one row per input per session: the "what works" log
```

| File in each trial folder | Format | Contents | Open with |
|---|---|---|---|
| `eeg.npz` | NumPy compressed archive | raw EEG (samples × channels), timestamps, channel names, sample rate | Python `numpy.load`, `reanalyze.py` |
| `eeg.csv` (optional) | CSV | time + one column per channel (`reanalyze.py latest --csv`) | Excel, MATLAB, anything |
| `cues.csv` | CSV | what the wearer was asked to do, and when | Excel |
| `events.csv` | CSV | what the app detected, and when | Excel |
| `features.csv` | CSV | processed signals, 25 rows per second | Excel |
| `results.json` | JSON | scores, verdicts, settings used, app version, shuffle seed | text editor |

ANT's eego software saves its own copy of a session as `.cnt` / `.edf` (standard EEG formats).

Profiles record the app version that made each result and carry a schema version, so
newer app versions can still read them.

### Web app (simple version: flicker + jaw only)

```bash
uv run python bci_demo/server.py --fake     # then open http://localhost:8780
uv run python bci_demo/server.py
```

Options: `--pdf FILE`, `--ssvep-threshold 0.42`, `--clench-factor 5`, `--port 8780`.

## Files

```
bci_demo/
  desktop_app.py  the desktop app: start screen, calibrate & test, viewer (PySide6 / Qt)
  live_engine.py  reads the LSL stream; the shared Pipeline (features -> events) used live and offline
  features.py     raw EEG -> jaw / brow / blink / glance / alpha / hand-rhythm features, SSVEP scores
  gestures.py     features -> events, using a profile's thresholds
  calibration.py  calibration and live-test steps (randomized) and scoring
  storage.py      profiles and trial recordings on disk
  reanalyze.py    re-score saved trials with the current code; export EEG to CSV
  selftest.py     2-minute readiness check on the practice headset
  version.py      app version, stored with every result
  fake_eeg.py     practice headset: a fake 24-channel eego stream that can do every input
  make_media.py   generates the sample PDF, audio and video
  icon.ico        app icon
  server.py, engine.py, detect.py, static/index.html
                  the simpler web version (flicker + jaw)
```

## Status

- **Real headset:** connects to ANT's eego stream (and reconnects by itself when eego restarts
  its stream, which plain LSL can't do for this stream). Flicker and jaw proven on one wearer's
  recording; the viewer driven by that recording works end to end (see *Results on a real brain*).
- **Not yet done:** the viewer used live by a wearer; more than one person; retesting blink,
  glance and eyebrows with spoken cues and a settled headset.
- Everything also runs on the practice headset (calibration, live test, viewer).

## Safety

The arrows flicker at 8–19 Hz. Flicker in this range can trigger seizures in people with
photosensitive epilepsy. Don't use it if that applies to you or anyone watching.
