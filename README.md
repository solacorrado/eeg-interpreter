# Hands-Free Viewer

Read a PDF, listen to audio and watch video **without touching anything**.
Look at a glowing arrow to move; clench your jaw to act. Both signals come from an
EEG headset streaming over [LSL](https://labstreaminglayer.org/) (Lab Streaming Layer).

Built by Team 7 for the Neurotech @ NC State BCI Hackathon (Nov 2026), ANT Neuro
track: *movement-based neurotech applications* on ANT Neuro's 24-channel eego system.

## How it works

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

```bash
# No headset: a fake one, driven by the keyboard
uv run python bci_demo/server.py --fake

# Real eego: connect to ANT's network first
uv run python bci_demo/server.py                    # first LSL stream of type EEG
uv run python bci_demo/server.py --stream "NAME"    # or pick a stream by name
```

Open **http://localhost:8780**, read the flashing warning, and press **Start**. Sit still
and relax your jaw for the first 3 seconds while it calibrates.

In `--fake` mode, hold **Shift** with an arrow key or `J` to send *fake brain signals*
through the real detectors (an arrow then takes about a second, like a real look).

Options:

| Flag | Default | What it does |
|------|---------|--------------|
| `--pdf FILE` | `bci_demo/media/sample.pdf` | the PDF to show |
| `--ssvep-threshold` | `0.42` | how strong a look must be. Raise it if arrows fire on their own; lower it if they never fire |
| `--clench-factor` | `5` | how many times above resting the jaw signal must jump. Raise it if clenches fire too easily |
| `--port` | `8780` | web page port |

The top bar shows live numbers to tune with: each arrow's match score (a look
needs about 0.42+) and the jaw level (× resting; turns yellow during a clench).

## Files

```
bci_demo/
  server.py       LSL reader + web server; sends events to the page
  detect.py       SSVEP (CCA) and jaw-clench detectors
  fake_eeg.py     simulated 24-channel eego stream for testing without hardware
  make_media.py   generates the sample PDF, audio and video
  static/index.html  the app (PDF / Audio / Video tabs, flicker, keyboard)
```

## Status

- Working end to end against the **simulated** headset.
- Not yet tested on the real eego. Expect to tune `--ssvep-threshold`, `--clench-factor`
  and the flicker strength on real data.

## Safety

The arrows flicker at 12–15 Hz. Flicker in this range can trigger seizures in people with
photosensitive epilepsy. Don't use it if that applies to you or anyone watching.
