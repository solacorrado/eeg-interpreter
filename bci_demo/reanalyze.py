"""Re-run saved trials through the CURRENT code, with no headset needed.

Use it after a rotation: change the feature or detector code, then check it against
the real recordings instead of waiting for the next slot.

  uv run python bci_demo/reanalyze.py                       # list saved trials
  uv run python bci_demo/reanalyze.py latest                # newest trial
  uv run python bci_demo/reanalyze.py TRIAL_FOLDER          # a specific trial
  uv run python bci_demo/reanalyze.py all --profile wearer1   # every trial for one wearer
Options:
  --profile NAME   use this profile's CURRENT settings for the live-test replay
  --csv            also write eeg.csv (time + every channel) into the trial folder,
                   for Excel / MATLAB / anything else

Calibration trials are re-scored from the raw EEG (new thresholds and grades).
Live-test trials are replayed through the detectors with the profile's settings, so you
see whether today's settings would have caught each cue. Old vs new scores are printed.
"""
import argparse
import ast
import csv
import json
import sys
from pathlib import Path

import numpy as np

import calibration as C
from live_engine import Pipeline
from storage import DEFAULT_SETTINGS, Profile, data_dir


def load_trial(folder):
    z = np.load(folder / "eeg.npz")
    data, ts = z["data"].astype(float), z["timestamps"]
    names, rate = [str(c) for c in z["channels"]], float(z["rate"])
    cues = []
    with open(folder / "cues.csv", encoding="utf8") as f:
        for r in csv.DictReader(f):
            if not r.get("t_start"):
                continue
            r["t_start"], r["t_end"] = float(r["t_start"]), float(r["t_end"])
            r["expect"] = ast.literal_eval(r["expect"]) if r.get("expect") else []
            cues.append(r)
    saved = json.loads((folder / "results.json").read_text(encoding="utf8"))
    return data, ts, names, rate, cues, saved


def replay(data, ts, names, rate, settings, freqs=C.SSVEP_FREQS, chunk_s=0.05):
    pipe = Pipeline(rate, names, settings)
    rows, events = [], []
    step = max(1, int(chunk_s * rate))
    for i in range(0, len(data), step):
        r, e = pipe.process(data[i:i + step], ts[i:i + step], freqs)
        rows += r
        events += e
    return rows, events


def export_csv(folder, data, ts, names):
    out = folder / "eeg.csv"
    with open(out, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["time_s"] + names)
        t0 = ts[0] if len(ts) else 0.0
        for t, row in zip(ts, data):
            w.writerow([f"{t - t0:.4f}"] + [f"{v:.6g}" for v in row])
    return out


def analyse(folder, profile_name=None, want_csv=False):
    data, ts, names, rate, cues, saved = load_trial(folder)
    meta = saved.get("meta", {})
    kind = meta.get("kind", folder.name.split("_", 1)[-1])
    print(f"\n=== {folder.name}  ({kind}, {meta.get('profile')}, app {meta.get('app_version')}, "
          f"{len(ts)} samples @ {rate:g} Hz, {len(names)} ch)")
    if len(ts) == 0:
        print("   no EEG in this trial")
        return
    pname = profile_name or meta.get("profile")
    settings = Profile.load(pname).settings if pname else dict(DEFAULT_SETTINGS)
    rows, events = replay(data, ts, names, rate, settings)
    if kind.startswith("calibration"):
        results, update = C.analyze_calibration(rows, cues)
    else:
        results, summary = C.score_test(events, cues)
        print(f"   false alarms while resting: {summary['false_alarms']} ({summary['false_alarms_per_min']}/min)")
    old = {r["key"]: r for r in saved.get("results", [])}
    print(f"   {'input':18s} {'then':>6s} {'now':>6s}  verdict  detail")
    for r in results:
        o = old.get(r["key"], {})
        then = f"{o['accuracy']:.0%}" if "accuracy" in o else "–"
        print(f"   {r['item']:18s} {then:>6s} {r['accuracy']:6.0%}  {r['verdict']:7s}  {r['detail']}")
    if want_csv:
        print("   wrote", export_csv(folder, data, ts, names))


def all_trials():
    root = data_dir() / "trials"
    return sorted((p for p in root.glob("*/*") if (p / "eeg.npz").exists()), key=lambda p: p.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trial", nargs="?", help="trial folder, 'latest' or 'all'")
    ap.add_argument("--profile", help="profile whose settings to use (and, with 'all', whose trials)")
    ap.add_argument("--csv", action="store_true", help="also export eeg.csv")
    a = ap.parse_args()
    trials = all_trials()
    if not a.trial:
        print(f"Saved trials in {data_dir() / 'trials'}:")
        for t in trials:
            print("  ", t.parent.name, "/", t.name)
        print("\nRun with a folder, 'latest' or 'all'.")
        return
    if a.trial == "latest":
        if not trials:
            sys.exit("No saved trials yet.")
        targets = [max(trials, key=lambda p: p.name)]
    elif a.trial == "all":
        targets = [t for t in trials if not a.profile or t.parent.name.lower() == a.profile.replace(" ", "_").lower()]
    else:
        targets = [Path(a.trial)]
    for t in targets:
        analyse(t, a.profile, a.csv)


if __name__ == "__main__":
    main()
