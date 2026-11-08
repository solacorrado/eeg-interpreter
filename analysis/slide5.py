"""Re-score the recorded sessions for the presentation's results slide (slide 5).

Reads only each session's recorded app signals (features.csv, 25 rows per second) and
cues (cues.csv). Nothing is tuned on the data being scored. See analysis/README.md.

  uv run python analysis/slide5.py                 # uses ./recordings
  uv run python analysis/slide5.py path/to/recordings --json out.json
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

VIEWER_HZ = [9.0, 11.0, 13.0, 15.0]   # the viewer's four arrows
SS_THR, SS_MARGIN = 0.42, 0.08         # app defaults (bci_demo/storage.py DEFAULT_SETTINGS)
SS_STEP_S, SS_BLOCK_S = 0.25, 1.0      # app checks every 0.25 s; 1 s pause after a command
WINDOW_FULL_S = 2.0                    # flicker scores use a 2 s window; skip the first 2 s of a look
JAW_PRE_S, JAW_POST_S = 2.0, 2.5       # a clench counts for a cue if it starts 2 s before to 2.5 s after
REST_SKIP_S = 1.0                      # ignore the first 1 s of each rest period


# ---------------------------------------------------------------- loading and helpers
def load_session(folder):
    rows = list(csv.DictReader(open(folder / "features.csv")))
    sig = {k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows]) for k in rows[0]}
    cues = [dict(r, t_start=float(r["t_start"]), t_end=float(r["t_end"]))
            for r in csv.DictReader(open(folder / "cues.csv"))]
    return sig, cues


def cues_of(cues, inp, phase):
    return [c for c in cues if c["input"] == inp and c["phase"] == phase]


def rest_periods(cues):
    return [(c["t_start"] + REST_SKIP_S, c["t_end"]) for c in cues if c["phase"] == "rest"]


def in_rest(t, rests):
    return any(a <= t <= b for a, b in rests)


def rest_minutes(rests):
    return sum(max(0.0, b - a) for a, b in rests) / 60


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------- gaze (flicker)
def gaze_scores(sig):
    return np.vstack([sig[f"ss_{f:g}"] for f in VIEWER_HZ])


def gaze_trials(sig, cues):
    """4-choice per look: the arrow with the highest mean score over the look wins."""
    t, scores = sig["t"], gaze_scores(sig)
    looks = [c for c in cues_of(cues, "ssvep", "action") if float(c["cls"]) in VIEWER_HZ]
    correct = 0
    for c in looks:
        m = (t >= c["t_start"] + WINDOW_FULL_S) & (t <= c["t_end"] + 0.2)
        winner = VIEWER_HZ[int(np.nanargmax(np.nanmean(scores[:, m], axis=1)))]
        correct += winner == float(c["cls"])
    return correct, len(looks)


def gaze_live_commands(sig):
    """The app's live rule (bci_demo/gestures.py update_ssvep) at default settings."""
    t, scores = sig["t"], gaze_scores(sig)
    cmds, hist, blocked_until, next_check = [], [], -1e9, t[0]
    for i in range(len(t)):
        if t[i] < next_check:
            continue
        next_check = t[i] + SS_STEP_S
        if t[i] < blocked_until:
            hist = []
            continue
        s = scores[:, i]
        if np.any(np.isnan(s)):
            continue
        order = np.argsort(s)[::-1]
        best, lead = order[0], s[order[0]] - s[order[1]]
        hist = (hist + [best if s[best] >= SS_THR and lead > SS_MARGIN else None])[-3:]
        if len(hist) == 3 and hist[0] is not None and len(set(hist)) == 1:
            cmds.append((t[i], VIEWER_HZ[best]))
            hist, blocked_until = [], t[i] + SS_BLOCK_S
    return cmds


def gaze_session(sig, cues):
    correct, n = gaze_trials(sig, cues)
    cmds = gaze_live_commands(sig)
    rests = rest_periods(cues)
    times = []
    for c in [c for c in cues_of(cues, "ssvep", "action") if float(c["cls"]) in VIEWER_HZ]:
        right = [tt for tt, f in cmds if c["t_start"] <= tt <= c["t_end"] + 0.5 and f == float(c["cls"])]
        if right:
            times.append(right[0] - c["t_start"])
    false = [f for tt, f in cmds if in_rest(tt, rests)]
    return {"correct": int(correct), "looks": n, "time_to_command_s": times,
            "false_at_rest": len(false), "false_at_rest_by_hz": {f"{f:g}": false.count(f) for f in set(false)},
            "rest_min": rest_minutes(rests)}


# ---------------------------------------------------------------- jaw
def jaw_peaks(sig, cues, trials=None):
    """Peak jaw level per clench cue, and per 1.5 s chunk of jaw rest periods."""
    t, j = sig["t"], sig["jaw"]
    acts, rests = cues_of(cues, "jaw", "action"), cues_of(cues, "jaw", "rest")
    if trials is not None:
        acts, rests = [acts[i] for i in trials], [rests[i] for i in trials]
    clench = [np.nanmax(j[(t >= c["t_start"] - JAW_PRE_S) & (t <= c["t_end"] + 0.6)]) for c in acts]
    rest = []
    for c in rests:
        x = c["t_start"] + 0.8
        while x + 1.5 <= c["t_end"]:
            seg = j[(t >= x) & (t <= x + 1.5)]
            if len(seg):
                rest.append(np.nanmax(seg))
            x += 1.5
    return clench, rest


def jaw_threshold(clench, rest):
    """The app's own formula (bci_demo/calibration.py _separate), clamped like the app."""
    thr = math.sqrt(np.percentile(clench, 25) * np.percentile(rest, 90))
    return float(min(max(thr, 2.5), 200.0))


def jaw_live_events(sig, factor):
    """The app's live rule (bci_demo/gestures.py _emg): 2 rows over -> start, 2 rows under 0.6x -> end."""
    t, v = sig["t"], sig["jaw"] / factor
    starts, active, above, below, last_end = [], False, 0, 0, -1e9
    for i in range(len(t)):
        if np.isnan(v[i]):
            continue
        if not active:
            above = above + 1 if v[i] >= 1 else 0
            if above >= 2 and t[i] - last_end > 0.2:
                active, below = True, 0
                starts.append(t[i])
        else:
            below = below + 1 if v[i] < 0.6 else 0
            if below >= 2:
                active, above, last_end = False, 0, t[i]
    return starts


def jaw_score(sig, cues, factor, trials=None):
    events = jaw_live_events(sig, factor)
    all_acts = cues_of(cues, "jaw", "action")
    acts = all_acts if trials is None else [all_acts[i] for i in trials]
    used, times = set(), []
    for c in acts:
        near = [e for e in events if c["t_start"] - JAW_PRE_S <= e <= c["t_start"] + JAW_POST_S and e not in used]
        if near:
            e = min(near, key=lambda e: abs(e - c["t_start"]))
            used.add(e)
            times.append(e - c["t_start"])
    rests = rest_periods(cues)
    near_cue = lambda e: any(c["t_start"] - JAW_PRE_S <= e <= c["t_start"] + JAW_POST_S for c in all_acts)
    false = [e for e in events if e not in used and not near_cue(e) and in_rest(e, rests)]
    return {"detected": len(times), "clenches": len(acts), "time_s": times,
            "false_at_rest": len(false), "rest_min": rest_minutes(rests), "threshold_x": round(factor, 2)}


def jaw_wearer(sessions):
    """Threshold always learned on data other than the clenches being scored."""
    out = []
    if len(sessions) > 1:                      # leave one session out
        for name, (sig, cues) in sessions.items():
            clench, rest = [], []
            for other, (s2, c2) in sessions.items():
                if other != name:
                    a, r = jaw_peaks(s2, c2)
                    clench, rest = clench + a, rest + r
            out.append({"session": name, "scheme": "threshold from the other sessions",
                        **jaw_score(sig, cues, jaw_threshold(clench, rest))})
    else:                                      # one session: split in half, train on one half, test the other
        name, (sig, cues) = next(iter(sessions.items()))
        n = len(cues_of(cues, "jaw", "action"))
        odd, even = list(range(0, n, 2)), list(range(1, n, 2))
        for train, test in ((odd, even), (even, odd)):
            r = jaw_score(sig, cues, jaw_threshold(*jaw_peaks(sig, cues, train)), test)
            r["false_at_rest"], r["rest_min"] = 0, 0.0      # counted once below, not per half
            out.append({"session": name, "scheme": f"threshold from trials {train}, tested on {test}", **r})
        full = jaw_score(sig, cues, jaw_threshold(*jaw_peaks(sig, cues)))
        out.append({"session": name, "scheme": "false alarms only, threshold from all of this session",
                    "detected": 0, "clenches": 0, "time_s": [], "false_at_rest": full["false_at_rest"],
                    "rest_min": full["rest_min"], "threshold_x": full["threshold_x"]})
    return out


# ---------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="?", default="recordings")
    ap.add_argument("--json", help="also write all per-session numbers to this file")
    args = ap.parse_args()

    root = Path(args.recordings)
    wearers = {}
    for folder in sorted(root.glob("wearer*/2026*")):
        if (folder / "features.csv").exists():
            wearers.setdefault(folder.parent.name, {})[folder.name] = load_session(folder)
    if not wearers:
        raise SystemExit(f"No sessions with features.csv under {root}")

    report = {"gaze": {}, "jaw": {}}
    for who, sessions in wearers.items():
        report["gaze"][who] = {name: gaze_session(*s) for name, s in sessions.items()}
        report["jaw"][who] = jaw_wearer(sessions)

    g = [r for w in report["gaze"].values() for r in w.values()]
    j = [r for w in report["jaw"].values() for r in w]
    g_ok, g_n = sum(r["correct"] for r in g), sum(r["looks"] for r in g)
    j_ok, j_n = sum(r["detected"] for r in j), sum(r["clenches"] for r in j)
    g_times = [x for r in g for x in r["time_to_command_s"]]
    j_times = [x for r in j for x in r["time_s"]]
    w1 = list(report["gaze"].get("wearer1", {}).values())
    w1_false = sum(r["false_at_rest"] for r in w1) / sum(r["rest_min"] for r in w1) if w1 else float("nan")
    j_false = sum(r["false_at_rest"] for r in j) / sum(r["rest_min"] for r in j)

    print(f"{len(wearers)} wearers, {sum(len(s) for s in wearers.values())} sessions\n")
    print(f"{'Measure':<28}{'Gaze: which of 4 arrows':<28}Jaw clench")
    print(f"{'Correct':<28}{f'{g_ok} of {g_n} ({g_ok / g_n:.0%})':<28}{j_ok} of {j_n} ({j_ok / j_n:.0%})")
    (glo, ghi), (jlo, jhi) = wilson(g_ok, g_n), wilson(j_ok, j_n)
    print(f"{'95% confidence range':<28}{f'{glo:.0%}-{ghi:.0%}':<28}{jlo:.0%}-{jhi:.0%}")
    print(f"{'Median time to a command':<28}{f'{np.median(g_times):.1f} s':<28}{np.median(j_times):.1f} s")
    print(f"{'False triggers at rest':<28}{f'{w1_false:.1f} per min (wearer 1)':<28}{j_false:.1f} per min")

    print("\nPer session")
    for who, sessions in report["gaze"].items():
        for name, r in sessions.items():
            print(f"  gaze {who}/{name}: {r['correct']}/{r['looks']} looks, false at rest {r['false_at_rest']} "
                  f"in {r['rest_min']:.1f} min {r['false_at_rest_by_hz']}")
    for who, rows in report["jaw"].items():
        for r in rows:
            print(f"  jaw  {who}/{r['session']} ({r['scheme']}, {r['threshold_x']}x): "
                  f"{r['detected']}/{r['clenches']}, false at rest {r['false_at_rest']} in {r['rest_min']:.1f} min")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
