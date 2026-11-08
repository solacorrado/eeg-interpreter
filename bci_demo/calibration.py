"""Calibration and live-test sessions: the steps, and how they are scored.

CALIBRATION alternates "rest" and "action" steps for each chosen input, records the
feature values, then for each input finds the threshold that best separates action
from rest. Its score is the balanced accuracy at that threshold (50% = no better than
chance, 100% = perfect): Good >= 90%, OK >= 75%, otherwise Poor. Good and OK inputs
update the person's profile; Poor ones are reported but leave the settings alone.

LIVE TEST gives cues in random order and checks whether the live detectors (already
using the profile) fire the right event in time: hit rate, reaction time, false alarms.
"""
import random
from dataclasses import dataclass, field

import numpy as np

# Flicker sweep: no rate is a multiple of another (e.g. 7.5 would echo at 15), so each is judged fairly.
SSVEP_FREQS = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 17.0, 19.0]

INPUTS = {  # key: (label, approx. calibration seconds)
    "ssvep": ("Flicker sweep 8–19 Hz (10 rates)", 185),
    "jaw": ("Jaw clench", 30),
    "brow": ("Eyebrow raise", 30),
    "blink": ("Blink", 36),
    "glance": ("Glance left / right", 40),
    "eyes_closed": ("Eyes closed", 40),
    "fist": ("Fist squeeze, right / left", 64),
}


@dataclass
class Step:
    input: str
    phase: str            # "settle", "rest", "action"
    cls: str              # e.g. "12", "left", "right", "once", "double"
    text: str
    dur: float
    fake: list = field(default_factory=list)   # what the fake person does: (action, seconds, delay)
    flicker: float = None                      # show a flicker target at this rate
    arrow: str = None                          # show an arrow at the "left"/"right" edge
    expect: tuple = ()                         # live test: event types that count as a hit


def _rest(inp, text, dur):
    return Step(inp, "rest", "rest", text, dur)


def _trials(inp, ssvep_freqs=None):
    """(rest, action) pairs for one input, in a fixed order (build_calibration shuffles them)."""
    pairs = []
    if inp == "ssvep":
        for f in list(ssvep_freqs or SSVEP_FREQS) * 2:
            pairs.append((_rest(inp, "Relax — look at the dot", 3),
                          Step(inp, "action", f"{f:g}", "Stare at the glowing circle", 6,
                               [("look", 5.6, 0.3, f)], flicker=f)))
    elif inp == "jaw":
        pairs = [(_rest(inp, "Relax your jaw", 3),
                  Step(inp, "action", "once", "Clench your jaw once, quickly", 2, [("clench", 0.5, 0.4)]))
                 for _ in range(6)]
    elif inp == "brow":
        pairs = [(_rest(inp, "Relax your face", 3),
                  Step(inp, "action", "once", "Raise your eyebrows once", 2, [("brow", 0.6, 0.4)]))
                 for _ in range(6)]
    elif inp == "blink":
        pairs = [(_rest(inp, "Look at the dot (blink normally)", 3),
                  Step(inp, "action", "once", "Blink once, firmly", 1.5, [("blink", 0.25, 0.4)]))
                 for _ in range(8)]
    elif inp == "glance":
        pairs = [(_rest(inp, "Look at the dot", 3),
                  Step(inp, "action", side, f"Look at the {side.upper()} arrow, then back to the dot", 2,
                       [(f"glance_{side}", 0.8, 0.35)], arrow=side))
                 for side in ["left", "right"] * 4]
    elif inp == "eyes_closed":
        pairs = [(_rest(inp, "Eyes open — look at the dot", 5),
                  Step(inp, "action", "closed", "Close your eyes. Open them at the beep.", 5,
                       [("eyes_closed", 4.3, 0.5)]))
                 for _ in range(4)]
    elif inp == "fist":
        pairs = [(_rest(inp, "Relax both hands", 4),
                  Step(inp, "action", side, f"Squeeze your {side.upper()} fist (keep squeezing)", 4,
                       [(f"fist_{side}", 3.4, 0.4)]))
                 for side in ["right", "left"] * 4]
    return pairs


def build_calibration(inputs, seed=None, randomize=True, ssvep_freqs=None, keep_order=False):
    """Calibration steps. With randomize (the default), the order of the input blocks and of
    the trials inside each block is shuffled, and every rest gets a random length (-20% to
    +30%), so the wearer can't anticipate cues and slow drift can't line up with one class.
    Each input stays in its own block so the instructions don't change every few seconds.
    The same seed always gives the same order (it is saved with the trial).
    ssvep_freqs: only these flicker rates (e.g. the viewer's). keep_order: run the input blocks in
    the order given (the trials inside each block are still shuffled)."""
    rng = random.Random(seed)
    blocks = []
    for inp in inputs:
        pairs = _trials(inp, ssvep_freqs)
        if randomize:
            rng.shuffle(pairs)
            for rest, _ in pairs:
                rest.dur = round(rest.dur * rng.uniform(0.8, 1.3), 1)
        blocks.append(pairs)
    if randomize and not keep_order:
        rng.shuffle(blocks)
    # 20 s at the start: on a real headset the first ~1.5 min can be very noisy while the gel and
    # the wearer settle (seen in the first real session), so nothing is scored during that time.
    steps = [Step("all", "settle", "", "Sit still and relax while the signal settles. Look at the dot.", 20)]
    for pairs in blocks:
        for rest, action in pairs:
            steps += [rest, action]
    return steps


TEST_ITEMS = {  # input -> list of (cls, text, seconds, fake actions, expected events, extras)
    "ssvep": [(f"{f:g}", "Stare at the glowing circle", 6, [("look", 5.6, 0.3, f)], ("ssvep",), {"flicker": f})
              for f in SSVEP_FREQS],
    "jaw": [("once", "Clench your jaw once", 3, [("clench", 0.5, 0.4)], ("clench", "hold"), {}),
            ("hold", "Clench and HOLD until it registers", 3, [("clench", 1.6, 0.4)], ("hold",), {})],
    "brow": [("once", "Raise your eyebrows once", 3, [("brow", 0.6, 0.4)], ("brow",), {})],
    "blink": [("once", "Blink once", 2.5, [("blink", 0.25, 0.4)], ("blink",), {}),
              ("double", "Blink twice, quickly", 3, [("blink", 0.25, 0.4), ("blink", 0.25, 0.8)],
               ("double_blink",), {})],
    "glance": [("left", "Look at the LEFT arrow, then back", 3, [("glance_left", 0.8, 0.35)],
                ("glance_left",), {"arrow": "left"}),
               ("right", "Look at the RIGHT arrow, then back", 3, [("glance_right", 0.8, 0.35)],
                ("glance_right",), {"arrow": "right"})],
    "eyes_closed": [("closed", "Close your eyes. Open them at the beep.", 5, [("eyes_closed", 4.3, 0.5)],
                     ("eyes_closed",), {})],
    "fist": [("right", "Squeeze your RIGHT fist", 5, [("fist_right", 4.2, 0.4)], ("fist_right",), {}),
             ("left", "Squeeze your LEFT fist", 5, [("fist_left", 4.2, 0.4)], ("fist_left",), {})],
}
# events that belong to each input (so a blink during a "double blink" cue isn't a false alarm)
FAMILY = {"ssvep": {"ssvep"}, "jaw": {"clench", "hold"}, "brow": {"brow"}, "blink": {"blink", "double_blink"},
          "glance": {"glance_left", "glance_right"}, "eyes_closed": {"eyes_closed"},
          "fist": {"fist_right", "fist_left"}}
IGNORED_EVENTS = {"clench_start", "clench_end", "eyes_open"}


def build_test(inputs, reps=3, seed=None):
    rng = random.Random(seed)
    cues = []
    for inp in inputs:
        n = 1 if inp == "ssvep" else reps  # 10 flicker rates: once each keeps the test short
        for cls, text, dur, fake, expect, extra in TEST_ITEMS[inp]:
            cues += [Step(inp, "action", cls, text, dur, fake, expect=expect, **extra) for _ in range(n)]
    rng.shuffle(cues)
    steps = [Step("all", "settle", "", "Relax, sit still and look at the dot", 4)]
    for c in cues:
        steps.append(Step(c.input, "rest", "rest", "Relax — look at the dot", round(rng.uniform(3, 4.5), 1)))
        steps.append(c)
    return steps


def total_seconds(steps):
    return sum(s.dur for s in steps)


# ------------------------------------------------------------------ scoring helpers
def verdict(acc):
    return "Good" if acc >= 0.9 else "OK" if acc >= 0.75 else "Poor"


def _arrays(rows):
    cols = sorted({k for r in rows for k in r})
    return {k: np.array([r.get(k, np.nan) for r in rows], float) for k in cols}


def _win(A, col, a, b):
    if col not in A:
        return np.array([])
    m = (A["t"] >= a) & (A["t"] <= b)
    v = A[col][m]
    return v[~np.isnan(v)]


def _chop(a, b, length):
    if length <= 0.2:
        return []
    out, x = [], a
    while x + length <= b + 1e-9:
        out.append((x, x + length))
        x += length
    return out


def _separate(act, rest, higher=True, geometric=True, p_act=25, p_rest=90):
    """Best threshold between action values and rest values, and the balanced accuracy there."""
    act, rest = np.asarray(act, float), np.asarray(rest, float)
    if len(act) == 0 or len(rest) == 0:
        return None
    if higher:
        a_edge, r_edge = np.percentile(act, p_act), np.percentile(rest, p_rest)
    else:
        a_edge, r_edge = np.percentile(act, 100 - p_act), np.percentile(rest, 100 - p_rest)
    if geometric and a_edge > 0 and r_edge > 0:
        thr = float(np.sqrt(a_edge * r_edge))
    else:
        thr = float((a_edge + r_edge) / 2)
    if higher:
        acc = (np.mean(act >= thr) + np.mean(rest < thr)) / 2
    else:
        acc = (np.mean(act <= thr) + np.mean(rest > thr)) / 2
    return {"thr": thr, "acc": float(acc), "act": float(np.median(act)), "rest": float(np.median(rest))}


def _result(key, inp, item, sep, thr_fmt, detail, settings):
    acc = sep["acc"] if sep else 0.0
    return {"key": key, "input": inp, "item": item, "accuracy": round(acc, 3), "verdict": verdict(acc),
            "threshold": thr_fmt, "detail": detail, "settings": settings if sep else {}}


def _clamp(x, lo, hi):
    return float(min(max(x, lo), hi))


# ------------------------------------------------------------------ calibration scoring
def analyze_calibration(rows, cues):
    """rows: feature rows; cues: dicts with input, phase, cls, t_start, t_end. Returns (results, settings_update)."""
    A = _arrays(rows)
    if "t" not in A or len(A["t"]) < 50:
        return [], {}
    by = lambda inp, phase, cls=None: [c for c in cues if c["input"] == inp and c["phase"] == phase
                                       and (cls is None or c["cls"] == cls)]
    results = []

    def peaks(col, inp, sign=1.0, cls=None, lag=(0.2, 0.6)):
        acts = [c for c in by(inp, "action", cls)]
        length = (acts[0]["t_end"] - acts[0]["t_start"]) + lag[1] - lag[0] if acts else 2.0
        a = [np.max(sign * v) for c in acts if len(v := _win(A, col, c["t_start"] + lag[0], c["t_end"] + lag[1]))]
        r = [np.max(sign * v) for c in by(inp, "rest")
             for w in _chop(c["t_start"] + 0.8, c["t_end"], min(length, c["t_end"] - c["t_start"] - 0.8))
             if len(v := _win(A, col, *w))]
        return a, r

    def levels(col, inp, cls=None, skip=1.5):
        a = [np.median(v) for c in by(inp, "action", cls) if len(v := _win(A, col, c["t_start"] + skip, c["t_end"] + 0.3))]
        r = [np.median(v) for c in by(inp, "rest") if len(v := _win(A, col, c["t_start"] + 1.5, c["t_end"]))]
        return a, r

    inputs = {c["input"] for c in cues}

    if "jaw" in inputs:
        a, r = peaks("jaw", "jaw")
        sep = _separate(a, r)
        settings = {}
        if sep:
            thr = _clamp(sep["thr"], 2.5, 200)
            durs = []  # how long a quick clench stays over threshold -> sets the "hold" time
            for c in by("jaw", "action"):
                m = (A["t"] >= c["t_start"]) & (A["t"] <= c["t_end"] + 0.6)
                over = A["jaw"][m] >= thr
                if over.any():
                    durs.append(over.sum() * 0.04)
            hold = _clamp(max(1.0, 1.5 * np.percentile(durs, 90)), 0.8, 2.0) if durs else 1.0
            settings = {"jaw_factor": round(thr, 2), "hold_s": round(hold, 2)}
        results.append(_result("jaw", "jaw", "Jaw clench", sep, settings.get("jaw_factor"),
                               f"clench {sep['act']:.1f}× vs rest {sep['rest']:.1f}×" if sep else "no data", settings))

    if "brow" in inputs:
        a, r = peaks("brow", "brow")
        sep = _separate(a, r)
        s = {"brow_factor": round(_clamp(sep["thr"], 2.5, 200), 2)} if sep else {}
        results.append(_result("brow", "brow", "Eyebrow raise", sep, s.get("brow_factor"),
                               f"raise {sep['act']:.1f}× vs rest {sep['rest']:.1f}×" if sep else "no data", s))

    if "blink" in inputs:
        signed = [v[np.argmax(np.abs(v))] for c in by("blink", "action")
                  if len(v := _win(A, "blink", c["t_start"] + 0.2, c["t_end"] + 0.6))]
        sign = 1 if (not signed or np.median(signed) >= 0) else -1
        a, r = peaks("blink", "blink", sign)
        sep = _separate(a, r, geometric=False)
        s = {"blink_z": round(_clamp(sep["thr"], 2.5, 500), 2), "blink_sign": sign} if sep else {}
        results.append(_result("blink", "blink", "Blink", sep, s.get("blink_z"),
                               f"blink {sep['act']:.0f} vs rest {sep['rest']:.0f} (noise units)" if sep else "no data", s))

    if "glance" in inputs:
        signed = [v[np.argmax(np.abs(v))] for c in by("glance", "action", "left")
                  if len(v := _win(A, "heog", c["t_start"] + 0.2, c["t_end"] + 0.6))]
        sign = 1 if (not signed or np.median(signed) >= 0) else -1
        for side, sgn in (("left", sign), ("right", -sign)):
            a, r = peaks("heog", "glance", sgn, cls=side)
            sep = _separate(a, r, geometric=False)
            s = {f"glance_{side}_z": round(_clamp(sep["thr"], 2.0, 500), 2), "heog_sign": sign} if sep else {}
            results.append(_result(f"glance_{side}", "glance", f"Glance {side}", sep, s.get(f"glance_{side}_z"),
                                   f"look {sep['act']:.0f} vs rest {sep['rest']:.0f} (noise units)" if sep else "no data", s))

    if "eyes_closed" in inputs:
        a, r = levels("alpha", "eyes_closed")
        sep = _separate(a, r, p_act=0, p_rest=100)
        s = {"alpha_factor": round(_clamp(sep["thr"], 1.2, 50), 2)} if sep else {}
        results.append(_result("eyes_closed", "eyes_closed", "Eyes closed", sep, s.get("alpha_factor"),
                               f"alpha {sep['act']:.1f}× vs open {sep['rest']:.1f}×" if sep else "no data", s))

    if "fist" in inputs:
        for side, col, other in (("right", "mu_l", "mu_r"), ("left", "mu_r", "mu_l")):
            a, r = levels(col, "fist", cls=side, skip=1.0)
            sep = _separate(a, r, higher=False, p_act=0, p_rest=100)
            lat = [np.median(_win(A, col, c["t_start"] + 1, c["t_end"])) - np.median(_win(A, other, c["t_start"] + 1, c["t_end"]))
                   for c in by("fist", "action", side) if len(_win(A, col, c["t_start"] + 1, c["t_end"]))]
            ok_side = bool(lat) and np.median(lat) < 0
            s = {f"fist_{side}_ratio": round(_clamp(sep["thr"], 0.2, 0.95), 3)} if sep else {}
            detail = (f"rhythm {sep['act']:.2f}× vs rest {sep['rest']:.2f}×; "
                      f"{'drops more on the correct side' if ok_side else 'NOT clearly one-sided'}") if sep else "no data"
            res = _result(f"fist_{side}", "fist", f"Fist {side}", sep, s.get(f"fist_{side}_ratio"), detail, s)
            if sep and not ok_side and res["verdict"] != "Poor":
                res["verdict"] = "OK"
            results.append(res)

    if "ssvep" in inputs:
        # score the rates that were actually tested (all 10, or just the viewer's in a quick calibration)
        freqs = sorted({float(c["cls"]) for c in cues if c["input"] == "ssvep" and c["phase"] == "action"})
        cols = [f"ss_{f:g}" for f in freqs]
        for f in freqs:
            col, cls = f"ss_{f:g}", f"{f:g}"
            act = np.concatenate([_win(A, col, c["t_start"] + 2.0, c["t_end"] + 0.2)
                                  for c in by("ssvep", "action", cls)] or [np.array([])])
            others = [c for c in cues if c["input"] == "ssvep" and c["phase"] in ("rest", "action") and c["cls"] != cls]
            non = np.concatenate([_win(A, col, c["t_start"] + 2.0, c["t_end"]) for c in others] or [np.array([])])
            sep = _separate(act, non, geometric=False, p_act=30, p_rest=90)
            # also: how often is this rate the top match while staring at it?
            hits = tot = 0
            for c in by("ssvep", "action", cls):
                m = (A["t"] >= c["t_start"] + 2.0) & (A["t"] <= c["t_end"] + 0.2)
                if all(k in A for k in cols) and m.any():
                    best = np.nanargmax(np.vstack([A[k][m] for k in cols]), axis=0)
                    hits += int(np.sum(best == freqs.index(f)))
                    tot += int(m.sum())
            top = hits / tot if tot else 0.0
            # graded on: does THIS rate's signal clearly rise when stared at? (the head-to-head
            # "top match" is shown too, but a strong 10 Hz alpha rhythm can win it unfairly)
            s = {"ssvep_thr": {cls: round(_clamp(sep["thr"], 0.2, 0.9), 3)}} if sep else {}
            res = _result(f"ssvep_{cls}", "ssvep", f"Flicker {cls} Hz", sep, s.get("ssvep_thr", {}).get(cls),
                          (f"stare {sep['act']:.2f} vs not {sep['rest']:.2f}; top of all rates {top:.0%}"
                           + ("; HIGH WHEN NOT STARING (alpha?) - may fire falsely" if sep["rest"] > 0.4 else ""))
                          if sep else "no data", s)
            if sep and sep["rest"] > 0.4 and res["verdict"] == "Good":
                res["verdict"] = "OK"
            results.append(res)

    update = {}
    for r in results:
        if r["verdict"] != "Poor":
            for k, v in r["settings"].items():
                if isinstance(v, dict):
                    update.setdefault(k, {}).update(v)
                else:
                    update[k] = v
    return results, update


# ------------------------------------------------------------------ live test scoring
def score_test(events, cues):
    events = [e for e in events if e["type"] not in IGNORED_EVENTS]
    actions = [c for c in cues if c["phase"] == "action"]
    rests = [c for c in cues if c["phase"] == "rest"]
    families = set().union(*(FAMILY[c["input"]] for c in actions)) if actions else set()
    per_item = {}
    for c in actions:
        lo, hi = c["t_start"], c["t_end"] + 0.7
        expect = set(c["expect"])
        hit = [e for e in events if e["type"] in expect and lo <= e["t"] <= hi
               and (e["type"] != "ssvep" or f"{e.get('freq', 0):g}" == c["cls"])]
        key = (c["input"], c["cls"])
        d = per_item.setdefault(key, {"n": 0, "hits": 0, "lat": [], "wrong": 0, "text": c["text"]})
        d["n"] += 1
        if hit:
            d["hits"] += 1
            d["lat"].append(hit[0]["t"] - lo)
        fam = FAMILY[c["input"]]
        d["wrong"] += sum(1 for e in events if lo <= e["t"] <= hi and e["type"] in families and e["type"] not in fam)
    rest_time = sum(c["t_end"] - c["t_start"] - 0.7 for c in rests if c["t_end"] - c["t_start"] > 0.7)
    false_alarms = [e for e in events if e["type"] in families
                    and any(c["t_start"] + 0.7 <= e["t"] <= c["t_end"] for c in rests)]
    results = []
    for (inp, cls), d in per_item.items():
        acc = d["hits"] / d["n"]
        lat = f"{np.mean(d['lat']):.1f} s" if d["lat"] else "–"
        label = {"ssvep": f"Flicker {cls} Hz", "jaw": f"Jaw {'clench' if cls == 'once' else 'hold'}",
                 "brow": "Eyebrow raise", "blink": "Blink" if cls == "once" else "Double blink",
                 "glance": f"Glance {cls}", "eyes_closed": "Eyes closed", "fist": f"Fist {cls}"}[inp]
        results.append({"key": f"test_{inp}_{cls}", "input": inp, "item": label, "accuracy": round(acc, 3),
                        "verdict": verdict(acc), "threshold": None,
                        "detail": f"{d['hits']}/{d['n']} detected, reaction {lat}, {d['wrong']} wrong-type",
                        "settings": {}})
    fa_rate = len(false_alarms) / (rest_time / 60) if rest_time > 0 else 0.0
    summary = {"false_alarms": len(false_alarms), "rest_minutes": round(rest_time / 60, 2),
               "false_alarms_per_min": round(fa_rate, 2),
               "false_alarm_types": sorted({e["type"] for e in false_alarms})}
    return results, summary
