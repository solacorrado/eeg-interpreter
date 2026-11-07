"""Turn feature rows into events, using one person's calibrated settings.

Events (each a dict with "type" and "t", the time in seconds on the LSL clock):
  clench_start, clench (quick, released), hold (held for hold_s), clench_end (held one released)
  brow                         eyebrow raise
  blink, double_blink          a double blink also sends two "blink"s first
  glance_left, glance_right    eyes move to one side
  eyes_closed, eyes_open
  fist_right, fist_left        motor rhythm drops over the opposite side of the head
  ssvep (+ "freq")             staring at a flicker target
"""


def fkey(f):
    return f"{float(f):g}"


class _EMG:
    def __init__(self):
        self.active = self.held = False
        self.above = self.below = 0
        self.t0 = 0.0
        self.last_end = -1e9


class Gestures:
    def __init__(self, settings):
        self.s = dict(settings)
        self.jaw, self.brow = _EMG(), _EMG()
        self.blink_on, self.blink_t0, self.last_blink = False, 0.0, -1e9
        self.glance = {"left": {"above": 0, "armed": True, "block": -1e9},
                       "right": {"above": 0, "armed": True, "block": -1e9}}
        self.eyes_closed, self.eyes_since, self.open_since = False, None, None
        self.fist = {"right": {"latched": False, "since": None}, "left": {"latched": False, "since": None}}
        self.ss_hist, self.ss_block = [], -1e9

    def apply(self, settings):
        self.s = dict(settings)

    # ------------------------------------------------------------------ per feature row
    def update(self, row):
        t = row["t"]
        ev = []
        ev += self._emg(row, t)
        ev += self._blink(row, t)
        ev += self._glance(row, t)
        ev += self._eyes(row, t)
        ev += self._fists(row, t)
        return ev

    def _emg(self, row, t):
        s, ev = self.s, []
        jv, bv = row["jaw"] / s["jaw_factor"], row["brow"] / s["brow_factor"]  # >1 = over threshold
        for m, val, other, other_m, name in ((self.jaw, jv, bv, self.brow, "jaw"),
                                             (self.brow, bv, jv, self.jaw, "brow")):
            if not m.active:
                # whichever muscle is further over its own threshold wins
                m.above = m.above + 1 if (val >= 1 and val >= other) else 0
                if m.above >= 2 and t - m.last_end > 0.2 and not other_m.active:
                    m.active, m.held, m.below, m.t0 = True, False, 0, t
                    if name == "jaw":
                        ev.append({"type": "clench_start", "t": t})
            else:
                m.below = m.below + 1 if val < 0.6 else 0
                if m.below >= 2:
                    m.active, m.above, m.last_end = False, 0, t
                    self.ss_block = max(self.ss_block, t + 1.0)
                    if name == "jaw":
                        ev.append({"type": "clench_end" if m.held else "clench", "t": t})
                    elif t - m.t0 < 2.5:
                        ev.append({"type": "brow", "t": t})
                elif name == "jaw" and not m.held and t - m.t0 >= s["hold_s"]:
                    m.held = True
                    ev.append({"type": "hold", "t": t})
        return ev

    def _blink(self, row, t):
        v, thr = self.s["blink_sign"] * row["blink"], self.s["blink_z"]
        if not self.blink_on:
            if v >= thr:
                self.blink_on, self.blink_t0 = True, t
            return []
        if v < thr * 0.5:
            self.blink_on = False
            if t - self.blink_t0 <= 0.6:
                ev = [{"type": "blink", "t": t}]
                if t - self.last_blink <= 0.9:
                    ev.append({"type": "double_blink", "t": t})
                    self.last_blink = -1e9
                else:
                    self.last_blink = t
                return ev
        return []

    def _glance(self, row, t):
        v = self.s["heog_sign"] * row["heog"]
        ev = []
        for side, val, thr, other in (("left", v, self.s["glance_left_z"], "right"),
                                      ("right", -v, self.s["glance_right_z"], "left")):
            g = self.glance[side]
            if not g["armed"]:
                if val < thr * 0.5:
                    g["armed"] = True
                continue
            g["above"] = g["above"] + 1 if val >= thr else 0
            if g["above"] >= 2 and t >= g["block"]:
                ev.append({"type": f"glance_{side}", "t": t})
                g["armed"], g["above"], g["block"] = False, 0, t + 1.0
                o = self.glance[other]  # ignore the eyes coming back to the centre
                o["block"] = max(o["block"], t + 1.5)
        return ev

    def _eyes(self, row, t):
        a, thr = row["alpha"], self.s["alpha_factor"]
        if not self.eyes_closed:
            if a >= thr:
                self.eyes_since = self.eyes_since or t
                if t - self.eyes_since >= 0.8:
                    self.eyes_closed, self.open_since = True, None
                    return [{"type": "eyes_closed", "t": t}]
            else:
                self.eyes_since = None
        else:
            if a < thr * 0.7:
                self.open_since = self.open_since or t
                if t - self.open_since >= 0.4:
                    self.eyes_closed, self.eyes_since = False, None
                    return [{"type": "eyes_open", "t": t}]
            else:
                self.open_since = None
        return []

    def _fists(self, row, t):
        ev = []
        for side, key, other, thr_key in (("right", "mu_l", "mu_r", "fist_right_ratio"),
                                          ("left", "mu_r", "mu_l", "fist_left_ratio")):
            f, val, thr = self.fist[side], row[key], self.s[thr_key]
            if not f["latched"]:
                if val <= thr and val < 0.9 * row[other]:
                    f["since"] = f["since"] or t
                    if t - f["since"] >= 0.6:
                        f["latched"] = True
                        ev.append({"type": f"fist_{side}", "t": t})
                else:
                    f["since"] = None
            elif val > min(thr * 1.15, 0.95):
                f["latched"], f["since"] = False, None
        return ev

    # ------------------------------------------------------------------ every 0.25 s
    def update_ssvep(self, t, scores):
        if not scores or t < self.ss_block or self.jaw.active or self.brow.active:
            self.ss_hist = []
            return []
        ranked = sorted(scores, key=scores.get, reverse=True)
        best = ranked[0]
        margin = scores[best] - (scores[ranked[1]] if len(ranked) > 1 else 0.0)
        thr = self.s["ssvep_thr"].get(fkey(best), 0.42)
        self.ss_hist = (self.ss_hist + [best if scores[best] >= thr and margin > self.s["ssvep_margin"] else None])[-3:]
        if len(self.ss_hist) == 3 and self.ss_hist[0] is not None and len(set(self.ss_hist)) == 1:
            self.ss_hist, self.ss_block = [], t + 1.0
            return [{"type": "ssvep", "freq": float(best), "t": t}]
        return []
