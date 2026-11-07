"""Turn raw EEG into commands.

Two detectors:

SSVEP (where are you looking?)
    Each on-screen arrow flickers at its own rate (e.g. 12 Hz). When you stare
    at it, the visual cortex at the back of the head (O1, Oz, O2, POz) picks up
    that same rate. Every 0.25 s we compare the last 2 s of those channels with
    pure sine waves at each arrow's rate (a method called CCA) and pick the
    best match. The same winner 3 times in a row = a command.

Jaw clench
    Clenching fires the big chewing muscles next to the temporal electrodes
    (T7, T8, M1, M2). Muscle noise is fast (30-100 Hz) and huge compared with
    brain signals, so a sudden jump in 30-100 Hz power to several times its
    normal level = a clench.
"""
import time
from collections import deque

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, sosfiltfilt, tf2sos

OCCIPITAL = ["O1", "Oz", "O2", "POz"]
TEMPORAL = ["T7", "T8", "M1", "M2"]


def pick(names, wanted):
    idx = [names.index(n) for n in wanted if n in names]
    return idx or list(range(len(names)))


def cca_corr(x, y):
    """Largest canonical correlation between two sets of signals (samples x channels)."""
    qx, _ = np.linalg.qr(x - x.mean(0))
    qy, _ = np.linalg.qr(y - y.mean(0))
    return float(np.linalg.svd(qx.T @ qy, compute_uv=False)[0])


class SSVEPDetector:
    def __init__(self, rate, names, window_s=2.0, threshold=0.42, needed=3, harmonics=2):
        self.rate = rate
        self.ch = pick(names, OCCIPITAL)
        self.n = int(window_s * rate)
        self.threshold = threshold
        self.needed = needed
        self.harmonics = harmonics
        self.sos = butter(4, [6, min(45, rate * 0.45)], btype="band", fs=rate, output="sos")
        self.freqs = []
        self.history = deque(maxlen=needed)
        self.blocked_until = 0.0
        self.scores = {}

    def set_freqs(self, freqs):
        self.freqs = sorted(float(f) for f in freqs)
        self.history.clear()
        self.scores = {}

    def block(self, seconds):
        self.blocked_until = time.time() + seconds
        self.history.clear()

    def step(self, buf):
        """buf: samples x channels (all channels). Returns a frequency or None."""
        if not self.freqs or len(buf) < self.n + self.rate // 2:
            return None
        x = sosfiltfilt(self.sos, buf[-(self.n + int(self.rate // 2)):, self.ch], axis=0)[-self.n:]
        t = np.arange(self.n) / self.rate
        self.scores = {}
        for f in self.freqs:
            ref = np.column_stack([fn(2 * np.pi * h * f * t)
                                   for h in range(1, self.harmonics + 1) for fn in (np.sin, np.cos)])
            self.scores[f] = cca_corr(x, ref)
        if time.time() < self.blocked_until:
            return None
        best = max(self.scores, key=self.scores.get)
        others = [v for k, v in self.scores.items() if k != best]
        margin = self.scores[best] - (max(others) if others else 0)
        self.history.append(best if self.scores[best] >= self.threshold and margin > 0.08 else None)
        if len(self.history) == self.needed and len(set(self.history)) == 1 and self.history[0]:
            self.block(1.0)  # short pause so one look = one command
            return best
        return None


class ClenchDetector:
    """Reports clench events, so the app can tell a quick clench from a held one:
      "start"  - a clench began
      "hold"   - it has lasted hold_s seconds (sent once, while still clenched)
      "clench" - a SHORT clench ended (released before hold_s)
      "end"    - a HELD clench ended
    """

    def __init__(self, rate, names, factor=5.0, hold_s=1.0, calib_s=3.0):
        self.rate = rate
        self.ch = pick(names, TEMPORAL)
        hi = min(100, rate * 0.45)
        sos = butter(4, [30, hi], btype="band", fs=rate, output="sos")
        if hi > 62:  # remove 60 Hz power-line hum
            b, a = iirnotch(60, 30, fs=rate)
            sos = np.vstack([sos, tf2sos(b, a)])
        self.sos = sos
        self.zi = np.repeat(sosfilt_zi(sos)[:, :, None], len(self.ch), axis=2)
        self.win = deque(maxlen=int(0.2 * rate))
        self.history = deque(maxlen=int(30 / 0.1))  # ~30 s of normal power levels
        self.factor = factor
        self.hold_s = hold_s
        self.calib_s = calib_s
        self.step_n = max(1, int(0.04 * rate))  # check every 40 ms of data
        self.n = 0                               # samples seen; time is counted in samples,
        self.ratio = 0.0                         # so late-arriving batches are timed correctly
        self.above = self.below = 0
        self.active = self.held = False
        self.start_t = self.last_end = -10.0

    @property
    def calibrating(self):
        return self.n / self.rate < self.calib_s

    def process(self, chunk):
        """Feed new samples (samples x all channels). Returns the list of events they contain."""
        events = []
        for i in range(0, len(chunk), self.step_n):
            part = chunk[i:i + self.step_n]
            y, self.zi = sosfilt(self.sos, part[:, self.ch], axis=0, zi=self.zi)
            self.win.extend((y ** 2).mean(axis=1))
            self.n += len(part)
            ev = self._step()
            if ev:
                events.append(ev)
        return events

    def _step(self):
        if len(self.win) < self.win.maxlen:
            return None
        p = float(np.mean(self.win))
        now = self.n / self.rate
        base = float(np.median(self.history)) if self.history else p
        self.ratio = p / base if base > 0 else 0.0
        if self.calibrating:
            self.history.append(p)
            return None

        if not self.active:
            if self.ratio < self.factor * 0.5 and now - self.last_end > 0.5:
                self.history.append(p)  # only learn "normal" from quiet periods
            self.above = self.above + 1 if self.ratio >= self.factor else 0
            if self.above >= 2 and now - self.last_end > 0.2:
                self.active, self.held, self.below, self.start_t = True, False, 0, now
                return "start"
            return None

        # currently clenched: wait for release (power back near normal for 2 steps in a row)
        self.below = self.below + 1 if self.ratio < self.factor * 0.6 else 0
        if self.below >= 2:
            self.active, self.above, self.last_end = False, 0, now
            return "end" if self.held else "clench"
        if not self.held and now - self.start_t >= self.hold_s:
            self.held = True
            return "hold"
        return None
