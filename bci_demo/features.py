"""Turn raw EEG into a few easy-to-read numbers ("features"), 25 times a second.

  jaw    jaw-muscle power (30-100 Hz) at T7/T8/M1/M2, as a multiple of resting
  brow   forehead-muscle power (30-100 Hz) at Fp1/Fpz/Fp2/F7/F8, multiple of resting
  blink  slow eye signal at Fp1/Fpz/Fp2 (0.2-8 Hz), in units of resting noise ("z")
  heog   left/right eye movement: F7 minus F8 (0.2-8 Hz), in units of resting noise
  alpha  8-13 Hz power at O1/Oz/O2/POz (rises when eyes close), multiple of resting
  mu_l   8-30 Hz power at C3 (left motor area; drops when the RIGHT hand moves)
  mu_r   8-30 Hz power at C4 (right motor area; drops when the LEFT hand moves)

"Resting" levels are learned in the first few seconds and then keep updating slowly
during quiet moments, so slow drifts in electrode contact don't matter. Time is
counted from the samples themselves, so data that arrives late is still timed right.

SSVEPScorer separately scores how well the back of the head matches each flicker rate.
"""
from collections import deque

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, sosfiltfilt, tf2sos

from detect import cca_corr

DEFAULT_24 = ["Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8", "M1", "T7", "C3", "Cz",
              "C4", "T8", "M2", "P7", "P3", "Pz", "P4", "P8", "POz", "O1", "Oz", "O2"]
GROUPS = {
    "occ": ["O1", "Oz", "O2", "POz"],
    "temporal": ["T7", "T8", "M1", "M2"],
    "frontal": ["Fp1", "Fpz", "Fp2", "F7", "F8"],
    "fp": ["Fp1", "Fpz", "Fp2"],
}
FEATURES = ["jaw", "brow", "blink", "heog", "alpha", "mu_l", "mu_r"]


def resolve_names(names, n):
    """Channel names from the stream; if it sends none, assume the standard eego 24 layout."""
    names = list(names or [])
    if len(names) == n and any(names):
        return names
    if n == 24:
        return list(DEFAULT_24)
    return [f"ch{i}" for i in range(n)]


def pick(names, wanted):
    idx = [names.index(w) for w in wanted if w in names]
    return idx or list(range(len(names)))


class StreamFilter:
    """A filter that keeps its memory between chunks (so chunk edges don't glitch).
    It starts as if the first sample had always been there, so big DC offsets don't ring."""

    def __init__(self, sos):
        self.sos = sos
        self.zi = None

    def __call__(self, x):
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


class Baseline:
    """Resting level = median of recent quiet values."""

    def __init__(self, n=750):
        self.h = deque(maxlen=n)
        self.cache = None
        self.k = 0

    def learn(self, p):
        self.h.append(p)
        self.k += 1

    def ratio(self, p):
        if self.cache is None or self.k >= 10:
            self.cache = float(np.median(self.h)) if self.h else None
            self.k = 0
        return p / self.cache if self.cache else 1.0


class Spread:
    """Resting centre and spread (robust standard deviation) of a signal, for z-scores."""

    def __init__(self, n=500):
        self.h = deque(maxlen=n)
        self.cache = (0.0, None)
        self.k = 0

    def learn(self, v):
        self.h.append(v)
        self.k += 1

    def z(self, v):
        if self.cache[1] is None or self.k >= 10:
            if len(self.h) >= 10:
                a = np.asarray(self.h)
                c = float(np.median(a))
                self.cache = (c, max(1.4826 * float(np.median(np.abs(a - c))), 1e-12))
            self.k = 0
        c, s = self.cache
        return (v - c) / s if s else 0.0


class FeatureExtractor:
    def __init__(self, rate, names, calib_s=3.0):
        self.rate = rate
        self.names = names
        self.ch_occ, self.ch_tmp = pick(names, GROUPS["occ"]), pick(names, GROUPS["temporal"])
        self.ch_front, self.ch_fp = pick(names, GROUPS["frontal"]), pick(names, GROUPS["fp"])
        ix = lambda n: names.index(n) if n in names else None
        self.f7, self.f8, self.c3, self.c4 = ix("F7"), ix("F8"), ix("C3"), ix("C4")

        hi = min(100.0, rate * 0.45)
        emg = butter(4, [30, hi], btype="band", fs=rate, output="sos")
        if hi > 62:  # remove 60 Hz power-line hum
            b, a = iirnotch(60, 30, fs=rate)
            emg = np.vstack([emg, tf2sos(b, a)])
        eog = butter(2, [0.2, 8], btype="band", fs=rate, output="sos")
        self.f_jaw, self.f_brow = StreamFilter(emg), StreamFilter(emg)
        self.f_blink, self.f_heog = StreamFilter(eog), StreamFilter(eog)
        self.f_alpha = StreamFilter(butter(4, [8, 13], btype="band", fs=rate, output="sos"))
        self.f_mu = StreamFilter(butter(4, [8, 30], btype="band", fs=rate, output="sos"))

        self.step_n = max(1, int(0.04 * rate))       # one feature row per 40 ms of data
        fast, slow = int(0.2 * rate), int(1.0 * rate)  # averaging windows
        self.p = {"jaw": deque(maxlen=fast), "brow": deque(maxlen=fast), "alpha": deque(maxlen=slow),
                  "mu_l": deque(maxlen=slow), "mu_r": deque(maxlen=slow)}
        self.base = {k: Baseline() for k in self.p}
        self.spread = {"blink": Spread(), "heog": Spread()}
        self.calib_s = calib_s
        self.n = 0
        self.last = None

    @property
    def calibrating(self):
        return self.n / self.rate < self.calib_s

    def process(self, chunk, ts):
        """chunk: samples x channels, ts: one timestamp per sample. Returns feature rows."""
        rows = []
        for i in range(0, len(chunk), self.step_n):
            j = min(i + self.step_n, len(chunk))
            rows.append(self._step(chunk[i:j], float(ts[j - 1])))
        return rows

    def _step(self, x, t):
        self.n += len(x)
        y = self.f_jaw(x[:, self.ch_tmp])
        self.p["jaw"].extend((y ** 2).mean(axis=1))
        y = self.f_brow(x[:, self.ch_front])
        self.p["brow"].extend((y ** 2).mean(axis=1))
        y = self.f_alpha(x[:, self.ch_occ])
        self.p["alpha"].extend((y ** 2).mean(axis=1))
        if self.c3 is not None and self.c4 is not None:
            y = self.f_mu(x[:, [self.c3, self.c4]])
            self.p["mu_l"].extend(y[:, 0] ** 2)
            self.p["mu_r"].extend(y[:, 1] ** 2)
        blink = self.f_blink(x[:, self.ch_fp].mean(axis=1, keepdims=True))[:, 0]
        if self.f7 is not None and self.f8 is not None:
            heog = self.f_heog(x[:, [self.f7]] - x[:, [self.f8]])[:, 0]
        else:
            heog = np.zeros(len(x))

        row = {"t": t}
        cal = self.calibrating
        for k, d in self.p.items():
            p = float(np.mean(d)) if d else 0.0
            r = self.base[k].ratio(p) if p > 0 else 1.0
            row[k] = r
            quiet = (r < 2.0) if k in ("jaw", "brow") else (r < 1.6) if k == "alpha" else (0.7 < r < 1.5)
            if p > 0 and (cal or quiet):
                self.base[k].learn(p)
        for k, sig in (("blink", blink), ("heog", heog)):
            sp = self.spread[k]
            peak = float(sig[np.argmax(np.abs(sig))])  # biggest swing in this 40 ms
            row[k] = sp.z(peak)
            end = float(sig[-1])
            if cal or abs(sp.z(end)) < 3:
                sp.learn(end)
        self.last = row
        return row


class SSVEPScorer:
    """How well the back of the head matches each flicker rate (0-1, CCA), over the last 2 s."""

    def __init__(self, rate, names, window_s=2.0, harmonics=2):
        self.rate = rate
        self.ch = pick(names, GROUPS["occ"])
        self.n = int(window_s * rate)
        self.keep = self.n + int(0.5 * rate)
        self.buf = np.zeros((0, len(self.ch)))
        self.sos = butter(4, [6, min(45.0, rate * 0.45)], btype="band", fs=rate, output="sos")
        self.harmonics = harmonics
        self.refs = {}

    def add(self, chunk):
        self.buf = np.vstack([self.buf, chunk[:, self.ch]])[-self.keep:]

    def _ref(self, f):
        if f not in self.refs:
            t = np.arange(self.n) / self.rate
            self.refs[f] = np.column_stack([fn(2 * np.pi * h * f * t) for h in range(1, self.harmonics + 1)
                                            for fn in (np.sin, np.cos)])
        return self.refs[f]

    def scores(self, freqs):
        if not freqs or len(self.buf) < self.keep:
            return {}
        x = sosfiltfilt(self.sos, self.buf, axis=0)[-self.n:]
        return {f: cca_corr(x, self._ref(f)) for f in freqs}
