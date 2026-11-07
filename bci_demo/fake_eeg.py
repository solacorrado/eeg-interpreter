"""A fake eego headset for testing at home.

Publishes a 24-channel, 500 Hz EEG stream over LSL. The web page's Shift+arrow
and Shift+J shortcuts tell it what to fake:
  - "looking at" an arrow -> adds that arrow's flicker rate to O1/Oz/O2/POz
  - "clench"              -> adds a burst of fast muscle noise to T7/T8/M1/M2
So the real detectors and the real LSL code get exercised end to end.
"""
import threading
import time

import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock
from scipy.signal import butter, sosfilt

RATE = 500
CHANNELS = ["Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8", "M1", "T7", "C3", "Cz",
            "C4", "T8", "M2", "P7", "P3", "Pz", "P4", "P8", "POz", "O1", "Oz", "O2"]
NAME = "FakeEego"


class FakeHeadset:
    def __init__(self):
        # What the fake person is doing, stamped with real times. Each sample is generated
        # from these stamps, so the signal stays correct even if this thread runs late
        # (e.g. while a busy window keeps Python occupied) and has to catch up.
        self.gaze_changes = [(0.0, None)]   # (time, flicker frequency or None)
        self.clench_spans = []              # [start, end] (end = inf while held)
        self.lock = threading.Lock()
        info = StreamInfo(NAME, "EEG", len(CHANNELS), RATE, "float32", "fake-eego-bci-demo")
        chans = info.desc().append_child("channels")
        for n in CHANNELS:
            chans.append_child("channel").append_child_value("label", n)
        self.outlet = StreamOutlet(info)
        self.occ = [CHANNELS.index(c) for c in ("O1", "Oz", "O2", "POz")]
        self.tmp = [CHANNELS.index(c) for c in ("T7", "T8", "M1", "M2")]
        self.emg_sos = butter(4, [30, 120], btype="band", fs=RATE, output="sos")

    def look(self, freq):
        with self.lock:
            self.gaze_changes.append((time.time(), float(freq) if freq else None))

    def clench(self, how=True):
        """True = a quick 0.5 s clench, "on" = start clenching until "off"."""
        now = time.time()
        with self.lock:
            if how == "off":
                for span in self.clench_spans:
                    if span[1] == float("inf"):
                        span[1] = now
            else:
                self.clench_spans.append([now, float("inf") if how == "on" else now + 0.5])

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        rng = np.random.default_rng()
        chunk = 20
        n = 0
        start = local_clock()
        wall0 = time.time()
        alpha_amp = 4.0
        while True:
            t = (n + np.arange(chunk)) / RATE
            wall = wall0 + t                      # real time of each sample
            data = rng.normal(0, 6, (chunk, len(CHANNELS)))
            alpha_amp = float(np.clip(alpha_amp + rng.normal(0, 0.2), 1, 8))
            data[:, self.occ] += (alpha_amp * np.sin(2 * np.pi * 10.2 * t))[:, None]
            with self.lock:
                old = [g for g in self.gaze_changes if g[0] <= wall[0]]
                self.gaze_changes = old[-1:] + [g for g in self.gaze_changes if g[0] > wall[0]]
                self.clench_spans = [s for s in self.clench_spans if s[1] > wall[0] - 10]
                gaze_changes, spans = list(self.gaze_changes), [tuple(s) for s in self.clench_spans]
            gaze = np.full(chunk, np.nan)         # frequency being looked at, per sample
            for g_t, freq in sorted(gaze_changes):
                gaze[wall >= g_t] = np.nan if freq is None else freq
            looking = ~np.isnan(gaze)
            if looking.any():
                f = np.where(looking, gaze, 0.0)
                ss = 2.5 * np.sin(2 * np.pi * f * t) + 1.2 * np.sin(2 * np.pi * 2 * f * t + 0.7)
                data[:, self.occ] += (ss * looking)[:, None]
            clenching = np.zeros(chunk, bool)
            for a, b in spans:
                clenching |= (wall >= a) & (wall < b)
            if clenching.any():
                emg = sosfilt(self.emg_sos, rng.normal(0, 120, (chunk, len(self.tmp))), axis=0)
                data[:, self.tmp] += emg * clenching[:, None]
            self.outlet.push_chunk(data.astype(np.float32).tolist())
            n += chunk
            delay = start + n / RATE - local_clock()
            if delay > 0:
                time.sleep(delay)
