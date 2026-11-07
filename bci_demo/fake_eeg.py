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
        self.gaze = None          # flicker frequency being "looked at"
        self.clench_until = 0.0
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
            self.gaze = float(freq) if freq else None

    def clench(self, how=True):
        """True = a quick 0.5 s clench, "on" = start clenching until "off"."""
        with self.lock:
            if how == "on":
                self.clench_until = float("inf")
            elif how == "off":
                self.clench_until = time.time()
            else:
                self.clench_until = time.time() + 0.5

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        rng = np.random.default_rng()
        chunk = 20
        n = 0
        start = local_clock()
        alpha_amp = 4.0
        while True:
            t = (n + np.arange(chunk)) / RATE
            data = rng.normal(0, 6, (chunk, len(CHANNELS)))
            alpha_amp = float(np.clip(alpha_amp + rng.normal(0, 0.2), 1, 8))
            data[:, self.occ] += (alpha_amp * np.sin(2 * np.pi * 10.2 * t))[:, None]
            with self.lock:
                gaze, clenching = self.gaze, time.time() < self.clench_until
            if gaze:
                ss = 2.5 * np.sin(2 * np.pi * gaze * t) + 1.2 * np.sin(2 * np.pi * 2 * gaze * t + 0.7)
                data[:, self.occ] += ss[:, None]
            if clenching:
                data[:, self.tmp] += sosfilt(self.emg_sos, rng.normal(0, 120, (chunk, len(self.tmp))), axis=0)
            self.outlet.push_chunk(data.astype(np.float32).tolist())
            n += chunk
            delay = start + n / RATE - local_clock()
            if delay > 0:
                time.sleep(delay)
