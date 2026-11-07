"""The EEG loop shared by the web app (server.py) and the desktop app (desktop_app.py).

Connects to an LSL EEG stream, runs the SSVEP and jaw-clench detectors, and calls
on_event(dict) from a background thread with one of:
  {"type": "ssvep", "freq": 13.0}          looked at the 13 Hz target
  {"type": "clench_start"}                 jaw clench began
  {"type": "clench"}                       short clench released
  {"type": "hold"}                         clench held for 1 s (sent while still held)
  {"type": "clench_end"}                   held clench released
  {"type": "status", "connected": ..., "stream": ..., "scores": {...}, "emg": ..., ...}  twice a second
"""
import threading
import time

import numpy as np
from pylsl import StreamInlet, resolve_byprop

from detect import ClenchDetector, SSVEPDetector

JAW_EVENTS = {"start": "clench_start", "hold": "hold", "clench": "clench", "end": "clench_end"}


class EEGEngine:
    def __init__(self, on_event, stream=None, fake=False, ssvep_threshold=0.42, clench_factor=5.0):
        self.on_event = on_event
        self.stream = stream
        self.ssvep_threshold = ssvep_threshold
        self.clench_factor = clench_factor
        self.freqs = []
        self.lock = threading.Lock()
        self.ssvep = None
        self.clench = None
        self.status = {"connected": False, "stream": None, "fake": fake}

    def set_freqs(self, freqs):
        """Which flicker rates are on screen right now (empty = stop looking for SSVEP)."""
        with self.lock:
            self.freqs = list(freqs)
            if self.ssvep:
                self.ssvep.set_freqs(self.freqs)

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _connect(self):
        while True:
            found = (resolve_byprop("name", self.stream, timeout=3) if self.stream
                     else resolve_byprop("type", "EEG", timeout=3))
            if found:
                return StreamInlet(found[0], max_buflen=10)
            self.status.update(connected=False, stream=None)
            self.on_event({"type": "status", **self.status, "msg": "Looking for EEG stream..."})

    def _run(self):
        inlet = self._connect()
        info = inlet.info()
        rate = info.nominal_srate()
        names = []
        ch = info.desc().child("channels").child("channel")
        for _ in range(info.channel_count()):
            names.append(ch.child_value("label"))
            ch = ch.next_sibling()
        print(f"Connected to {info.name()!r}: {info.channel_count()} ch @ {rate} Hz")
        print("Channels:", ", ".join(names) or "(no names)")
        self.status.update(connected=True, stream=info.name(), rate=rate)

        with self.lock:
            self.ssvep = SSVEPDetector(rate, names, threshold=self.ssvep_threshold)
            self.ssvep.set_freqs(self.freqs)
        self.clench = ClenchDetector(rate, names, factor=self.clench_factor)
        buf = np.zeros((0, info.channel_count()))
        keep = int(rate * 4)
        last_ssvep = last_status = 0.0

        while True:
            chunk, _ = inlet.pull_chunk(timeout=0.1)
            events = []
            if chunk:
                arr = np.asarray(chunk, dtype=float)
                buf = np.vstack([buf, arr])[-keep:]
                events = self.clench.process(arr)
            now = time.time()
            for ev in events:
                print("JAW", ev)
                self.on_event({"type": JAW_EVENTS[ev]})
            with self.lock:
                if self.clench.active or events:
                    self.ssvep.block(1.0)  # clench muscle noise would confuse SSVEP
                if now - last_ssvep >= 0.25:
                    last_ssvep = now
                    f = self.ssvep.step(buf)
                    if f:
                        print(f"SSVEP {f} Hz")
                        self.on_event({"type": "ssvep", "freq": f})
                scores = dict(self.ssvep.scores)
            if now - last_status >= 0.5:
                last_status = now
                self.on_event({"type": "status", **self.status,
                               "calibrating": self.clench.calibrating,
                               "clenching": self.clench.active,
                               "scores": {str(k): round(v, 2) for k, v in scores.items()},
                               "emg": round(self.clench.ratio, 1)})
