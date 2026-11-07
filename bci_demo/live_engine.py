"""Connects to an EEG headset over LSL and runs everything live in a background thread.

on_event(dict) is called with detector events (see gestures.py) and, 10 times a second,
{"type": "status", ...} carrying the connection state and the latest feature values.
A TrialRecorder can be attached to save raw data, features and events while it runs.
"""
import threading
import time

import numpy as np
from pylsl import StreamInlet, proc_clocksync, proc_dejitter, resolve_byprop, resolve_streams

from features import FeatureExtractor, SSVEPScorer, resolve_names
from gestures import Gestures, fkey


def find_streams(timeout=2.0):
    return [{"name": s.name(), "type": s.type(), "channels": s.channel_count(), "rate": s.nominal_srate(),
             "host": s.hostname()} for s in resolve_streams(timeout)]


class LiveEngine:
    def __init__(self, on_event, settings):
        self.on_event = on_event
        self.lock = threading.Lock()
        self.settings = dict(settings)
        self.freqs = []
        self.recorder = None
        self.gestures = None
        self.thread = None
        self.running = False
        self.target = None
        self.source_id = None
        self.info = {}
        self.status = {"connected": False, "stream": None}

    # ---------------------------------------------------------------- control (UI thread)
    def connect(self, stream_name=None, fake=False, source_id=None):
        """Start (or restart) looking for a stream: by unique source ID, by name, or the first of type EEG."""
        self.stop()
        self.target, self.source_id, self.running = stream_name, source_id, True
        self.status = {"connected": False, "stream": None, "fake": fake}
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None

    def apply_settings(self, settings):
        with self.lock:
            self.settings = dict(settings)
            if self.gestures:
                self.gestures.apply(settings)

    def set_freqs(self, freqs):
        with self.lock:
            self.freqs = [float(f) for f in freqs]

    def set_recorder(self, rec):
        with self.lock:
            self.recorder = rec

    # ---------------------------------------------------------------- background thread
    def _emit(self, m):
        try:
            self.on_event(m)
        except Exception:
            pass

    def _run(self):
        inlet = None
        while self.running and inlet is None:
            if self.source_id:
                found = resolve_byprop("source_id", self.source_id, timeout=2)
            elif self.target:
                found = resolve_byprop("name", self.target, timeout=2)
            else:  # first real EEG stream; never the practice one
                found = [s for s in resolve_byprop("type", "EEG", timeout=2) if s.name() != "FakeEego"]
            if found:
                inlet = StreamInlet(found[0], max_buflen=30, processing_flags=proc_clocksync | proc_dejitter)
            else:
                self.status["connected"] = False
                self._emit({"type": "status", **self.status, "msg": "Looking for EEG stream…"})
        if not self.running:
            return

        info = inlet.info()
        rate, n = info.nominal_srate(), info.channel_count()
        raw_names, ch = [], info.desc().child("channels").child("channel")
        for _ in range(n):
            raw_names.append(ch.child_value("label"))
            ch = ch.next_sibling()
        names = resolve_names(raw_names, n)
        self.info = {"name": info.name(), "rate": rate, "channels": names, "type": info.type(),
                     "names_from_stream": any(raw_names)}
        self.status.update(connected=True, stream=info.name(), rate=rate, n_channels=n,
                           names_from_stream=any(raw_names))
        print(f"Connected to {info.name()!r}: {n} ch @ {rate} Hz; channels: {', '.join(names)}")

        fx, ss = FeatureExtractor(rate, names), SSVEPScorer(rate, names)
        with self.lock:
            self.gestures = Gestures(self.settings)
        scores, last_ss, last_status = {}, -1e9, 0.0

        while self.running:
            chunk, ts = inlet.pull_chunk(timeout=0.1)
            with self.lock:
                rec, freqs = self.recorder, list(self.freqs)
            if chunk:
                arr, tsa = np.asarray(chunk, dtype=float), np.asarray(ts, dtype=float)
                if rec:
                    rec.add_raw(arr, tsa)
                ss.add(arr)
                for row in fx.process(arr, tsa):
                    for f, v in scores.items():
                        row[f"ss_{fkey(f)}"] = v
                    with self.lock:
                        evs = self.gestures.update(row)
                    if rec:
                        rec.add_row(row)
                    for e in evs:
                        if rec:
                            rec.add_event(e)
                        self._emit(e)
                t_last = float(tsa[-1])
                if t_last - last_ss >= 0.25:
                    last_ss = t_last
                    scores = ss.scores(freqs)
                    with self.lock:
                        evs = self.gestures.update_ssvep(t_last, scores)
                    for e in evs:
                        if rec:
                            rec.add_event(e)
                        self._emit(e)
            now = time.time()
            if now - last_status >= 0.1:
                last_status = now
                feats = {k: round(v, 3) for k, v in (fx.last or {}).items() if k != "t"}
                self._emit({"type": "status", **self.status, "calibrating": fx.calibrating,
                            "clenching": self.gestures.jaw.active, "features": feats,
                            "scores": {fkey(f): round(v, 3) for f, v in scores.items()}})
        inlet.close_stream()
