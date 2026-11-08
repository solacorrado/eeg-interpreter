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
        self.enabled = None  # inputs the current screen uses (None = all)
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

    def set_inputs(self, names):
        """Only detect these inputs (e.g. {"jaw", "ssvep"} in the viewer); None = all."""
        with self.lock:
            self.enabled = set(names) if names else None
            if self.gestures:
                self.gestures.enabled = self.enabled

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
        """Keep a connection alive: if the stream goes quiet (eego left its live view, Wi-Fi
        dropped), drop it and find the stream again, so data resumes without a restart."""
        self.reconnect_name = None
        while self.running:
            self._session()
            if self.running:
                self.status.update(connected=False)
                self._emit({"type": "status", **self.status, "msg": "Stream went quiet — reconnecting…"})

    def _session(self):
        inlet, tries = None, 0
        while self.running and inlet is None:
            tries += 1
            if self.source_id:
                found = resolve_byprop("source_id", self.source_id, timeout=2)
            elif self.target:
                found = resolve_byprop("name", self.target, timeout=2)
            elif self.reconnect_name and tries <= 3:  # after a drop: look for the same stream first
                found = resolve_byprop("name", self.reconnect_name, timeout=2)
            else:  # the most EEG-like real stream; never the practice one
                found = sorted((s for s in resolve_streams(2) if s.name() != "FakeEego" and s.nominal_srate() > 0
                                and s.channel_count() >= 4 and "marker" not in s.type().lower()),
                               key=lambda s: (s.type().strip().lower() != "eeg", -s.channel_count()))
            if found:
                inlet = StreamInlet(found[0], max_buflen=30, processing_flags=proc_clocksync | proc_dejitter)
            else:
                self.status["connected"] = False
                self._emit({"type": "status", **self.status,
                            "msg": "Reconnecting to the EEG stream…" if self.reconnect_name else "Looking for EEG stream…"})
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
        self.reconnect_name = info.name()
        pipe = Pipeline(rate, names, self.settings, self.enabled)
        self.status.update(connected=True, stream=info.name(), stream_type=info.type(), rate=rate, n_channels=n,
                           names_from_stream=any(raw_names), missing=pipe.fx.missing)
        print(f"Connected to {info.name()!r}: {n} ch @ {rate} Hz; channels: {', '.join(names)}")
        if pipe.fx.missing:
            print("Missing channels (using fallbacks):", pipe.fx.missing)
        with self.lock:
            self.gestures = pipe.gestures
        last_status, last_data = 0.0, time.time()

        while self.running:
            chunk, ts = inlet.pull_chunk(timeout=0.1)
            now = time.time()
            if chunk:
                last_data = now
                arr, tsa = np.asarray(chunk, dtype=float), np.asarray(ts, dtype=float)
                with self.lock:
                    rec, freqs = self.recorder, list(self.freqs)
                    rows, evs = pipe.process(arr, tsa, freqs)
                if rec:
                    rec.add_raw(arr, tsa)
                    for row in rows:
                        rec.add_row(row)
                for e in evs:
                    if rec:
                        rec.add_event(e)
                    self._emit(e)
            if now - last_data > 5:  # quiet for 5 s: reconnect
                print(f"No data for {now - last_data:.0f} s from {info.name()!r}; reconnecting")
                break
            if now - last_status >= 0.1:
                last_status = now
                feats = {k: round(v, 3) for k, v in (pipe.fx.last or {}).items() if k != "t"}
                self._emit({"type": "status", **self.status, "calibrating": pipe.fx.calibrating,
                            "stalled": round(now - last_data, 1) if now - last_data > 2 else 0,
                            "clenching": pipe.gestures.jaw.active, "features": feats,
                            "scores": {fkey(f): round(v, 3) for f, v in pipe.scores.items()}})
        inlet.close_stream()


class Pipeline:
    """Raw EEG chunks -> feature rows + events. Used live (LiveEngine) and offline
    (reanalyze.py), so re-analysing a saved trial gives exactly what the app saw."""

    def __init__(self, rate, names, settings, enabled=None):
        self.fx = FeatureExtractor(rate, names)
        self.ss = SSVEPScorer(rate, names)
        self.gestures = Gestures(settings, enabled)
        self.scores = {}
        self.last_ss = -1e9

    def process(self, arr, tsa, freqs):
        rows, events = [], []
        self.ss.add(arr)
        for row in self.fx.process(arr, tsa):
            for f, v in self.scores.items():
                row[f"ss_{fkey(f)}"] = v
            rows.append(row)
            events += self.gestures.update(row)
        t_last = float(tsa[-1])
        if t_last - self.last_ss >= 0.25:
            self.last_ss = t_last
            self.scores = self.ss.scores(freqs)
            events += self.gestures.update_ssvep(t_last, self.scores)
        return rows, events
