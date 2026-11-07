"""A fake eego headset for testing at home.

Publishes a 24-channel, 500 Hz EEG stream over LSL ("FakeEego") with a resting brain
signal, and can fake every input the app knows:
  look(freq)                     staring at a flicker target (SSVEP at the back of the head)
  do(action, seconds, delay)     one action; start(action) / stop(action) for held ones
  actions: clench, brow, blink, glance_left, glance_right, eyes_closed, fist_right, fist_left

Every change is stamped with real time and each sample is built from those stamps,
so the signal stays right even if this thread runs late, and actions can be scheduled
a little in the future (delay), like a person reacting to a cue.
"""
import os
import threading
import time

import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock
from scipy.signal import butter, sosfilt

RATE = 500
CHANNELS = ["Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8", "M1", "T7", "C3", "Cz",
            "C4", "T8", "M2", "P7", "P3", "Pz", "P4", "P8", "POz", "O1", "Oz", "O2"]
NAME = "FakeEego"
ACTIONS = ("clench", "brow", "blink", "glance_left", "glance_right", "eyes_closed", "fist_right", "fist_left")
INF = float("inf")


class FakeHeadset:
    def __init__(self):
        self.gaze_changes = [(0.0, None)]       # (time, flicker frequency or None)
        self.spans = {a: [] for a in ACTIONS}  # [start, end] per action
        self.lock = threading.Lock()
        # unique ID, so an app always reads ITS OWN fake headset even if another one is running
        self.source_id = f"fake-eego-{os.getpid()}-{id(self)}"
        info = StreamInfo(NAME, "EEG", len(CHANNELS), RATE, "float32", self.source_id)
        chans = info.desc().append_child("channels")
        for n in CHANNELS:
            chans.append_child("channel").append_child_value("label", n)
        self.outlet = StreamOutlet(info)
        ix = lambda names: [CHANNELS.index(c) for c in names]
        self.occ = ix(("O1", "Oz", "O2", "POz"))
        self.tmp = ix(("T7", "T8", "M1", "M2"))
        self.front = ix(("Fp1", "Fpz", "Fp2", "F7", "F8"))
        self.fp = ix(("Fp1", "Fpz", "Fp2"))
        self.f7, self.f8 = CHANNELS.index("F7"), CHANNELS.index("F8")
        self.c3, self.c4 = CHANNELS.index("C3"), CHANNELS.index("C4")
        self.emg_sos = butter(4, [30, 120], btype="band", fs=RATE, output="sos")
        self.started = False

    # ------------------------------------------------------------- controls
    def look(self, freq, delay=0.0, seconds=None):
        now = time.time() + delay
        with self.lock:
            self.gaze_changes.append((now, float(freq) if freq else None))
            if seconds is not None:
                self.gaze_changes.append((now + seconds, None))

    def do(self, action, seconds, delay=0.0):
        now = time.time() + delay
        with self.lock:
            self.spans[action].append([now, now + seconds])

    def start(self, action=None):
        """start() begins streaming (used by the web version); start(action) begins a held action."""
        if action is None:
            return self.begin()
        with self.lock:
            self.spans[action].append([time.time(), INF])

    def stop(self, action):
        now = time.time()
        with self.lock:
            for s in self.spans[action]:
                if s[1] == INF:
                    s[1] = now

    def clench(self, how=True):
        """Used by the web version: True = quick 0.5 s clench, "on"/"off" = hold."""
        if how == "on":
            self.start("clench")
        elif how == "off":
            self.stop("clench")
        else:
            self.do("clench", 0.5)

    def begin(self):
        if not self.started:
            self.started = True
            threading.Thread(target=self._run, daemon=True).start()

    # ------------------------------------------------------------- signal
    def _shape(self, action, spans, wall):
        """0..1 per sample: how much of the action is happening."""
        out = np.zeros(len(wall))
        for a, b in spans:
            inside = (wall >= a) & (wall < b)
            if not inside.any():
                continue
            if action == "blink":  # a smooth bump
                out[inside] = np.maximum(out[inside], np.sin(np.pi * (wall[inside] - a) / (b - a)) ** 2)
            elif action.startswith("glance"):  # quick move, hold, quick move back
                ramp = np.minimum(np.minimum(wall[inside] - a, b - wall[inside]) / 0.05, 1.0)
                out[inside] = np.maximum(out[inside], ramp)
            else:
                out[inside] = 1.0
        return out

    def _run(self):
        rng = np.random.default_rng()
        chunk, n = 20, 0
        start, wall0 = local_clock(), time.time()
        alpha_amp, ph_mu = 4.0, rng.uniform(0, 2 * np.pi)
        while True:
            t = (n + np.arange(chunk)) / RATE
            wall = wall0 + t
            data = rng.normal(0, 6, (chunk, len(CHANNELS)))
            with self.lock:
                old = [g for g in self.gaze_changes if g[0] <= wall[0]]
                self.gaze_changes = old[-1:] + [g for g in self.gaze_changes if g[0] > wall[0]]
                for a in ACTIONS:
                    self.spans[a] = [s for s in self.spans[a] if s[1] > wall[0] - 5]
                gaze_changes = sorted(self.gaze_changes)
                spans = {a: [tuple(s) for s in v] for a, v in self.spans.items()}
            sh = {a: self._shape(a, spans[a], wall) for a in ACTIONS}

            # resting brain rhythms: alpha at the back, mu/beta over the motor areas
            alpha_amp = float(np.clip(alpha_amp + rng.normal(0, 0.1), 3, 5))
            alpha = (alpha_amp + 12 * sh["eyes_closed"]) * np.sin(2 * np.pi * 10.2 * t)
            data[:, self.occ] += alpha[:, None]
            mu = 5 * np.sin(2 * np.pi * 10.5 * t + ph_mu) + 2.5 * np.sin(2 * np.pi * 21 * t)
            data[:, self.c3] += mu * (1 - 0.65 * sh["fist_right"])
            data[:, self.c4] += mu * (1 - 0.65 * sh["fist_left"])

            # staring at a flicker target
            gaze = np.full(chunk, np.nan)
            for g_t, freq in gaze_changes:
                gaze[wall >= g_t] = np.nan if freq is None else freq
            looking = ~np.isnan(gaze)
            if looking.any():
                f = np.where(looking, gaze, 0.0)
                ss = 2.5 * np.sin(2 * np.pi * f * t) + 1.2 * np.sin(2 * np.pi * 2 * f * t + 0.7)
                data[:, self.occ] += (ss * looking)[:, None]

            # muscles: jaw is strongest at the temples, eyebrows at the forehead
            for action, strong, weak in (("clench", self.tmp, self.front), ("brow", self.front, self.tmp)):
                m = sh[action]
                if m.any():
                    data[:, strong] += sosfilt(self.emg_sos, rng.normal(0, 120, (chunk, len(strong))), axis=0) * m[:, None]
                    data[:, weak] += sosfilt(self.emg_sos, rng.normal(0, 25, (chunk, len(weak))), axis=0) * m[:, None]

            # eyes: blinks show at the forehead; sideways looks pull F7 and F8 apart
            data[:, self.fp] += (150 * sh["blink"])[:, None]
            data[:, self.f7] += 40 * sh["blink"] + 60 * sh["glance_left"] - 60 * sh["glance_right"]
            data[:, self.f8] += 40 * sh["blink"] - 60 * sh["glance_left"] + 60 * sh["glance_right"]

            self.outlet.push_chunk(data.astype(np.float32).tolist())
            n += chunk
            delay = start + n / RATE - local_clock()
            if delay > 0:
                time.sleep(delay)
