"""Profiles (per-person settings) and trial recordings, saved to disk.

Everything lives in Documents/HandsFreeViewer (or the folder in the HFV_DATA_DIR
environment variable), outside the code folder, so updating the app or re-cloning
the repo never touches it:

  profiles/<name>.json      one per wearer: settings, latest calibration results,
                            and the full history of every calibration
  profiles/<name>.json.bak  the previous version of that file
  trials/<name>/<date>_<kind>/
      eeg.npz               raw EEG (samples x channels), timestamps, channel names, rate
      features.csv          the signal features, 25 rows a second
      cues.csv              what the wearer was asked to do, and when
      events.csv            what the app detected, and when
      results.json          scores for this trial
  trials/index.csv          one row per input per trial: the "what works" log

Profiles carry a schema version. Older files are upgraded when loaded, unknown
keys are kept, and missing settings get defaults, so profiles survive new versions.
"""
import copy
import csv
import json
import os
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from version import APP_VERSION

SCHEMA_VERSION = 1

DEFAULT_SETTINGS = {
    "ssvep_thr": {"12": 0.42, "13": 0.42, "14": 0.42, "15": 0.42},
    "ssvep_margin": 0.08,
    "jaw_factor": 5.0,        # jaw muscle power must be this many times resting
    "hold_s": 1.0,            # a clench this long counts as "held"
    "brow_factor": 5.0,       # forehead muscle power, times resting
    "blink_z": 6.0,           # blink size, in units of resting noise
    "blink_sign": 1,
    "glance_left_z": 4.0,     # left/right eye movement size, in units of resting noise
    "glance_right_z": 4.0,
    "heog_sign": 1,           # +1 if looking left makes F7-F8 go up
    "alpha_factor": 2.0,      # eyes-closed alpha power, times eyes-open
    "fist_right_ratio": 0.6,  # C3 rhythm must drop below this fraction of resting
    "fist_left_ratio": 0.6,   # C4 likewise
    "flicker_strength": 0.6,
    "flicker_style": "glow",
    # viewer arrow rates: the best, well-spaced rates on the first real recording (wearer1, 92-100%)
    "viewer_rates": {"up": 13, "right": 9, "down": 15, "left": 11, "back": 9, "fwd": 13},
    "last_files": {},         # last PDF / audio / video opened, per kind
}

_KNOWN = {"schema_version", "name", "created", "updated", "app_version", "settings", "calibration", "history"}


def data_dir():
    env = os.environ.get("HFV_DATA_DIR")
    if env:
        d = Path(env)
    else:
        try:
            from PySide6.QtCore import QStandardPaths
            d = Path(QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)) / "HandsFreeViewer"
        except Exception:
            d = Path.home() / "Documents" / "HandsFreeViewer"
    d.mkdir(parents=True, exist_ok=True)
    return d


def slug(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_") or "profile"


def merge(defaults, saved):
    """Deep-merge saved values over defaults; keeps keys the defaults don't know."""
    out = copy.deepcopy(defaults)
    for k, v in (saved or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _migrate(d):
    """Upgrade an older profile dict in place. (Version 1 is the first format.)"""
    d.setdefault("schema_version", 1)
    return d


class Profile:
    def __init__(self, name, settings=None, calibration=None, history=None, created=None, updated=None, extra=None):
        self.name = name
        self.settings = merge(DEFAULT_SETTINGS, settings)
        self.calibration = calibration or {}
        self.history = history or []
        self.created = created or _now()
        self.updated = updated or self.created
        self.extra = extra or {}

    @staticmethod
    def folder():
        d = data_dir() / "profiles"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def path(self):
        return self.folder() / f"{slug(self.name)}.json"

    def to_dict(self):
        return {**self.extra, "schema_version": SCHEMA_VERSION, "name": self.name, "created": self.created,
                "updated": self.updated, "app_version": APP_VERSION, "settings": self.settings,
                "calibration": self.calibration, "history": self.history}

    @classmethod
    def from_dict(cls, d):
        d = _migrate(dict(d))
        extra = {k: v for k, v in d.items() if k not in _KNOWN}
        return cls(d.get("name", "profile"), d.get("settings"), d.get("calibration"), d.get("history"),
                   d.get("created"), d.get("updated"), extra)

    @classmethod
    def load(cls, name):
        p = cls.folder() / f"{slug(name)}.json"
        for candidate in (p, p.with_suffix(".json.bak")):
            if candidate.exists():
                try:
                    return cls.from_dict(json.loads(candidate.read_text(encoding="utf8")))
                except (OSError, ValueError):
                    continue
        return cls(name)

    def save(self):
        self.updated = _now()
        p = self.path
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf8")
        if p.exists():
            shutil.copy2(p, p.with_suffix(".json.bak"))
        os.replace(tmp, p)

    def record(self, kind, results, trial_folder, settings_update=None):
        """Store a calibration or test result. Old results stay in the history."""
        stamp = _now()
        for r in results:
            self.calibration[r["key"]] = {**r, "kind": kind, "date": stamp, "app_version": APP_VERSION,
                                          "trial": str(trial_folder)}
        self.history.append({"kind": kind, "date": stamp, "app_version": APP_VERSION, "trial": str(trial_folder),
                             "results": results, "settings_update": settings_update or {}})
        if settings_update:
            self.settings = merge(self.settings, settings_update)


def list_profiles():
    names = []
    for f in sorted(Profile.folder().glob("*.json")):
        try:
            names.append(json.loads(f.read_text(encoding="utf8")).get("name", f.stem))
        except (OSError, ValueError):
            pass
    return names


def load_state():
    p = data_dir() / "state.json"
    try:
        return json.loads(p.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return {}


def save_state(**kw):
    state = {**load_state(), **kw}
    (data_dir() / "state.json").write_text(json.dumps(state, indent=2), encoding="utf8")


class TrialRecorder:
    """Collects everything during one calibration or test; save() writes it to disk.
    The headset thread adds data; the screen adds cues. A lock keeps them apart."""

    def __init__(self, kind, profile_name, stream_info):
        self.kind = kind
        self.profile_name = profile_name
        self.stream_info = dict(stream_info or {})
        self.started = datetime.now()
        self.lock = threading.Lock()
        self.raw, self.ts, self.rows, self.events, self.cues = [], [], [], [], []

    def add_raw(self, arr, ts):
        with self.lock:
            self.raw.append(np.asarray(arr, np.float32))
            self.ts.append(np.asarray(ts, np.float64))

    def add_row(self, row):
        with self.lock:
            self.rows.append(dict(row))

    def add_event(self, ev):
        with self.lock:
            self.events.append(dict(ev))

    def add_cue(self, t_start, t_end, input_, phase, cls, text, **extra):
        with self.lock:
            self.cues.append({"t_start": t_start, "t_end": t_end, "input": input_, "phase": phase,
                              "cls": cls, "text": text, **extra})

    def snapshot(self):
        with self.lock:
            return list(self.rows), list(self.events), list(self.cues)

    def save(self, results, extra=None):
        with self.lock:
            raw = np.concatenate(self.raw) if self.raw else np.zeros((0, 0), np.float32)
            ts = np.concatenate(self.ts) if self.ts else np.zeros(0)
            rows, events, cues = list(self.rows), list(self.events), list(self.cues)
        folder = data_dir() / "trials" / slug(self.profile_name) / f"{self.started:%Y%m%d-%H%M%S}_{self.kind}"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "eeg.npz", data=raw, timestamps=ts,
                            channels=np.array(self.stream_info.get("channels", []), dtype=str),
                            rate=np.array(self.stream_info.get("rate", 0.0)))
        _write_csv(folder / "features.csv", rows)
        _write_csv(folder / "cues.csv", cues)
        _write_csv(folder / "events.csv", events)
        duration = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
        meta = {"kind": self.kind, "profile": self.profile_name, "started": self.started.isoformat(timespec="seconds"),
                "app_version": APP_VERSION, "stream": self.stream_info.get("name"), "rate": self.stream_info.get("rate"),
                "channels": self.stream_info.get("channels"), "duration_s": round(duration, 1),
                "samples": int(len(ts)), **(extra or {})}
        (folder / "results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2, default=float),
                                             encoding="utf8")
        _append_index(folder, meta, results)
        return folder


def _write_csv(path, rows):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=cols or ["empty"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})


INDEX_COLS = ["date", "profile", "kind", "input", "item", "accuracy", "verdict", "threshold", "detail",
              "app_version", "stream", "trial_folder"]


def _append_index(folder, meta, results):
    path = data_dir() / "trials" / "index.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLS)
        if new:
            w.writeheader()
        for r in results:
            w.writerow({"date": meta["started"], "profile": meta["profile"], "kind": meta["kind"],
                        "input": r.get("input"), "item": r.get("item"), "accuracy": r.get("accuracy"),
                        "verdict": r.get("verdict"), "threshold": r.get("threshold"), "detail": r.get("detail", ""),
                        "app_version": meta["app_version"], "stream": meta["stream"], "trial_folder": str(folder)})
