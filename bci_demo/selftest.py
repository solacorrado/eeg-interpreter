"""Check this laptop is ready, before going to the headset station (about 2 minutes, no headset).

  uv run python bci_demo/selftest.py

It checks the libraries, the sample files and the data folder, then runs a short
calibration and live test against the practice headset in a throwaway folder (your real
profiles are not touched). Ends with PASS or FAIL.
"""
import importlib
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent
ok = True


def check(label, cond, hint=""):
    global ok
    print(f"  {'OK  ' if cond else 'FAIL'} {label}" + (f"  -> {hint}" if not cond and hint else ""))
    ok &= bool(cond)


print("1. Libraries")
for mod in ("numpy", "scipy", "pylsl", "pypdfium2", "PySide6.QtMultimedia"):
    try:
        importlib.import_module(mod)
        check(mod, True)
    except Exception as e:
        check(mod, False, f"{e} (run: uv sync)")

print("2. Sample files")
for f in ("media/sample.pdf", "static/media/sample_audio.wav", "static/media/sample_video.mp4"):
    check(f, (HERE / f).exists(), "run: uv run python bci_demo/make_media.py")

print("3. Real data folder")
from storage import data_dir  # noqa: E402
real = data_dir()
probe = real / ".write_test"
try:
    probe.write_text("x")
    probe.unlink()
    check(f"can save to {real}", True)
except OSError as e:
    check(f"can save to {real}", False, str(e))

print("4. Practice run (calibration + live test, ~2 min)")
os.environ["HFV_DATA_DIR"] = tempfile.mkdtemp(prefix="hfv_selftest_")
from pylsl import local_clock  # noqa: E402

import calibration as C  # noqa: E402
from fake_eeg import NAME, FakeHeadset  # noqa: E402
from live_engine import LiveEngine  # noqa: E402
from storage import Profile, TrialRecorder  # noqa: E402

inputs = ["jaw", "blink", "glance"]
fake = FakeHeadset()
fake.begin()
prof = Profile("selftest")
eng = LiveEngine(lambda m: None, prof.settings)
eng.connect(NAME, fake=True, source_id=fake.source_id)
t0 = time.time()
while not eng.status.get("connected") and time.time() - t0 < 15:
    time.sleep(0.2)
check("practice headset streams over LSL", eng.status.get("connected"), "LSL blocked? check the firewall")
if not eng.status.get("connected"):
    sys.exit("FAIL")
eng.set_freqs(C.SSVEP_FREQS)
time.sleep(4)


def run(kind, steps):
    rec = TrialRecorder(kind, prof.name, eng.info)
    eng.set_recorder(rec)
    for st in steps:
        t = local_clock()
        for spec in st.fake:
            fake.do(spec[0], spec[1], delay=spec[2])
        time.sleep(st.dur)
        rec.add_cue(t, local_clock(), st.input, st.phase, st.cls, st.text, expect=list(st.expect))
    time.sleep(1.5)
    eng.set_recorder(None)
    return rec


rec = run("calibration", C.build_calibration(inputs, seed=7))
rows, events, cues = rec.snapshot()
results, update = C.analyze_calibration(rows, cues)
folder = rec.save(results)
prof.record("calibration", results, folder, update)
prof.save()
for r in results:
    check(f"calibration {r['item']}: {r['accuracy']:.0%} {r['verdict']}", r["verdict"] != "Poor")
eng.apply_settings(prof.settings)
rec = run("test", C.build_test(inputs, reps=1, seed=7))
rows, events, cues = rec.snapshot()
results, summary = C.score_test(events, cues)
rec.save(results, {"summary": summary})
hits = sum(r["accuracy"] for r in results) / max(len(results), 1)
check(f"live test detected {hits:.0%} of cues, {summary['false_alarms']} false alarms", hits >= 0.8)
check("trial files saved", all((folder / f).exists() for f in ("eeg.npz", "features.csv", "cues.csv", "results.json")))
eng.stop()

print("\nPASS — this laptop is ready." if ok else "\nFAIL — fix the items marked FAIL above.")
sys.exit(0 if ok else 1)
