"""Hands-Free Viewer, desktop version (a normal window, no browser).

Same controls as the web version:
  PDF    look at ▲ ▶ ▼ ◀ (12/13/14/15 Hz) to scroll; double clench = arrows on/off;
         hold a clench ~1 s = zoom in/out
  Audio  double clench = play/pause; single clench = back 15 s
  Video  look at ⏪15 (12 Hz) / 15⏩ (15 Hz) to skip; clench = play/pause
Keyboard backup: arrow keys; J = jaw (tap, tap twice, hold 1 s).
Fake-headset mode: hold Shift as well to send fake brain signals through the detectors.

Run (from the ant-hackathon folder):
  uv run python bci_demo/desktop_app.py --fake          # fake headset
  uv run python bci_demo/desktop_app.py                 # real headset (first EEG stream on LSL)
  uv run python bci_demo/desktop_app.py --stream NAME   # pick the stream by name
"""
import argparse
import math
import sys
import time
from pathlib import Path

import pypdfium2 as pdfium
from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QRadialGradient
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (QApplication, QButtonGroup, QComboBox, QGridLayout, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QPushButton, QSlider, QStackedWidget, QVBoxLayout,
                               QWidget)

from engine import EEGEngine

HERE = Path(__file__).parent
FREQ = {"up": 12, "right": 13, "down": 14, "left": 15, "back": 12, "fwd": 15}
GLYPH = {"up": "▲", "right": "▶", "down": "▼", "left": "◀", "back": "◀◀ 15", "fwd": "15 ▶▶"}
TABS = ["pdf", "audio", "video"]
TAB_ACTS = {"pdf": ["up", "right", "down", "left"], "audio": [], "video": ["back", "fwd"]}
PDF_KEYS = {Qt.Key_Up: "up", Qt.Key_Right: "right", Qt.Key_Down: "down", Qt.Key_Left: "left"}
VIDEO_KEYS = {Qt.Key_Left: "back", Qt.Key_Right: "fwd"}
DOUBLE_MS = 1500   # two clenches closer than this = a double clench
HOLD_MS = 1000     # J held this long = a held clench
ZOOM_IN = 2.5

BG, PANEL, LINE, TEXT, MUTED = "#0d0f12", "#171a20", "#2a2f38", "#e8eaed", "#8b93a1"
ACCENT, WARN, ARROW = "#4cc38a", "#f2b84b", "#ff3b30"
# Gaussian falloff (radius fraction -> brightness): soft edges, no hard flashing border
GLOW_STOPS = [(0.0, 1.0), (0.25, 0.82), (0.5, 0.46), (0.75, 0.17), (1.0, 0.0)]


class Settings:
    strength = 0.6   # 0.1-1.0: how far brightness swings (1.0 = full black to white)
    style = "glow"   # "glow" or "box"


class Bridge(QObject):
    """Carries engine events from the EEG thread to the window (Qt queues them safely)."""
    event = Signal(dict)


# ---------------------------------------------------------------- flicker target
class FlickerTarget(QWidget):
    clicked = Signal(str)

    def __init__(self, act):
        super().__init__()
        self.act, self.freq = act, FREQ[act]
        self.off = False
        self.hit_until = 0.0
        self.setMinimumSize(90, 90)
        self.setCursor(Qt.PointingHandCursor)

    def flash(self):
        self.hit_until = time.perf_counter() + 0.4

    def mousePressEvent(self, e):
        self.clicked.emit(self.act)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        now = time.perf_counter()
        # smooth sine brightness: works on any monitor refresh rate
        level = 0.5 + 0.5 * Settings.strength * math.sin(2 * math.pi * self.freq * now)
        bg = QColor("#0a0b0e")
        if not self.off and Settings.style == "box":
            v = int(255 * level)
            bg = QColor(v, v, v)
        p.setPen(Qt.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(r, 10, 10)
        if not self.off and Settings.style == "glow":
            radius = min(r.width(), r.height()) * 0.475
            g = QRadialGradient(r.center(), radius)
            for at, a in GLOW_STOPS:
                g.setColorAt(at, QColor(255, 255, 255, int(255 * a * level)))
            p.setBrush(g)
            p.drawEllipse(r.center(), radius, radius)
        if now < self.hit_until:
            p.setPen(QPen(QColor(ACCENT), 3))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r, 10, 10)
        arrow = QColor(ARROW)
        arrow.setAlphaF(0.3 if self.off else 1.0)
        p.setPen(arrow)
        p.setFont(QFont("Segoe UI", 30 if len(GLYPH[self.act]) == 1 else 20, QFont.Bold))
        p.drawText(r, Qt.AlignCenter, GLYPH[self.act])
        p.setPen(QColor(MUTED))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(r.adjusted(0, 0, 0, -6), Qt.AlignHCenter | Qt.AlignBottom, f"{self.freq} Hz")


# ---------------------------------------------------------------- PDF view
class PdfView(QWidget):
    def __init__(self, path):
        super().__init__()
        self.doc = pdfium.PdfDocument(str(path))
        self.pages = len(self.doc)
        self.cache = {}
        self.page, self.zoom, self.x, self.y = 0, 1.0, 0.0, 0.0
        self.shown = [0.0, 0.0, 0.0]  # animated x, y, scale
        self.arrows_on = True
        self.setMinimumSize(200, 200)

    def image(self, i):
        if i not in self.cache:
            pil = self.doc[i].render(scale=2).to_pil().convert("RGB")
            self.cache[i] = QImage(pil.tobytes(), pil.width, pil.height, 3 * pil.width, QImage.Format_RGB888).copy()
        return self.cache[i]

    def scale(self):
        return self.width() / self.image(self.page).width() * self.zoom

    def limits(self):
        img, s = self.image(self.page), self.scale()
        return max(0.0, img.width() * s - self.width()), max(0.0, img.height() * s - self.height())

    def clamp(self, snap=False):
        mx, my = self.limits()
        self.x, self.y = min(max(self.x, 0), mx), min(max(self.y, 0), my)
        if snap:
            self.shown = [self.x, self.y, self.scale()]
        self.update()

    def go_page(self, i, at_bottom=False):
        self.page = max(0, min(self.pages - 1, i))
        self.x, self.y = 0.0, (1e9 if at_bottom else 0.0)
        self.clamp(snap=True)

    def move(self, d):
        mx, my = self.limits()
        sx, sy = self.width() * 0.4, self.height() * 0.5
        if d == "down":
            if self.y >= my - 1:
                return self.go_page(self.page + 1)
            self.y += sy
        elif d == "up":
            if self.y <= 1:
                return self.page > 0 and self.go_page(self.page - 1, at_bottom=True)
            self.y -= sy
        elif d == "right":
            if self.x >= mx - 1:
                return self.go_page(self.page + 1)
            self.x += sx
        elif d == "left":
            if self.x <= 1:
                return self.page > 0 and self.go_page(self.page - 1)
            self.x -= sx
        self.clamp()

    def toggle_zoom(self):
        before = self.scale()
        cx, cy = (self.x + self.width() / 2) / before, (self.y + self.height() / 2) / before
        self.zoom = 1.0 if self.zoom != 1.0 else ZOOM_IN
        after = self.scale()
        self.x, self.y = cx * after - self.width() / 2, cy * after - self.height() / 2
        self.clamp()
        return self.zoom != 1.0

    def tick(self):
        """Ease the shown position toward the target (~0.25 s glide)."""
        target = [self.x, self.y, self.scale()]
        if any(abs(a - b) > 0.5 for a, b in zip(self.shown[:2], target[:2])) or abs(self.shown[2] - target[2]) > 1e-3:
            self.shown = [a + (b - a) * 0.25 for a, b in zip(self.shown, target)]
            self.update()

    def resizeEvent(self, e):
        self.clamp(snap=True)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), Qt.white)
        x, y, s = self.shown
        if s <= 0:
            self.shown = [self.x, self.y, self.scale()]
            x, y, s = self.shown
        img = self.image(self.page)
        p.drawImage(QRectF(-x, -y, img.width() * s, img.height() * s), img)
        info = (f"page {self.page + 1} / {self.pages} · {'fit' if self.zoom == 1 else f'{self.zoom}×'}"
                f" · arrows {'on' if self.arrows_on else 'off'}")
        p.setFont(QFont("Segoe UI", 9))
        w = p.fontMetrics().horizontalAdvance(info) + 16
        box = QRectF(self.width() - w - 8, self.height() - 30, w, 22)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 170))
        p.drawRoundedRect(box, 4, 4)
        p.setPen(Qt.white)
        p.drawText(box, Qt.AlignCenter, info)


# ---------------------------------------------------------------- main window
class Viewer(QMainWindow):
    def __init__(self, args, fake):
        super().__init__()
        self.fake = fake
        self.tab = "pdf"
        self.last_clench = 0.0
        self.j_down = self.j_held = self.j_via_eeg = False
        self.look_key = None
        self.setWindowTitle("Hands-Free Viewer")
        self.resize(1200, 820)
        self.setStyleSheet(f"""
            QWidget {{ background:{BG}; color:{TEXT}; font: 10pt 'Segoe UI'; }}
            #bar {{ background:{PANEL}; border-bottom:1px solid {LINE}; }}
            #bar QLabel, #bar QSlider {{ background:transparent; }}
            QPushButton {{ background:transparent; border:1px solid {LINE}; color:{MUTED}; padding:5px 14px; border-radius:6px; }}
            QPushButton:checked {{ color:{TEXT}; border-color:{ACCENT}; }}
            QComboBox {{ background:{BG}; border:1px solid {LINE}; border-radius:5px; padding:2px 6px; }}
            QSlider::groove:horizontal {{ height:4px; background:{LINE}; border-radius:2px; }}
            QSlider::handle:horizontal {{ background:{ACCENT}; width:14px; margin:-5px 0; border-radius:7px; }}
            #toast {{ background:{PANEL}; border:1px solid {ACCENT}; border-radius:8px; padding:8px 18px; font-size:12pt; }}
            #big {{ font-size:90pt; }}
            #hint {{ color:{MUTED}; }}
        """)

        # --- top bar
        bar = QWidget(objectName="bar")
        top = QHBoxLayout(bar)
        top.setContentsMargins(14, 8, 14, 8)
        title = QLabel("<b>Hands-Free Viewer</b>")
        title.setStyleSheet("background:transparent; font-size:12pt;")
        top.addWidget(title)
        self.tab_buttons = QButtonGroup(self)
        for i, name in enumerate(TABS):
            b = QPushButton(name.upper() if name == "pdf" else name.capitalize(), checkable=True)
            b.setFocusPolicy(Qt.NoFocus)
            self.tab_buttons.addButton(b, i)
            top.addWidget(b)
        self.tab_buttons.button(0).setChecked(True)
        self.tab_buttons.idClicked.connect(lambda i: self.set_tab(TABS[i]))
        top.addSpacing(16)
        top.addWidget(QLabel("Flicker"))
        self.slider = QSlider(Qt.Horizontal, minimum=10, maximum=100, value=60, singleStep=5, pageStep=10)
        self.slider.setFixedWidth(120)
        self.slider.setFocusPolicy(Qt.NoFocus)
        self.slider.setToolTip("Lower = gentler on the eyes, but a weaker brain signal")
        self.strength_label = QLabel("60%")
        self.slider.valueChanged.connect(self.set_strength)
        top.addWidget(self.slider)
        top.addWidget(self.strength_label)
        top.addSpacing(10)
        top.addWidget(QLabel("Style"))
        self.style_box = QComboBox()
        self.style_box.addItems(["Soft glow", "Box"])
        self.style_box.setFocusPolicy(Qt.NoFocus)
        self.style_box.currentIndexChanged.connect(lambda i: setattr(Settings, "style", ["glow", "box"][i]))
        top.addWidget(self.style_box)
        top.addStretch(1)
        self.status = QLabel("connecting…")
        self.status.setStyleSheet(f"color:{MUTED}; background:transparent;")
        top.addWidget(self.status)

        # --- pages
        self.targets = {a: FlickerTarget(a) for a in FREQ}
        for t in self.targets.values():
            t.clicked.connect(self.act)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.build_pdf(args.pdf))
        self.stack.addWidget(self.build_audio())
        self.stack.addWidget(self.build_video())

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(bar)
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(root)

        self.toast_label = QLabel(root, objectName="toast")
        self.toast_label.hide()
        self.toast_timer = QTimer(self, singleShot=True, timeout=self.toast_label.hide)
        self.single_timer = QTimer(self, singleShot=True, timeout=self.audio_back)
        self.hold_timer = QTimer(self, singleShot=True, timeout=self.j_hold_fired)

        # --- EEG engine (runs in a background thread, talks to us through the bridge)
        self.bridge = Bridge()
        self.bridge.event.connect(self.on_event)
        self.engine = EEGEngine(self.safe_emit, stream=args.stream, fake=fake is not None,
                                ssvep_threshold=args.ssvep_threshold, clench_factor=args.clench_factor)
        self.engine.start()

        # --- repaint once per screen refresh (more would only steal time from the EEG thread)
        hz = (self.screen().refreshRate() if self.screen() else 60) or 60
        self.frame = QTimer(self, interval=max(4, int(1000 / hz)), timerType=Qt.PreciseTimer, timeout=self.on_frame)
        self.frame.start()
        QApplication.instance().installEventFilter(self)
        self.set_tab("pdf")

    # ---------- page builders
    def build_pdf(self, pdf_path):
        w = QWidget()
        g = QGridLayout(w)
        g.setContentsMargins(12, 12, 12, 12)
        g.setSpacing(10)
        self.pdf = PdfView(pdf_path)
        g.addWidget(self.targets["up"], 0, 1)
        g.addWidget(self.targets["left"], 1, 0)
        g.addWidget(self.pdf, 1, 1)
        g.addWidget(self.targets["right"], 1, 2)
        g.addWidget(self.targets["down"], 2, 1)
        g.setRowMinimumHeight(0, 110)
        g.setRowMinimumHeight(2, 110)
        g.setColumnMinimumWidth(0, 110)
        g.setColumnMinimumWidth(2, 110)
        g.setRowStretch(1, 1)
        g.setColumnStretch(1, 1)
        return w

    def build_audio(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addStretch(1)
        self.a_icon = QLabel("▶", objectName="big", alignment=Qt.AlignCenter)
        self.a_time = QLabel("0:00", objectName="hint", alignment=Qt.AlignCenter)
        self.a_bar = QSlider(Qt.Horizontal, enabled=False)
        self.a_bar.setFixedWidth(500)
        hint = QLabel("Clench twice quickly to play / pause · clench once to go back 15 s",
                      objectName="hint", alignment=Qt.AlignCenter)
        v.addWidget(self.a_icon)
        v.addWidget(self.a_bar, alignment=Qt.AlignCenter)
        v.addWidget(self.a_time)
        v.addWidget(hint)
        v.addStretch(1)
        self.audio = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.audio.setAudioOutput(self.audio_out)
        self.audio.setSource(QUrl.fromLocalFile(str(HERE / "static" / "media" / "sample_audio.wav")))
        self.audio.setLoops(QMediaPlayer.Infinite)
        self.audio.positionChanged.connect(self.audio_progress)
        self.audio.playbackStateChanged.connect(
            lambda s: self.a_icon.setText("❚❚" if s == QMediaPlayer.PlayingState else "▶"))
        return w

    def build_video(self):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(12, 12, 12, 12)
        h.setSpacing(12)
        self.targets["back"].setFixedWidth(150)
        self.targets["fwd"].setFixedWidth(150)
        vw = QVideoWidget()
        vw.setFocusPolicy(Qt.NoFocus)
        self.video = QMediaPlayer(self)
        self.video_out = QAudioOutput(self)
        self.video.setAudioOutput(self.video_out)
        self.video.setVideoOutput(vw)
        self.video.setSource(QUrl.fromLocalFile(str(HERE / "static" / "media" / "sample_video.mp4")))
        h.addWidget(self.targets["back"])
        h.addWidget(vw, 1)
        h.addWidget(self.targets["fwd"])
        return w

    # ---------- helpers
    def safe_emit(self, msg):
        try:
            self.bridge.event.emit(msg)
        except RuntimeError:  # window already closed; the EEG thread is shutting down
            pass

    def toast(self, msg):
        self.toast_label.setText(msg)
        self.toast_label.adjustSize()
        c = self.centralWidget()
        self.toast_label.move((c.width() - self.toast_label.width()) // 2, c.height() - self.toast_label.height() - 24)
        self.toast_label.show()
        self.toast_label.raise_()
        self.toast_timer.start(1300)

    def set_strength(self, v):
        Settings.strength = v / 100
        self.strength_label.setText(f"{v}%")

    def send_targets(self):
        acts = [] if (self.tab == "pdf" and not self.pdf.arrows_on) else TAB_ACTS[self.tab]
        self.engine.set_freqs([FREQ[a] for a in acts])

    def set_tab(self, name):
        self.tab = name
        self.single_timer.stop()
        self.last_clench = 0.0
        self.tab_buttons.button(TABS.index(name)).setChecked(True)
        self.stack.setCurrentIndex(TABS.index(name))
        if name != "audio":
            self.audio.pause()
        if name != "video":
            self.video.pause()
        self.send_targets()

    def on_frame(self):
        for a in TAB_ACTS[self.tab]:
            self.targets[a].update()
        if self.tab == "pdf":
            self.pdf.tick()

    def audio_progress(self, pos):
        dur = max(1, self.audio.duration())
        self.a_bar.setMaximum(dur)
        self.a_bar.setValue(pos)
        fmt = lambda ms: f"{ms // 60000}:{ms // 1000 % 60:02d}"
        self.a_time.setText(f"{fmt(pos)} / {fmt(dur)}")

    @staticmethod
    def toggle(player):
        if player.playbackState() == QMediaPlayer.PlayingState:
            player.pause()
            return "pause"
        player.play()
        return "play"

    def skip(self, player, ms):
        player.setPosition(max(0, min(player.duration() - 100, player.position() + ms)))

    def audio_back(self):
        self.skip(self.audio, -15000)
        self.toast("Back 15 s")

    # ---------- actions and gestures
    def act(self, name):
        self.targets[name].flash()
        if name in ("up", "down", "left", "right"):
            self.pdf.move(name)
        elif name == "back":
            self.skip(self.video, -15000)
            self.toast("Back 15 s")
        elif name == "fwd":
            self.skip(self.video, 15000)
            self.toast("Forward 15 s")

    def set_arrows(self, on):
        self.pdf.arrows_on = on
        for a in TAB_ACTS["pdf"]:
            self.targets[a].off = not on
            self.targets[a].update()
        self.pdf.update()
        self.send_targets()
        self.toast("Double clench → arrows ON" if on else "Double clench → arrows OFF (read in peace)")

    def clench(self):
        now = time.monotonic() * 1000
        is_double = now - self.last_clench < DOUBLE_MS
        self.last_clench = 0.0 if is_double else now
        if self.tab == "pdf":
            if is_double:
                self.set_arrows(not self.pdf.arrows_on)
            else:
                self.toast("Clench again to turn arrows " + ("off" if self.pdf.arrows_on else "on"))
        elif self.tab == "audio":
            self.single_timer.stop()
            if is_double:
                self.toast(f"Double clench → {self.toggle(self.audio)} audio")
            else:
                self.toast("Back 15 s… (clench again now to play/pause instead)")
                self.single_timer.start(DOUBLE_MS)
        else:
            self.toast(f"Clench → {self.toggle(self.video)} video")

    def hold(self):
        self.last_clench = 0.0
        if self.tab == "pdf":
            self.toast("Held clench → zoom in" if self.pdf.toggle_zoom() else "Held clench → zoom out")
        else:
            self.clench()

    # ---------- events from the EEG engine
    def on_event(self, m):
        t = m["type"]
        if t == "clench":
            self.clench()
        elif t == "hold":
            self.hold()
        elif t == "ssvep":
            for a in TAB_ACTS[self.tab]:
                if FREQ[a] == m["freq"]:
                    self.act(a)
                    self.toast(f"Looked at {m['freq']:g} Hz")
        elif t == "status":
            if not m.get("connected"):
                self.status.setText(f"<span style='color:#c44'>●</span> {m.get('msg', 'no EEG stream')}")
                return
            scores = "  ".join(f"{float(f):g} Hz {v:.2f}" for f, v in m.get("scores", {}).items())
            jaw_color = WARN if m.get("clenching") else TEXT
            calib = " · calibrating, stay still" if m.get("calibrating") else ""
            self.status.setText(
                f"<span style='color:{ACCENT}'>●</span> {m['stream']}{' (fake)' if m.get('fake') else ''}{calib}"
                f"   <span style='color:{MUTED}'>{scores}</span>"
                f"   jaw <b style='color:{jaw_color}'>{m.get('emg', '–')}×</b>")

    # ---------- keyboard (demo backup), caught for the whole app
    def eventFilter(self, obj, e):
        if e.type() not in (QEvent.KeyPress, QEvent.KeyRelease):
            return False
        press = e.type() == QEvent.KeyPress
        key, repeat = e.key(), e.isAutoRepeat()
        via_eeg = self.fake is not None and bool(e.modifiers() & Qt.ShiftModifier)
        keymap = PDF_KEYS if self.tab == "pdf" else VIDEO_KEYS if self.tab == "video" else {}
        if key == Qt.Key_J:
            if repeat:
                return True
            if press and not self.j_down:
                self.j_down, self.j_held, self.j_via_eeg = True, False, via_eeg
                if via_eeg:
                    self.fake.clench("on")
                else:
                    self.hold_timer.start(HOLD_MS)
            elif not press and self.j_down:
                self.j_down = False
                if self.j_via_eeg:
                    self.fake.clench("off")
                else:
                    self.hold_timer.stop()
                    if not self.j_held:
                        self.clench()
            return True
        if key in keymap:
            if press:
                if via_eeg:
                    if not repeat:
                        self.look_key = key
                        self.fake.look(FREQ[keymap[key]])
                else:
                    self.act(keymap[key])
            elif not repeat and key == self.look_key:
                self.look_key = None
                self.fake.look(None)
            return True
        if not press and not repeat and key == self.look_key:
            self.look_key = None
            self.fake.look(None)
        return False

    def j_hold_fired(self):
        if self.j_down:
            self.j_held = True
            self.hold()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="use a fake headset (keyboard controlled)")
    ap.add_argument("--stream", help="LSL stream name (default: first stream of type EEG)")
    ap.add_argument("--pdf", default=str(HERE / "media" / "sample.pdf"))
    ap.add_argument("--ssvep-threshold", type=float, default=0.42)
    ap.add_argument("--clench-factor", type=float, default=5.0)
    ap.add_argument("--no-warning", action="store_true", help="skip the flashing warning (for testing)")
    args = ap.parse_args()
    if not Path(args.pdf).exists():
        raise SystemExit(f"No PDF at {args.pdf}. Run: uv run python bci_demo/make_media.py  (or pass --pdf FILE)")

    app = QApplication(sys.argv)
    if not args.no_warning:
        ok = QMessageBox.warning(
            None, "Hands-Free Viewer",
            "Flashing warning: the arrows flicker 12–15 times a second. Don't use this if you or anyone "
            "watching has photosensitive epilepsy.\n\n"
            "Stare at a flickering arrow to press it. Clench your jaw to act (PDF: twice = arrows on/off, "
            "hold = zoom; Audio: twice = play/pause, once = back 15 s; Video: play/pause).\n\n"
            "Keyboard backup: arrow keys, and J for the jaw (tap / tap twice / hold).",
            QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok)
        if ok != QMessageBox.Ok:
            return

    fake = None
    if args.fake:
        from fake_eeg import NAME, FakeHeadset
        fake = FakeHeadset()
        fake.start()
        args.stream = NAME
        print("Fake headset running. Hold Shift+arrow to fake a 'look', Shift+J to fake a 'clench'.")

    win = Viewer(args, fake)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
