"""Hands-Free Viewer — desktop app.

Screens:
  Start           pick or create a profile (one per wearer), connect the headset
                  (a real one over LSL, or the fake one for practice)
  Calibrate & test
                  guided calibration of every input type (flicker, jaw, eyebrows, blink,
                  glance, eyes closed, fists) -> per-person settings saved to the profile;
                  live test with random cues -> hit rate, reaction time, false alarms;
                  live meters of every signal; every session saved as a trial
  Viewer          PDF / audio / video, controlled by flicker + jaw, using the profile

Run (from the ant-hackathon folder):
  uv run python bci_demo/desktop_app.py            # real headset (connect on the start screen)
  uv run python bci_demo/desktop_app.py --fake     # start with the fake headset connected
Data (profiles, trials) is saved in Documents/HandsFreeViewer.
"""
import argparse
import math
import os
import random
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from pylsl import local_clock
from PySide6.QtCore import QEvent, QObject, QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QImage, QPainter, QPen, QPolygonF, QRadialGradient
from PySide6.QtCore import QPointF
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QSoundEffect
from PySide6.QtMultimediaWidgets import QVideoWidget
try:
    from PySide6.QtTextToSpeech import QTextToSpeech
except ImportError:  # speech is optional
    QTextToSpeech = None
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
                               QHBoxLayout,
                               QHeaderView, QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QRadioButton, QSlider, QStackedWidget, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)

import calibration as C
from fake_eeg import NAME as FAKE_NAME, FakeHeadset
from gestures import fkey
from live_engine import LiveEngine, find_streams
from storage import Profile, TrialRecorder, data_dir, list_profiles, load_state, save_state
from version import APP_VERSION

HERE = Path(__file__).parent
BG, PANEL, PANEL2, LINE, TEXT, MUTED = "#0b0f16", "#121923", "#18212d", "#243041", "#eef2f8", "#8a97ab"
ACCENT, WARN, BAD, ARROW = "#34d399", "#fbbf24", "#f87171", "#ff5a52"
ACCENT_SOFT = "#123327"
FONT, FONT_DISPLAY = "Segoe UI Variable Text", "Segoe UI Variable Display"
VERDICT_COLOR = {"Good": ACCENT, "OK": WARN, "Poor": BAD}
GLOW_STOPS = [(0.0, 1.0), (0.25, 0.82), (0.5, 0.46), (0.75, 0.17), (1.0, 0.0)]  # Gaussian falloff
MINT_GRADIENT = "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3ee6a3, stop:1 #22c3b5)"

STYLE = f"""
QWidget {{ background:{BG}; color:{TEXT}; font-family:"{FONT}", "Segoe UI"; font-size:10.5pt; }}
QToolTip {{ background:{PANEL2}; color:{TEXT}; border:1px solid {LINE}; padding:6px 8px; }}
#bar {{ background:{PANEL}; border-bottom:1px solid {LINE}; }}
#bar QLabel, #bar QSlider {{ background:transparent; }}
#panel {{ background:{PANEL}; border:1px solid {LINE}; border-radius:16px; }}
#panel QLabel, #panel QCheckBox, #panel QRadioButton {{ background:transparent; }}
QPushButton {{ background:{PANEL2}; border:1px solid {LINE}; color:{TEXT}; padding:8px 16px; border-radius:10px; }}
QPushButton:hover {{ background:#1f2a39; border-color:#34445c; }}
QPushButton:pressed {{ background:#0f151e; }}
QPushButton:checked {{ background:{ACCENT_SOFT}; border-color:{ACCENT}; color:#d6fbe8; }}
QPushButton:disabled {{ color:#4d5767; background:#10161f; border-color:#1b2330; }}
QPushButton#primary {{ background:{MINT_GRADIENT}; color:#06140f; border:none; font-weight:700; padding:10px 18px; }}
QPushButton#primary:hover {{ background:qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #5af0b4, stop:1 #38d4c6); }}
QPushButton#primary:pressed {{ background:#24b88a; }}
QPushButton#primary:disabled {{ background:#18291f; color:#4f6b60; }}
QComboBox {{ background:{PANEL2}; border:1px solid {LINE}; border-radius:10px; padding:7px 12px; }}
QComboBox:hover {{ border-color:#34445c; }}
QComboBox::drop-down {{ border:none; width:26px; }}
QComboBox QAbstractItemView {{ background:{PANEL2}; border:1px solid {LINE}; selection-background-color:{ACCENT_SOFT};
                               selection-color:{TEXT}; outline:none; padding:4px; }}
QSlider::groove:horizontal {{ height:6px; background:{PANEL2}; border-radius:3px; }}
QSlider::sub-page:horizontal {{ background:{MINT_GRADIENT}; border-radius:3px; }}
QSlider::handle:horizontal {{ background:#eafff5; border:2px solid {ACCENT}; width:14px; margin:-6px 0; border-radius:9px; }}
QProgressBar {{ background:{PANEL2}; border:none; border-radius:4px; max-height:8px; text-align:center; }}
QProgressBar::chunk {{ background:{MINT_GRADIENT}; border-radius:4px; }}
QTabWidget::pane {{ border:1px solid {LINE}; border-radius:14px; background:{PANEL}; top:-1px; }}
QTabBar::tab {{ background:transparent; color:{MUTED}; padding:9px 18px; margin-right:4px; border:none;
                border-bottom:2px solid transparent; font-weight:600; }}
QTabBar::tab:selected {{ color:{TEXT}; border-bottom:2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color:{TEXT}; }}
QTableWidget {{ background:{PANEL}; alternate-background-color:{PANEL2}; gridline-color:transparent; border:none; }}
QHeaderView::section {{ background:{PANEL}; color:{MUTED}; border:none; border-bottom:1px solid {LINE}; padding:6px;
                        font-weight:600; }}
QScrollBar:vertical {{ background:transparent; width:10px; margin:2px; }}
QScrollBar::handle:vertical {{ background:#2b3748; border-radius:4px; min-height:30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height:0; width:0; }}
QCheckBox {{ spacing:10px; }}
QRadioButton {{ spacing:10px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width:16px; height:16px; border:1.5px solid #4a5870; background:{BG}; }}
QCheckBox::indicator {{ border-radius:5px; }}
QRadioButton::indicator {{ border-radius:9px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{ background:{ACCENT}; border-color:{ACCENT}; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color:{ACCENT}; }}
QCheckBox::indicator:disabled {{ border-color:{LINE}; }}
#toast {{ background:rgba(18, 25, 35, 240); border:1px solid {ACCENT}; border-radius:14px; padding:10px 22px;
          font-size:13pt; color:{TEXT}; }}
#big_icon {{ font-size:54pt; color:{ACCENT}; background:{ACCENT_SOFT}; border:2px solid {ACCENT};
             border-radius:90px; min-width:180px; max-width:180px; min-height:180px; max-height:180px; }}
#hint {{ color:{MUTED}; background:transparent; }}
#h1 {{ font-family:"{FONT_DISPLAY}", "Segoe UI"; font-size:26pt; font-weight:700; background:transparent; }}
#h2 {{ font-family:"{FONT_DISPLAY}", "Segoe UI"; font-size:13pt; font-weight:600; background:transparent; }}
#panel #badge, #badge {{ background:{ACCENT_SOFT}; color:{ACCENT}; border-radius:14px; font-weight:700; font-size:11pt;
          min-width:28px; max-width:28px; min-height:28px; max-height:28px; }}
#bar #pill, #pill {{ background:{PANEL2}; border:1px solid {LINE}; border-radius:14px; padding:5px 14px; }}
#card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:20px; }}
#card:hover {{ border:1px solid {ACCENT}; background:#14202c; }}
#card QLabel {{ background:transparent; }}
#cardtitle {{ font-family:"{FONT_DISPLAY}", "Segoe UI"; font-size:17pt; font-weight:700; }}
#cardicon {{ font-size:30pt; }}
#carddesc {{ color:{MUTED}; font-size:11pt; }}
#cardarrow {{ color:{ACCENT}; font-size:20pt; font-weight:700; }}
"""


class Flicker:
    """How the flicker targets look; set from the profile."""
    strength = 0.6
    style = "glow"


def flicker_level(freq, now):
    return 0.5 + 0.5 * Flicker.strength * math.sin(2 * math.pi * freq * now)


def paint_glow(p, center, radius, level):
    g = QRadialGradient(center, radius)
    for at, a in GLOW_STOPS:
        g.setColorAt(at, QColor(255, 255, 255, int(255 * a * level)))
    p.setPen(Qt.NoPen)
    p.setBrush(g)
    p.drawEllipse(center, radius, radius)


def short_stream(name):
    """'EE511-010009-200205_on_DESKTOP-UAPOU9S' -> 'EE511-010009-200205'."""
    return (name or "").split("_on_")[0]


def status_html(m, compact=False):
    if not m or not m.get("connected"):
        msg = (m or {}).get("msg", "Not connected")
        return f"<span style='color:{BAD}'>●</span> {msg}"
    if m.get("stalled"):
        return (f"<span style='color:{BAD}'>● NO DATA for {m['stalled']:.0f} s</span> from {m['stream']} — "
                "check the Wi-Fi / that eego is still streaming (it reconnects by itself)")
    f = m.get("features", {})
    calib = " · <b>calibrating, stay still</b>" if m.get("calibrating") else ""
    if compact:
        jaw = f.get("jaw")
        jaw_txt = (f" &nbsp;·&nbsp; jaw <b style='color:{WARN if m.get('clenching') else TEXT}'>{jaw:.1f}×</b>"
                   if jaw is not None else "")
        return (f"<span style='color:{ACCENT}'>●</span>&nbsp; {short_stream(m['stream'])}"
                f"{' · practice' if m.get('fake') else ''}{calib}{jaw_txt}")
    jaw = f.get("jaw")
    jaw_txt = f"   jaw <b style='color:{WARN if m.get('clenching') else TEXT}'>{jaw:.1f}×</b>" if jaw is not None else ""
    scores = "  ".join(f"{k} Hz {v:.2f}" for k, v in m.get("scores", {}).items())
    return (f"<span style='color:{ACCENT}'>●</span> {short_stream(m['stream'])}{' (practice)' if m.get('fake') else ''}{calib}"
            f"   <span style='color:{MUTED}'>{scores}</span>{jaw_txt}")


class Bridge(QObject):
    """Carries events from background threads to the screen (Qt queues them safely)."""
    event = Signal(dict)
    streams = Signal(list)


# =========================================================================== viewer pieces
GLYPH = {"up": "▲", "right": "▶", "down": "▼", "left": "◀", "back": "◀◀ 15", "fwd": "15 ▶▶"}
# Defaults = the best, well-spaced rates on the first real recording (Sola: 92-100% each).
# Each profile can override them with settings["viewer_rates"].
FREQ = {"up": 13.0, "right": 9.0, "down": 15.0, "left": 11.0, "back": 9.0, "fwd": 13.0}
TABS = ["pdf", "audio", "video"]
TAB_ACTS = {"pdf": ["up", "right", "down", "left"], "audio": [], "video": ["back", "fwd"]}
PDF_KEYS = {Qt.Key_Up: "up", Qt.Key_Right: "right", Qt.Key_Down: "down", Qt.Key_Left: "left"}
VIDEO_KEYS = {Qt.Key_Left: "back", Qt.Key_Right: "fwd"}
DOUBLE_MS, HOLD_MS, ZOOM_IN = 1500, 1000, 2.5


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
        r = QRectF(self.rect()).adjusted(3, 3, -3, -3)
        now = time.perf_counter()
        level = flicker_level(self.freq, now)
        bg = QColor("#0d131c")
        if not self.off and Flicker.style == "box":
            v = int(255 * level)
            bg = QColor(v, v, v)
        p.setPen(QPen(QColor(LINE), 1))
        p.setBrush(bg)
        p.drawRoundedRect(r, 18, 18)
        if not self.off and Flicker.style == "glow":
            paint_glow(p, r.center(), min(r.width(), r.height()) * 0.44, level)
        if now < self.hit_until:  # selected: mint outline and tint
            p.setPen(QPen(QColor(ACCENT), 3))
            p.setBrush(QColor(52, 211, 153, 40))
            p.drawRoundedRect(r, 18, 18)
        arrow = QColor(ARROW)
        arrow.setAlphaF(0.25 if self.off else 1.0)
        p.setPen(arrow)
        p.setFont(QFont(FONT_DISPLAY, 34 if len(GLYPH[self.act]) == 1 else 21, QFont.Bold))
        p.drawText(r, Qt.AlignCenter, GLYPH[self.act])
        # small rate label in a pill at the bottom
        label = "off" if self.off else f"{self.freq:g} Hz"
        p.setFont(QFont(FONT, 8, QFont.DemiBold))
        w = p.fontMetrics().horizontalAdvance(label) + 16
        pill = QRectF(r.center().x() - w / 2, r.bottom() - 26, w, 18)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 18))
        p.drawRoundedRect(pill, 9, 9)
        p.setPen(QColor(MUTED))
        p.drawText(pill, Qt.AlignCenter, label)


class PdfView(QWidget):
    def __init__(self, path):
        super().__init__()
        self.doc = pdfium.PdfDocument(str(path))
        self.pages = len(self.doc)
        self.cache = {}
        self.page, self.zoom, self.x, self.y = 0, 1.0, 0.0, 0.0
        self.shown = [0.0, 0.0, 0.0]
        self.arrows_on = True
        self.M = 16  # margin around the page, so it sits on a "desk" with a shadow
        self.setMinimumSize(200, 200)

    def inner(self):
        return max(1, self.width() - 2 * self.M), max(1, self.height() - 2 * self.M)

    def load(self, path):
        """Show a different PDF (from Open file / drag-and-drop)."""
        doc = pdfium.PdfDocument(str(path))
        if len(doc) == 0:
            raise ValueError("the PDF has no pages")
        self.doc, self.pages, self.cache = doc, len(doc), {}
        self.page, self.zoom, self.x, self.y = 0, 1.0, 0.0, 0.0
        self.clamp(snap=True)

    def image(self, i):
        if i not in self.cache:
            pil = self.doc[i].render(scale=2).to_pil().convert("RGB")
            self.cache[i] = QImage(pil.tobytes(), pil.width, pil.height, 3 * pil.width, QImage.Format_RGB888).copy()
        return self.cache[i]

    def scale(self):
        return self.inner()[0] / self.image(self.page).width() * self.zoom

    def limits(self):
        img, s = self.image(self.page), self.scale()
        iw, ih = self.inner()
        return max(0.0, img.width() * s - iw), max(0.0, img.height() * s - ih)

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
        sx, sy = self.inner()[0] * 0.4, self.inner()[1] * 0.5
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
        iw, ih = self.inner()
        before = self.scale()
        cx, cy = (self.x + iw / 2) / before, (self.y + ih / 2) / before
        self.zoom = 1.0 if self.zoom != 1.0 else ZOOM_IN
        after = self.scale()
        self.x, self.y = cx * after - iw / 2, cy * after - ih / 2
        self.clamp()
        return self.zoom != 1.0

    def tick(self):
        target = [self.x, self.y, self.scale()]
        if any(abs(a - b) > 0.5 for a, b in zip(self.shown[:2], target[:2])) or abs(self.shown[2] - target[2]) > 1e-3:
            self.shown = [a + (b - a) * 0.25 for a, b in zip(self.shown, target)]
            self.update()

    def resizeEvent(self, e):
        self.clamp(snap=True)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#151d29"))
        p.drawRoundedRect(QRectF(self.rect()), 16, 16)
        x, y, s = self.shown
        if s <= 0:
            self.shown = [self.x, self.y, self.scale()]
            x, y, s = self.shown
        img = self.image(self.page)
        M, (iw, ih) = self.M, self.inner()
        page = QRectF(M - x, M - y, img.width() * s, img.height() * s)
        view = QRectF(M, M, iw, ih)
        shown = page.intersected(view)
        for i, a in ((10, 18), (6, 28), (3, 40)):  # soft shadow
            p.setBrush(QColor(0, 0, 0, a))
            p.drawRoundedRect(shown.adjusted(-i, -i + 3, i, i + 3), 10, 10)
        p.save()
        p.setClipRect(view)
        p.drawImage(page, img)
        p.restore()
        info = (f"page {self.page + 1} / {self.pages}  ·  {'fit' if self.zoom == 1 else f'{self.zoom:g}×'}"
                f"  ·  arrows {'on' if self.arrows_on else 'off'}")
        p.setFont(QFont(FONT, 9, QFont.DemiBold))
        w = p.fontMetrics().horizontalAdvance(info) + 24
        box = QRectF(self.width() - w - M - 8, self.height() - M - 36, w, 26)
        p.setBrush(QColor(11, 15, 22, 215))
        p.drawRoundedRect(box, 13, 13)
        p.setPen(QColor(TEXT))
        p.drawText(box, Qt.AlignCenter, info)


def top_bar(title, on_home):
    bar = QWidget(objectName="bar")
    h = QHBoxLayout(bar)
    h.setContentsMargins(14, 8, 14, 8)
    home = QPushButton("⌂  Home")
    home.setFocusPolicy(Qt.NoFocus)
    home.clicked.connect(on_home)
    h.addWidget(home)
    t = QLabel(f"<b>{title}</b>")
    t.setStyleSheet("font-size:12pt;")
    h.addWidget(t)
    return bar, h


# =========================================================================== viewer page
PDF_EXT = {".pdf"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv"}
OPEN_FILTER = ("Documents and media (*.pdf *.mp3 *.wav *.m4a *.ogg *.flac *.aac *.wma *.mp4 *.mov *.mkv *.avi "
               "*.webm *.m4v *.wmv);;PDF (*.pdf);;Audio (*.mp3 *.wav *.m4a *.ogg *.flac *.aac *.wma);;"
               "Video (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.wmv)")
VIEWER_STYLE = f"""
#segment {{ background:{PANEL2}; border:1px solid {LINE}; border-radius:14px; }}
QPushButton#seg {{ background:transparent; border:1px solid transparent; font-size:14pt; padding:8px 22px;
                   border-radius:10px; min-height:40px; color:{MUTED}; }}
QPushButton#seg:hover {{ color:{TEXT}; background:#1f2a39; }}
QPushButton#seg:checked {{ background:{ACCENT_SOFT}; border:1px solid {ACCENT}; color:#d6fbe8; }}
QPushButton#bigctl {{ font-size:15pt; padding:12px 18px; border-radius:16px; min-height:60px;
                      background:{PANEL2}; border:1px solid {LINE}; }}
QPushButton#bigctl:hover {{ border-color:{ACCENT}; background:#1c2734; }}
QPushButton#bigctl:pressed {{ background:{ACCENT_SOFT}; }}
QPushButton#bigopen {{ font-size:14pt; padding:10px 22px; border-radius:12px; min-height:40px;
                       background:{MINT_GRADIENT}; color:#06140f; border:none; font-weight:700; }}
QPushButton#bigopen:hover {{ background:qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #5af0b4, stop:1 #38d4c6); }}
QPushButton#bighome {{ font-size:13pt; padding:10px 18px; border-radius:12px; min-height:40px; }}
QLabel#filename {{ font-size:12.5pt; color:{TEXT}; background:transparent; padding:2px 4px; font-weight:600; }}
"""


def kind_of(path):
    ext = Path(path).suffix.lower()
    return "pdf" if ext in PDF_EXT else "audio" if ext in AUDIO_EXT else "video" if ext in VIDEO_EXT else None


class ViewerPage(QWidget):
    """Big, easy controls. Flicker arrows + jaw clench (what worked best on real data), plus large
    buttons anyone can click, and Open file / drag-and-drop for your own PDFs, audio and videos."""

    def __init__(self, main, pdf_path):
        super().__init__()
        self.main = main
        self.tab = "pdf"
        self.last_clench = 0.0
        self.j_down = self.j_held = self.j_via_eeg = False
        self.look_key = None
        self.files = {"pdf": str(pdf_path), "audio": str(HERE / "static" / "media" / "sample_audio.wav"),
                      "video": str(HERE / "static" / "media" / "sample_video.mp4")}
        self.setStyleSheet(VIEWER_STYLE)
        self.setAcceptDrops(True)

        # ---- top bar: home, three big tabs, open file
        bar = QWidget(objectName="bar")
        top = QHBoxLayout(bar)
        top.setContentsMargins(14, 10, 14, 10)
        top.setSpacing(10)
        home = QPushButton("⌂  Home", objectName="bighome")
        home.clicked.connect(lambda: main.go("start"))
        top.addWidget(home)
        top.addSpacing(8)
        self.tab_buttons = QButtonGroup(self)
        segment = QFrame(objectName="segment")
        sl = QHBoxLayout(segment)
        sl.setContentsMargins(4, 4, 4, 4)
        sl.setSpacing(4)
        for i, label in enumerate(["📄  Read", "🎵  Listen", "🎬  Watch"]):
            b = QPushButton(label, objectName="seg", checkable=True)
            self.tab_buttons.addButton(b, i)
            sl.addWidget(b)
        top.addWidget(segment)
        self.tab_buttons.button(0).setChecked(True)
        self.tab_buttons.idClicked.connect(lambda i: self.set_tab(TABS[i]))
        top.addSpacing(8)
        open_btn = QPushButton("📂  Open file", objectName="bigopen")
        open_btn.setToolTip("Open your own PDF, audio or video (or drag a file onto this window)")
        open_btn.clicked.connect(self.open_dialog)
        top.addWidget(open_btn)
        top.addStretch(1)
        self.status = QLabel(objectName="pill")
        top.addWidget(self.status, alignment=Qt.AlignVCenter)
        for w in bar.findChildren(QPushButton):
            w.setFocusPolicy(Qt.NoFocus)

        self.targets = {a: FlickerTarget(a) for a in FREQ}
        for t in self.targets.values():
            t.clicked.connect(self.act)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.build_pdf(pdf_path))
        self.stack.addWidget(self.build_audio())
        self.stack.addWidget(self.build_video())

        # ---- bottom: what the jaw does here, and flicker comfort settings (small, out of the way)
        foot = QWidget(objectName="bar")
        fl = QHBoxLayout(foot)
        fl.setContentsMargins(14, 4, 14, 4)
        self.hint = QLabel(objectName="hint")
        fl.addWidget(self.hint, 1)
        fl.addWidget(QLabel("Flicker"))
        self.slider = QSlider(Qt.Horizontal, minimum=10, maximum=100, singleStep=5, pageStep=10)
        self.slider.setFixedWidth(120)
        self.slider.setToolTip("Lower = gentler on the eyes, but a weaker brain signal")
        self.strength_label = QLabel()
        self.slider.valueChanged.connect(self.set_strength)
        fl.addWidget(self.slider)
        fl.addWidget(self.strength_label)
        self.style_box = QComboBox()
        self.style_box.addItems(["Soft glow", "Box"])
        self.style_box.currentIndexChanged.connect(self.set_style)
        fl.addWidget(self.style_box)
        for w in (self.slider, self.style_box):
            w.setFocusPolicy(Qt.NoFocus)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(bar)
        lay.addWidget(self.stack, 1)
        lay.addWidget(foot)

        self.toast_label = QLabel(self, objectName="toast")
        self.toast_label.hide()
        self.toast_timer = QTimer(self, singleShot=True, timeout=self.toast_label.hide)
        self.single_timer = QTimer(self, singleShot=True, timeout=self.audio_back)
        self.hold_timer = QTimer(self, singleShot=True, timeout=self.j_hold_fired)

    # ---------- building
    @staticmethod
    def big_button(text, fn):
        b = QPushButton(text, objectName="bigctl")
        b.setFocusPolicy(Qt.NoFocus)
        b.clicked.connect(fn)
        return b

    @staticmethod
    def controls(buttons):
        row = QHBoxLayout()
        row.setSpacing(12)
        for b in buttons:
            row.addWidget(b, 1)
        return row

    def build_pdf(self, pdf_path):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(12, 8, 12, 12)
        v.setSpacing(8)
        self.pdf_name = QLabel(objectName="filename")
        v.addWidget(self.pdf_name)
        g = QGridLayout()
        g.setSpacing(10)
        self.pdf = PdfView(pdf_path)
        g.addWidget(self.targets["up"], 0, 1)
        g.addWidget(self.targets["left"], 1, 0)
        g.addWidget(self.pdf, 1, 1)
        g.addWidget(self.targets["right"], 1, 2)
        g.addWidget(self.targets["down"], 2, 1)
        for i in (0, 2):
            g.setRowMinimumHeight(i, 120)
            g.setColumnMinimumWidth(i, 130)
        g.setRowStretch(1, 1)
        g.setColumnStretch(1, 1)
        v.addLayout(g, 1)
        self.btn_arrows = self.big_button("👁  Arrows: ON", lambda: self.set_arrows(not self.pdf.arrows_on))
        v.addLayout(self.controls([
            self.big_button("⟨  Previous page", lambda: self.pdf.go_page(self.pdf.page - 1)),
            self.big_button("🔍  Zoom in / out", self.zoom_button),
            self.btn_arrows,
            self.big_button("Next page  ⟩", lambda: self.pdf.go_page(self.pdf.page + 1)),
        ]))
        return w

    def build_audio(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(12, 8, 12, 12)
        self.audio_name = QLabel(objectName="filename")
        v.addWidget(self.audio_name)
        v.addStretch(1)
        self.a_icon = QLabel("▶", objectName="big_icon", alignment=Qt.AlignCenter)
        self.a_time = QLabel("0:00", alignment=Qt.AlignCenter)
        self.a_time.setStyleSheet("font-size:16pt;")
        self.a_bar = QSlider(Qt.Horizontal, enabled=False)
        self.a_bar.setFixedWidth(560)
        v.addWidget(self.a_icon, alignment=Qt.AlignCenter)
        v.addSpacing(18)
        v.addWidget(self.a_bar, alignment=Qt.AlignCenter)
        v.addWidget(self.a_time)
        v.addStretch(1)
        self.btn_audio_play = self.big_button("▶  Play", lambda: self.toast(self.toggle(self.audio).capitalize()))
        v.addLayout(self.controls([
            self.big_button("⏪  Back 15 s", lambda: (self.skip(self.audio, -15000), self.toast("Back 15 s"))),
            self.btn_audio_play,
            self.big_button("Forward 15 s  ⏩", lambda: (self.skip(self.audio, 15000), self.toast("Forward 15 s"))),
        ]))
        self.audio = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.audio.setAudioOutput(self.audio_out)
        self.audio.positionChanged.connect(self.audio_progress)
        self.audio.playbackStateChanged.connect(self.audio_state)
        self.load_audio(self.files["audio"])
        return w

    def build_video(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(12, 8, 12, 12)
        v.setSpacing(8)
        self.video_name = QLabel(objectName="filename")
        v.addWidget(self.video_name)
        h = QHBoxLayout()
        h.setSpacing(12)
        self.targets["back"].setFixedWidth(160)
        self.targets["fwd"].setFixedWidth(160)
        vw = QVideoWidget()
        vw.setFocusPolicy(Qt.NoFocus)
        self.video = QMediaPlayer(self)
        self.video_out = QAudioOutput(self)
        self.video.setAudioOutput(self.video_out)
        self.video.setVideoOutput(vw)
        self.video.playbackStateChanged.connect(self.video_state)
        h.addWidget(self.targets["back"])
        h.addWidget(vw, 1)
        h.addWidget(self.targets["fwd"])
        v.addLayout(h, 1)
        self.btn_video_play = self.big_button("▶  Play", lambda: self.toast(self.toggle(self.video).capitalize()))
        v.addLayout(self.controls([
            self.big_button("⏪  Back 15 s", lambda: self.act("back")),
            self.btn_video_play,
            self.big_button("Forward 15 s  ⏩", lambda: self.act("fwd")),
        ]))
        self.load_video(self.files["video"])
        return w

    # ---------- files
    def open_dialog(self):
        start = str(Path(self.files.get(self.tab) or Path.home()).parent)
        path, _ = QFileDialog.getOpenFileName(self, "Open a PDF, audio or video file", start, OPEN_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path, remember=True):
        kind = kind_of(path)
        if not kind:
            QMessageBox.warning(self, "Can't open", f"{Path(path).name} isn't a PDF, audio or video file.")
            return False
        try:
            {"pdf": self.load_pdf, "audio": self.load_audio, "video": self.load_video}[kind](path)
        except Exception as e:
            QMessageBox.warning(self, "Can't open", f"Couldn't open {Path(path).name}:\n{e}")
            return False
        if remember:
            self.main.profile.settings.setdefault("last_files", {})[kind] = str(path)
            self.main.profile.save()
        self.set_tab(kind)
        self.toast(f"Opened {Path(path).name}")
        return True

    def load_pdf(self, path):
        self.pdf.load(path)
        self.files["pdf"] = str(path)
        self.pdf_name.setText(f"📄  {Path(path).name}")

    def load_audio(self, path):
        self.audio.stop()
        self.audio.setSource(QUrl.fromLocalFile(str(path)))
        self.audio.setLoops(QMediaPlayer.Infinite if "sample_audio" in str(path) else 1)
        self.files["audio"] = str(path)
        self.audio_name.setText(f"🎵  {Path(path).name}")

    def load_video(self, path):
        self.video.stop()
        self.video.setSource(QUrl.fromLocalFile(str(path)))
        self.files["video"] = str(path)
        self.video_name.setText(f"🎬  {Path(path).name}")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(kind_of(u.toLocalFile()) for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            if kind_of(u.toLocalFile()):
                self.open_path(u.toLocalFile())

    # ---------- page life
    def apply_rates(self):
        """Flicker rates come from the profile (defaults = what worked best on real data)."""
        for act, f in self.main.profile.settings.get("viewer_rates", {}).items():
            if act in FREQ:
                FREQ[act] = float(f)
                self.targets[act].freq = float(f)
                self.targets[act].update()

    def activate(self):
        s = self.main.profile.settings
        self.apply_rates()
        for kind, path in (s.get("last_files") or {}).items():
            if path and Path(path).exists() and path != self.files.get(kind):
                self.open_path(path, remember=False)
        self.pdf_name.setText(f"📄  {Path(self.files['pdf']).name}")
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(s["flicker_strength"] * 100)))
        self.slider.blockSignals(False)
        self.strength_label.setText(f"{self.slider.value()}%")
        self.style_box.blockSignals(True)
        self.style_box.setCurrentIndex(0 if s["flicker_style"] == "glow" else 1)
        self.style_box.blockSignals(False)
        self.set_tab(self.tab)

    def deactivate(self):
        self.audio.pause()
        self.video.pause()
        self.single_timer.stop()
        self.main.profile.save()

    def on_frame(self):
        for a in TAB_ACTS[self.tab]:
            self.targets[a].update()
        if self.tab == "pdf":
            self.pdf.tick()

    # ---------- helpers
    def toast(self, msg):
        self.toast_label.setText(msg)
        self.toast_label.adjustSize()
        self.toast_label.move((self.width() - self.toast_label.width()) // 2,
                              self.height() - self.toast_label.height() - 120)
        self.toast_label.show()
        self.toast_label.raise_()
        self.toast_timer.start(1500)

    def set_strength(self, v):
        Flicker.strength = v / 100
        self.main.profile.settings["flicker_strength"] = v / 100
        self.strength_label.setText(f"{v}%")

    def set_style(self, i):
        Flicker.style = ["glow", "box"][i]
        self.main.profile.settings["flicker_style"] = Flicker.style

    def send_targets(self):
        acts = [] if (self.tab == "pdf" and not self.pdf.arrows_on) else TAB_ACTS[self.tab]
        self.main.engine.set_freqs([FREQ[a] for a in acts])

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
        self.hint.setText({
            "pdf": "Look at a glowing arrow to scroll · jaw: clench twice = arrows on/off, hold 1 s = zoom",
            "audio": "Jaw: clench twice = play / pause · clench once = back 15 s",
            "video": "Look at a glowing arrow to skip 15 s · jaw: clench = play / pause"}[name])
        self.send_targets()

    def audio_progress(self, pos):
        dur = max(1, self.audio.duration())
        self.a_bar.setMaximum(dur)
        self.a_bar.setValue(pos)
        fmt = lambda ms: f"{ms // 60000}:{ms // 1000 % 60:02d}"
        self.a_time.setText(f"{fmt(pos)} / {fmt(dur)}")

    def audio_state(self, s):
        playing = s == QMediaPlayer.PlayingState
        self.a_icon.setText("❚❚" if playing else "▶")
        self.btn_audio_play.setText("❚❚  Pause" if playing else "▶  Play")

    def video_state(self, s):
        self.btn_video_play.setText("❚❚  Pause" if s == QMediaPlayer.PlayingState else "▶  Play")

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

    def zoom_button(self):
        self.toast("Zoomed in" if self.pdf.toggle_zoom() else "Zoomed out")

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
        self.btn_arrows.setText("👁  Arrows: ON" if on else "👁  Arrows: OFF")
        self.send_targets()
        self.toast("Arrows ON" if on else "Arrows OFF (read in peace)")

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
                self.toast(f"Double clench → {self.toggle(self.audio)}")
            else:
                self.toast("Back 15 s… (clench again now to play/pause instead)")
                self.single_timer.start(DOUBLE_MS)
        else:
            self.toast(f"Clench → {self.toggle(self.video)}")

    def hold(self):
        self.last_clench = 0.0
        if self.tab == "pdf":
            self.toast("Held clench → zoom in" if self.pdf.toggle_zoom() else "Held clench → zoom out")
        else:
            self.clench()

    def on_event(self, m):
        t = m["type"]
        if t == "status":
            self.status.setText(status_html(m, compact=True))
        elif not self.isVisible():
            return
        elif t == "clench":
            self.clench()
        elif t == "hold":
            self.hold()
        elif t == "ssvep":
            for a in TAB_ACTS[self.tab]:
                if FREQ[a] == m["freq"]:
                    self.act(a)
                    self.toast(f"Looked at {m['freq']:g} Hz")

    def handle_key(self, e):
        press = e.type() == QEvent.KeyPress
        key, repeat = e.key(), e.isAutoRepeat()
        fake = self.main.fake if self.main.engine.status.get("fake") else None
        via_eeg = fake is not None and bool(e.modifiers() & Qt.ShiftModifier)
        keymap = PDF_KEYS if self.tab == "pdf" else VIDEO_KEYS if self.tab == "video" else {}
        if key == Qt.Key_J:
            if repeat:
                return True
            if press and not self.j_down:
                self.j_down, self.j_held, self.j_via_eeg = True, False, via_eeg
                if via_eeg:
                    fake.start("clench")
                else:
                    self.hold_timer.start(HOLD_MS)
            elif not press and self.j_down:
                self.j_down = False
                if self.j_via_eeg:
                    fake.stop("clench")
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
                        fake.look(FREQ[keymap[key]])
                else:
                    self.act(keymap[key])
            elif not repeat and key == self.look_key:
                self.look_key = None
                fake.look(None)
            return True
        if not press and not repeat and key == self.look_key and fake:
            self.look_key = None
            fake.look(None)
        return False

    def j_hold_fired(self):
        if self.j_down:
            self.j_held = True
            self.hold()


# =========================================================================== calibration widgets
class CueCanvas(QWidget):
    """The big area the wearer looks at: instructions, a dot, flicker circle or glance arrow."""

    def __init__(self):
        super().__init__()
        self.text, self.sub = "Choose inputs on the left, then press Start calibration", ""
        self.flicker = None
        self.arrow = None
        self.phase = "idle"
        self.listening = False
        self.frac = 0.0  # how much of the current step has passed (0..1), drawn as a ring
        self.setMinimumSize(420, 360)

    def show_step(self, step):
        self.text, self.flicker, self.arrow, self.phase = step.text, step.flicker, step.arrow, step.phase
        self.listening, self.frac = False, 0.0
        self.update()

    def show_listen(self, step):
        """While the voice reads the instruction: text and dot only, no flicker or arrow yet."""
        self.text, self.sub, self.flicker, self.arrow, self.phase = step.text, "listen…", None, None, step.phase
        self.listening, self.frac = True, 0.0
        self.update()

    def set_message(self, text, sub=""):
        self.text, self.sub, self.flicker, self.arrow, self.phase = text, sub, None, None, "idle"
        self.listening, self.frac = False, 0.0
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect())
        bg = QRadialGradient(r.center(), max(r.width(), r.height()) * 0.7)
        bg.setColorAt(0, QColor("#0f1520"))
        bg.setColorAt(1, QColor("#070a0f"))
        p.setPen(Qt.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(r, 18, 18)
        c = QPointF(r.center().x(), r.center().y() + 30)
        active = self.phase in ("rest", "action", "settle") and not self.listening
        color = QColor(ACCENT if self.phase == "action" else MUTED)

        # the cue: flicker circle, or a dot to look at
        rad = min(r.width(), r.height()) * 0.2
        if self.flicker:
            level = flicker_level(self.flicker, time.perf_counter())
            if Flicker.style == "glow":
                paint_glow(p, c, rad, level)
            else:
                v = int(255 * level)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(v, v, v))
                p.drawRoundedRect(QRectF(c.x() - rad * 0.7, c.y() - rad * 0.7, rad * 1.4, rad * 1.4), 14, 14)
        elif self.phase != "idle":
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(TEXT))
            p.drawEllipse(c, 7, 7)

        # countdown ring: the arc shrinks as the step runs out
        if active:
            ring = rad * 1.25 if self.flicker else 46
            box = QRectF(c.x() - ring, c.y() - ring, 2 * ring, 2 * ring)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(LINE), 4))
            p.drawEllipse(box)
            pen = QPen(color, 4)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.drawArc(box, 90 * 16, -int(360 * 16 * max(0.0, 1.0 - self.frac)))

        if self.arrow:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(ARROW))
            x = r.left() + 44 if self.arrow == "left" else r.right() - 44
            d = 1 if self.arrow == "left" else -1
            p.drawPolygon(QPolygonF([QPointF(x, c.y()), QPointF(x + d * 56, c.y() - 38), QPointF(x + d * 56, c.y() + 38)]))

        # phase chip
        chip = {"settle": "GET READY", "rest": "REST", "action": "DO IT"}.get(self.phase, "")
        if self.listening:
            chip = "LISTEN"
        if chip:
            p.setFont(QFont(FONT, 9, QFont.Bold))
            w = p.fontMetrics().horizontalAdvance(chip) + 28
            cr = QRectF(r.center().x() - w / 2, 26, w, 26)
            chip_color = QColor(ACCENT if (self.phase == "action" and not self.listening) else "#7aa7ff" if self.listening else MUTED)
            fill = QColor(chip_color)
            fill.setAlpha(36)
            p.setPen(QPen(chip_color, 1))
            p.setBrush(fill)
            p.drawRoundedRect(cr, 13, 13)
            p.setPen(chip_color)
            p.drawText(cr, Qt.AlignCenter, chip)

        # instruction
        p.setPen(QColor(ACCENT if (self.phase == "action" and not self.listening) else TEXT))
        p.setFont(QFont(FONT_DISPLAY, 24 if self.phase == "action" else 20, QFont.Bold))
        p.drawText(QRectF(40, 64, r.width() - 80, 100), Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, self.text)
        p.setPen(QColor(MUTED))
        p.setFont(QFont(FONT, 13))
        p.drawText(QRectF(20, r.height() - 56, r.width() - 40, 36), Qt.AlignHCenter | Qt.AlignVCenter, self.sub)


METERS = [  # label, feature, scale, lo, hi, threshold setting
    ("Jaw muscle", "jaw", "log", 0.5, 300, "jaw_factor"),
    ("Eyebrow muscle", "brow", "log", 0.5, 300, "brow_factor"),
    ("Blink", "blink", "lin", -10, 80, "blink_z"),
    ("Glance  L+ / R−", "heog", "lin", -60, 60, "glance"),
    ("Eyes closed (alpha)", "alpha", "log", 0.3, 40, "alpha_factor"),
    ("Right hand (C3)", "mu_l", "log", 0.1, 3, "fist_right_ratio"),
    ("Left hand (C4)", "mu_r", "log", 0.1, 3, "fist_left_ratio"),
] + [(f"Flicker {f:g} Hz", f"ss_{f:g}", "lin", 0, 1, f"ssvep:{f:g}") for f in C.SSVEP_FREQS]


class MeterPanel(QWidget):
    """Live bar for every signal, with a white tick at the profile's threshold."""

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.values = {}
        self.setMinimumHeight(len(METERS) * 21 + 8)

    def set_values(self, feats, scores):
        self.values = dict(feats)
        self.values.update({f"ss_{k}": v for k, v in scores.items()})
        self.update()

    def _pos(self, v, scale, lo, hi, x0, w):
        if scale == "log":
            v = min(max(v, lo), hi)
            f = (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
        else:
            f = (min(max(v, lo), hi) - lo) / (hi - lo)
        return x0 + f * w

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        s = self.main.profile.settings
        x0, w = 150, self.width() - 150 - 60
        for i, (label, key, scale, lo, hi, thr_key) in enumerate(METERS):
            y = 4 + i * 21
            p.setPen(QColor(MUTED))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(QRectF(4, y, 144, 20), Qt.AlignVCenter | Qt.AlignLeft, label)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(LINE))
            p.drawRoundedRect(QRectF(x0, y + 7, w, 6), 3, 3)
            v = self.values.get(key)
            if key in ("blink", "heog") and v is not None:
                v = v * s["blink_sign" if key == "blink" else "heog_sign"]
            if thr_key == "glance":
                thrs = [s["glance_left_z"], -s["glance_right_z"]]
            elif thr_key.startswith("ssvep:"):
                thrs = [s["ssvep_thr"].get(thr_key.split(":")[1], 0.42)]
            else:
                thrs = [s[thr_key]]
            if v is not None:
                zero = self._pos(1.0 if scale == "log" else 0.0, scale, lo, hi, x0, w)
                xv = self._pos(v, scale, lo, hi, x0, w)
                if key == "heog":
                    over = v >= thrs[0] or v <= thrs[1]
                elif key in ("mu_l", "mu_r"):  # hand movement LOWERS the rhythm
                    over = v <= thrs[0]
                else:
                    over = v >= thrs[0]
                p.setBrush(QColor(ACCENT if over else "#5a6b80"))
                p.drawRoundedRect(QRectF(min(zero, xv), y + 7, max(2.0, abs(xv - zero)), 6), 3, 3)
                p.setPen(QColor(TEXT))
                p.drawText(QRectF(x0 + w + 6, y, 54, 20), Qt.AlignVCenter | Qt.AlignLeft,
                           f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}")
            p.setPen(QPen(QColor("#ffffff"), 2))
            for t in thrs:
                xt = self._pos(t, scale, lo, hi, x0, w)
                p.drawLine(QPointF(xt, y + 3), QPointF(xt, y + 17))


EVENT_TILES = [("clench", "Jaw clench"), ("hold", "Jaw hold"), ("brow", "Eyebrows"), ("blink", "Blink"),
               ("double_blink", "Double blink"), ("glance_left", "Glance ←"), ("glance_right", "Glance →"),
               ("eyes_closed", "Eyes closed"), ("fist_right", "Right fist"), ("fist_left", "Left fist")] + \
              [(f"ssvep_{f:g}", f"Flicker {f:g}") for f in C.SSVEP_FREQS]


class EventTiles(QWidget):
    def __init__(self):
        super().__init__()
        g = QGridLayout(self)
        g.setSpacing(6)
        g.setContentsMargins(0, 0, 0, 0)
        self.tiles, self.counts = {}, {}
        for i, (key, label) in enumerate(EVENT_TILES):
            t = QLabel(label, alignment=Qt.AlignCenter)
            t.setMinimumHeight(28)
            self.tiles[key], self.counts[key] = t, 0
            self._style(key, False)
            g.addWidget(t, i // 4, i % 4)

    def _style(self, key, lit):
        self.tiles[key].setStyleSheet(
            f"background:{ACCENT if lit else PANEL}; color:{'#08110c' if lit else TEXT}; border:1px solid {LINE};"
            "border-radius:6px; font-size:9pt;")

    def fire(self, m):
        key = f"ssvep_{m['freq']:g}" if m["type"] == "ssvep" else m["type"]
        if key not in self.tiles:
            return
        self.counts[key] += 1
        label = dict(EVENT_TILES)[key]
        self.tiles[key].setText(f"{label}  ×{self.counts[key]}")
        self._style(key, True)
        QTimer.singleShot(600, lambda: self._style(key, False))

    def reset(self):
        for key, label in EVENT_TILES:
            self.counts[key] = 0
            self.tiles[key].setText(label)


# =========================================================================== calibrate & test page
FAKE_KEYS = {Qt.Key_J: "clench", Qt.Key_E: "brow", Qt.Key_Left: "glance_left", Qt.Key_Right: "glance_right",
             Qt.Key_C: "eyes_closed", Qt.Key_R: "fist_right", Qt.Key_L: "fist_left"}
FAKE_LOOK_KEYS = {Qt.Key_1: 12, Qt.Key_2: 13, Qt.Key_3: 14, Qt.Key_4: 15}


def spoken(step):
    """Short phrase to read aloud, so the wearer doesn't have to read the screen."""
    if step.phase == "settle":
        return "Relax and sit still."
    if step.phase == "rest":
        return "Open your eyes. Relax." if step.input == "eyes_closed" else "Relax."
    t = step.text
    if step.input == "ssvep":
        return "Stare at the circle."
    if step.input == "glance":
        return f"Look {step.cls}, then back."
    if step.input == "fist":
        return f"Squeeze your {step.cls} fist."
    if step.input == "eyes_closed":
        return "Close your eyes."
    if step.input == "blink":
        return "Blink twice." if step.cls == "double" else "Blink once."
    if step.input == "jaw":
        return "Clench and hold." if step.cls == "hold" else "Clench your jaw."
    if step.input == "brow":
        return "Raise your eyebrows."
    return t


class CalibPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.steps, self.i, self.rec, self.kind = [], -1, None, None
        self.cur_t0 = None
        self.elapsed_before = 0.0
        self.held_keys = {}

        bar, top = top_bar("Calibrate & test", self.go_home)
        top.addStretch(1)
        self.status = QLabel()
        top.addWidget(self.status)

        # ---- left: what to run
        left = QFrame(objectName="panel")
        left.setFixedWidth(300)
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("Inputs to include", objectName="h2"))
        self.checks = {}
        for key, (label, secs) in C.INPUTS.items():
            cb = QCheckBox(f"{label}   ~{secs} s")
            cb.setChecked(True)
            cb.setFocusPolicy(Qt.NoFocus)
            cb.toggled.connect(self.update_total)
            self.checks[key] = cb
            lv.addWidget(cb)
        self.total = QLabel(objectName="hint")
        lv.addWidget(self.total)
        lv.addSpacing(8)
        self.btn_cal = QPushButton("Start calibration", objectName="primary")
        self.btn_redo = QPushButton("Redo Poor inputs only")
        self.btn_redo.setToolTip("Tick only the inputs whose latest calibration had a Poor result, then calibrate them")
        self.btn_test = QPushButton("Run live test")
        self.btn_stop = QPushButton("Stop")
        for b in (self.btn_cal, self.btn_redo, self.btn_test, self.btn_stop):
            b.setFocusPolicy(Qt.NoFocus)
            lv.addWidget(b)
        self.btn_cal.clicked.connect(lambda: self.start_session("calibration"))
        self.btn_test.clicked.connect(lambda: self.start_session("test"))
        self.btn_redo.clicked.connect(self.redo_poor)
        self.btn_stop.clicked.connect(self.stop_session)
        self.btn_stop.setEnabled(False)
        if main.tts:
            main.tts.stateChanged.connect(self.on_tts_state)
        self.speak = QCheckBox("Speak instructions out loud")
        self.speak.setChecked(True)
        self.speak.setFocusPolicy(Qt.NoFocus)
        self.speak.setEnabled(main.tts is not None)
        lv.addWidget(self.speak)
        self.auto = QCheckBox("Practice person follows the cues")
        self.auto.setChecked(True)
        self.auto.setFocusPolicy(Qt.NoFocus)
        lv.addWidget(self.auto)
        self.fake_help = QLabel("Practice keys (hold): J jaw · E eyebrows · B blink · ← → glance · "
                                "C eyes closed · R / L fist · 1–4 stare at 12–15 Hz", objectName="hint", wordWrap=True)
        lv.addWidget(self.fake_help)
        lv.addStretch(1)
        self.profile_label = QLabel(objectName="hint", wordWrap=True)
        lv.addWidget(self.profile_label)
        open_btn = QPushButton("Open saved trials folder")
        open_btn.setFocusPolicy(Qt.NoFocus)
        open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(data_dir()))))
        lv.addWidget(open_btn)

        # ---- centre: the cue
        centre = QWidget()
        cv = QVBoxLayout(centre)
        cv.setContentsMargins(0, 0, 0, 0)
        self.canvas = CueCanvas()
        cv.addWidget(self.canvas, 1)
        self.progress = QProgressBar(textVisible=False)
        cv.addWidget(self.progress)
        self.step_label = QLabel(objectName="hint")
        cv.addWidget(self.step_label)

        # ---- right: live signals + results
        self.tabs = QTabWidget()
        self.tabs.setFixedWidth(470)
        live = QWidget()
        lw = QVBoxLayout(live)
        self.meters = MeterPanel(main)
        lw.addWidget(self.meters)
        lw.addWidget(QLabel("Detections (white tick = this profile's threshold)", objectName="hint"))
        self.tiles = EventTiles()
        lw.addWidget(self.tiles)
        reset = QPushButton("Reset counts")
        reset.setFocusPolicy(Qt.NoFocus)
        reset.clicked.connect(self.tiles.reset)
        lw.addWidget(reset)
        lw.addStretch(1)
        self.tabs.addTab(live, "Live signals")
        res = QWidget()
        rv = QVBoxLayout(res)
        self.results_title = QLabel(objectName="h2", wordWrap=True)
        rv.addWidget(self.results_title)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Input", "Score", "Verdict", "Detail"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setWordWrap(True)
        rv.addWidget(self.table, 1)
        self.results_note = QLabel(objectName="hint", wordWrap=True)
        self.results_note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        rv.addWidget(self.results_note)
        self.tabs.addTab(res, "Results")

        body = QHBoxLayout()
        body.setContentsMargins(12, 12, 12, 12)
        body.setSpacing(12)
        body.addWidget(left)
        body.addWidget(centre, 1)
        body.addWidget(self.tabs)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(bar)
        lay.addLayout(body, 1)

        self.step_timer = QTimer(self, singleShot=True, timeout=self.next_step)
        self.tick_timer = QTimer(self, interval=100, timeout=self.tick)
        self.update_total()

    # ---------- page life
    def activate(self):
        fake = bool(self.main.engine.status.get("fake"))
        self.auto.setVisible(fake)
        self.fake_help.setVisible(fake)
        self.main.engine.set_freqs(C.SSVEP_FREQS)
        self.profile_label.setText(f"Profile: <b>{self.main.profile.name}</b><br>"
                                   f"Every session is saved as a trial in the {data_dir().name} folder")
        if not self.rec:
            self.show_saved_results()
        self.update_total()

    def deactivate(self):
        for action in list(self.held_keys.values()):
            self.release_fake(action)
        self.held_keys.clear()

    def go_home(self):
        if self.rec and QMessageBox.question(self, "Stop?", "A session is running. Stop it?") != QMessageBox.Yes:
            return
        self.stop_session()
        self.main.go("start")

    def update_total(self):
        inputs = self.selected()
        secs = C.total_seconds(C.build_calibration(inputs)) if inputs else 0
        self.total.setText(f"Calibration takes about {secs / 60:.1f} min" if inputs else "Pick at least one input")
        self.btn_cal.setEnabled(bool(inputs) and not self.rec)
        self.btn_test.setEnabled(bool(inputs) and not self.rec)
        self.btn_redo.setEnabled(bool(self.poor_inputs()) and not self.rec)

    def poor_inputs(self):
        """Inputs whose most recent CALIBRATION had any Poor item."""
        return sorted({r["input"] for r in self.main.profile.calibration.values()
                       if r.get("kind") == "calibration" and r.get("verdict") == "Poor" and r.get("input") in C.INPUTS})

    def redo_poor(self):
        poor = self.poor_inputs()
        if not poor:
            return
        for k, cb in self.checks.items():
            cb.setChecked(k in poor)
        self.start_session("calibration")

    def selected(self):
        return [k for k, cb in self.checks.items() if cb.isChecked()]

    def on_frame(self):
        if self.canvas.flicker:
            self.canvas.update()

    # ---------- running a session
    def start_session(self, kind):
        eng = self.main.engine
        if not eng.status.get("connected"):
            QMessageBox.warning(self, "No headset", "Connect a headset on the start screen first.")
            return
        inputs = self.selected()
        self.kind = kind
        self.seed = random.randrange(1_000_000)  # saved with the trial, so the order can be reproduced
        self.steps = (C.build_calibration(inputs, seed=self.seed) if kind == "calibration"
                      else C.build_test(inputs, reps=3, seed=self.seed))
        self.total_s = C.total_seconds(self.steps)
        self.rec = TrialRecorder(kind, self.main.profile.name, eng.info)
        eng.set_freqs(C.SSVEP_FREQS)
        eng.set_recorder(self.rec)
        self.i, self.cur_t0, self.elapsed_before = -1, None, 0.0
        self.voice_warning = ""
        for b, on in ((self.btn_cal, False), (self.btn_redo, False), (self.btn_test, False), (self.btn_stop, True)):
            b.setEnabled(on)
        for cb in self.checks.values():
            cb.setEnabled(False)
        self.tiles.reset()
        self.tabs.setCurrentIndex(0)
        self.tick_timer.start()
        self.next_step()

    def close_step(self):
        if self.cur_t0 is None:
            return
        st = self.steps[self.i]
        self.rec.add_cue(self.cur_t0, local_clock(), st.input, st.phase, st.cls, st.text, expect=list(st.expect))
        self.elapsed_before += st.dur
        self.cur_t0 = None

    def next_step(self):
        if not self.rec:
            return
        self.close_step()
        self.i += 1
        if self.i >= len(self.steps):
            self.canvas.set_message("Finishing…", "Saving the trial")
            QTimer.singleShot(1500, self.finish)
            return
        st = self.steps[self.i]
        if self.speak.isChecked() and self.main.tts:
            # read the whole instruction first. The cue (flicker circle / arrow) only appears and
            # the step's clock only starts once the voice has finished, so no action starts early
            # or without sound, and the recording is timed from when the wearer heard it.
            self.canvas.show_listen(st)
            self.waiting_speech, self.speech_started, self.speech_retried = True, False, False
            self.speech_deadline = time.monotonic() + 2.5
            self.main.say(st.text.replace("—", ","))
            return
        self.canvas.show_step(st)
        self.main.beep()
        self.begin_step()

    def on_tts_state(self, state):
        if not getattr(self, "waiting_speech", False):
            return
        if state == QTextToSpeech.State.Speaking:
            self.speech_started = True
        elif self.speech_started:
            self.begin_step()

    def begin_step(self):
        self.waiting_speech = False
        if not self.rec:
            return
        st = self.steps[self.i]
        self.canvas.show_step(st)  # the cue appears now, together with the clock starting
        self.cur_t0 = local_clock()
        fake = self.main.fake if self.main.engine.status.get("fake") else None
        if fake and self.auto.isChecked():
            for spec in st.fake:
                if spec[0] == "look":
                    fake.look(spec[3], delay=spec[2], seconds=spec[1])
                else:
                    fake.do(spec[0], spec[1], delay=spec[2])
        self.step_timer.start(int(st.dur * 1000))

    def tick(self):
        if self.rec and getattr(self, "waiting_speech", False) and time.monotonic() > self.speech_deadline:
            st = self.steps[self.i]
            if self.speech_started:  # it spoke but never reported finishing: carry on
                self.begin_step()
            elif not self.speech_retried:  # no sound yet: try the voice once more
                self.speech_retried = True
                self.speech_deadline = time.monotonic() + 2.5
                self.main.tts.stop()
                self.main.say(st.text.replace("—", ","))
            else:  # the voice isn't working: never start silently, beep instead
                self.main.beep()
                self.voice_warning = "⚠ Voice not working — using beeps. Check the volume / speaker. · "
                self.begin_step()
        if not self.rec or self.cur_t0 is None:
            return
        st = self.steps[self.i]
        in_step = local_clock() - self.cur_t0
        self.canvas.sub = f"{max(0.0, st.dur - in_step):.0f} s"
        self.canvas.frac = min(1.0, in_step / max(st.dur, 0.1))
        self.canvas.update()
        self.progress.setValue(int(100 * (self.elapsed_before + in_step) / max(self.total_s, 1)))
        left = self.total_s - self.elapsed_before - in_step
        self.step_label.setText(f"{getattr(self, 'voice_warning', '')}"
                                f"{'Calibration' if self.kind == 'calibration' else 'Live test'} · step "
                                f"{self.i + 1} of {len(self.steps)} · about {left / 60:.1f} min left")

    def end_ui(self):
        self.waiting_speech = False
        if self.main.tts:
            self.main.tts.stop()
        self.tick_timer.stop()
        self.step_timer.stop()
        self.main.engine.set_recorder(None)
        self.rec = None
        self.btn_stop.setEnabled(False)
        for cb in self.checks.values():
            cb.setEnabled(True)
        self.update_total()
        self.progress.setValue(0)

    def stop_session(self):
        if not self.rec:
            return
        self.step_timer.stop()
        self.close_step()
        done = sum(1 for c in self.rec.cues if c["phase"] == "action")
        if done and QMessageBox.question(
                self, "Save what was recorded?",
                f"{done} action step(s) were recorded. Save them as a partial trial?\n"
                "(Scores are shown, but a partial calibration does not change the profile's settings.)") \
                == QMessageBox.Yes:
            self.finish(partial=True)
            return
        self.end_ui()
        self.canvas.set_message("Stopped — nothing was saved")
        self.step_label.setText("")

    def finish(self, partial=False):
        rec, kind = self.rec, self.kind
        if not rec:
            return
        self.end_ui()
        if partial:
            rec.kind = f"{kind}_partial"
        rows, events, cues = rec.snapshot()
        prof = self.main.profile
        if kind == "calibration":
            results, update = C.analyze_calibration(rows, cues)
            folder = rec.save(results, {"inputs": self.selected(), "seed": self.seed, "randomized": True,
                                        "partial": partial, "settings_used": dict(prof.settings)})
            prof.record(rec.kind, results, folder, None if partial else update)
            note = ("Partial session: scores saved, settings NOT changed." if partial else
                    "Settings updated for Good and OK inputs; Poor ones kept their old settings.")
            summary = None
        else:
            results, summary = C.score_test(events, cues)
            folder = rec.save(results, {"inputs": self.selected(), "summary": summary, "seed": self.seed,
                                        "partial": partial, "settings_used": dict(prof.settings)})
            prof.record(rec.kind, results, folder)
            note = (f"False alarms while resting: {summary['false_alarms']} "
                    f"({summary['false_alarms_per_min']}/min"
                    f"{', from ' + ', '.join(summary['false_alarm_types']) if summary['false_alarm_types'] else ''}).")
        prof.save()
        self.main.apply_profile()
        good = sum(r["verdict"] == "Good" for r in results)
        ranked = sorted(results, key=lambda r: -r["accuracy"])
        best = ", ".join(f"{r['item']} {r['accuracy']:.0%}" for r in ranked[:5])
        note = f"BEST SIGNALS: {best}\n\n{note}"
        self.canvas.set_message(f"{'Calibration' if kind == 'calibration' else 'Live test'} done: "
                                f"{good} of {len(results)} Good", "See the Results tab")
        self.show_results(results, f"{'Calibration' if kind == 'calibration' else 'Live test'} — just now",
                          f"{note}\nSaved to profile '{prof.name}' and trial folder:\n{folder}")
        self.step_label.setText("")

    # ---------- results
    def show_results(self, results, title, note):
        results = sorted(results, key=lambda r: -r.get("accuracy", 0))  # best first
        self.results_title.setText(title)
        self.table.setRowCount(len(results))
        for i, r in enumerate(results):
            cells = [r["item"], f"{r['accuracy']:.0%}", r["verdict"], r.get("detail", "")]
            for j, txt in enumerate(cells):
                it = QTableWidgetItem(txt)
                if j == 2:
                    it.setForeground(QColor(VERDICT_COLOR.get(r["verdict"], TEXT)))
                self.table.setItem(i, j, it)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.resizeRowsToContents()
        self.results_note.setText(note)
        self.tabs.setCurrentIndex(1)

    def show_saved_results(self):
        cal = self.main.profile.calibration
        if not cal:
            self.results_title.setText("No results yet for this profile")
            self.table.setRowCount(0)
            self.results_note.setText("Run a calibration to see what works for this wearer.")
            return
        results = sorted(cal.values(), key=lambda r: (r.get("kind", ""), r.get("input", ""), r.get("item", "")))
        self.show_results(results, f"Latest saved results for '{self.main.profile.name}'",
                          "Each row is the most recent calibration or test of that input. The full history is in "
                          "the profile file and in trials/index.csv.")
        self.tabs.setCurrentIndex(0)
        for i, r in enumerate(results):
            self.table.item(i, 3).setText(f"{r.get('kind', '')} {r.get('date', '')[:16]} · {r.get('detail', '')}")

    # ---------- events
    def on_event(self, m):
        if m["type"] == "status":
            self.status.setText(status_html(m, compact=True))
            if self.isVisible():
                self.meters.set_values(m.get("features", {}), m.get("scores", {}))
        elif self.isVisible():
            self.tiles.fire(m)

    # ---------- practice keys (fake headset only)
    def release_fake(self, action):
        fake = self.main.fake
        if not fake:
            return
        if isinstance(action, float):
            fake.look(None)
        elif action != "blink":
            fake.stop(action)

    def handle_key(self, e):
        fake = self.main.fake if self.main.engine.status.get("fake") else None
        key = e.key()
        if not fake or (key not in FAKE_KEYS and key not in FAKE_LOOK_KEYS and key != Qt.Key_B):
            return False
        if e.isAutoRepeat():
            return True
        if e.type() == QEvent.KeyPress and key not in self.held_keys:
            if key == Qt.Key_B:
                fake.do("blink", 0.25)
                self.held_keys[key] = "blink"
            elif key in FAKE_LOOK_KEYS:
                fake.look(FAKE_LOOK_KEYS[key])
                self.held_keys[key] = float(FAKE_LOOK_KEYS[key])
            else:
                fake.start(FAKE_KEYS[key])
                self.held_keys[key] = FAKE_KEYS[key]
        elif e.type() == QEvent.KeyRelease and key in self.held_keys:
            self.release_fake(self.held_keys.pop(key))
        return True


# =========================================================================== start page
class ChoiceCard(QFrame):
    """A big clickable card: icon, title, one-line description, arrow."""

    def __init__(self, icon, title, desc, on_click):
        super().__init__(objectName="card")
        self.on_click = on_click
        self.setAttribute(Qt.WA_Hover)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(140)
        h = QHBoxLayout(self)
        h.setContentsMargins(26, 22, 26, 22)
        h.setSpacing(20)
        h.addWidget(QLabel(icon, objectName="cardicon"), alignment=Qt.AlignTop)
        v = QVBoxLayout()
        v.setSpacing(6)
        v.addWidget(QLabel(title, objectName="cardtitle"))
        d = QLabel(desc, objectName="carddesc", wordWrap=True)
        v.addWidget(d)
        v.addStretch(1)
        h.addLayout(v, 1)
        h.addWidget(QLabel("→", objectName="cardarrow"), alignment=Qt.AlignVCenter)

    def mousePressEvent(self, e):
        self.on_click()


class StartPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        outer = QVBoxLayout(self)
        outer.setContentsMargins(48, 34, 48, 26)
        outer.setSpacing(22)

        # ---- header: icon, name, tagline, live connection pill
        header = QHBoxLayout()
        header.setSpacing(18)
        if (HERE / "icon.ico").exists():
            logo = QLabel()
            logo.setPixmap(QIcon(str(HERE / "icon.ico")).pixmap(64, 64))
            header.addWidget(logo)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(QLabel("Hands-Free Viewer", objectName="h1"))
        tag = QLabel("Look at a glowing target. Clench your jaw. Read, listen and watch — no hands needed.",
                     objectName="hint")
        tag.setStyleSheet("font-size:12pt;")
        titles.addWidget(tag)
        header.addLayout(titles)
        header.addStretch(1)
        self.conn_pill = QLabel(objectName="pill")
        header.addWidget(self.conn_pill, alignment=Qt.AlignVCenter)
        outer.addLayout(header)

        def step_header(n, text):
            h = QHBoxLayout()
            h.setSpacing(12)
            badge = QLabel(str(n), objectName="badge", alignment=Qt.AlignCenter)
            h.addWidget(badge)
            h.addWidget(QLabel(text, objectName="h2"))
            h.addStretch(1)
            return h

        row = QHBoxLayout()
        row.setSpacing(18)
        # ---- profile
        prof = QFrame(objectName="panel")
        pv = QVBoxLayout(prof)
        pv.setContentsMargins(22, 20, 22, 20)
        pv.setSpacing(12)
        pv.addLayout(step_header(1, "Who is wearing the headset?"))
        pr = QHBoxLayout()
        self.profile_box = QComboBox()
        self.profile_box.setMinimumWidth(200)
        self.profile_box.activated.connect(lambda i: self.main.set_profile(self.profile_box.itemText(i)))
        pr.addWidget(self.profile_box, 1)
        new = QPushButton("＋  New profile")
        new.clicked.connect(self.new_profile)
        pr.addWidget(new)
        pv.addLayout(pr)
        self.profile_info = QLabel(objectName="hint", wordWrap=True)
        pv.addWidget(self.profile_info)
        pv.addStretch(1)
        row.addWidget(prof, 1)

        # ---- headset
        head = QFrame(objectName="panel")
        hv = QVBoxLayout(head)
        hv.setContentsMargins(22, 20, 22, 20)
        hv.setSpacing(12)
        hv.addLayout(step_header(2, "Connect the headset"))
        self.rb_real = QRadioButton("Real headset (eego over LSL)")
        self.rb_fake = QRadioButton("Practice headset (fake signals, keyboard-controlled)")
        self.rb_real.setChecked(True)
        hv.addWidget(self.rb_real)
        sr = QHBoxLayout()
        self.stream_box = QComboBox()
        self.stream_box.addItem("Best EEG stream found", None)
        sr.addWidget(self.stream_box, 1)
        self.search_btn = QPushButton("⟳  Search")
        self.search_btn.clicked.connect(self.search)
        sr.addWidget(self.search_btn)
        hv.addLayout(sr)
        hv.addWidget(self.rb_fake)
        connect = QPushButton("Connect", objectName="primary")
        connect.clicked.connect(self.connect)
        hv.addWidget(connect)
        self.conn = QLabel(wordWrap=True)
        hv.addWidget(self.conn)
        self.conn_detail = QLabel(objectName="hint", wordWrap=True)
        hv.addWidget(self.conn_detail)
        hv.addWidget(QLabel("Tip: on ANT's network, allow Python through the Windows firewall if asked.",
                            objectName="hint", wordWrap=True))
        hv.addStretch(1)
        row.addWidget(head, 1)
        outer.addLayout(row)

        # ---- the two choices, as big cards
        outer.addLayout(step_header(3, "Choose what to do"))
        choices = QHBoxLayout()
        choices.setSpacing(18)
        choices.addWidget(ChoiceCard("🧪", "Calibrate & test",
                                     "Guided calibration of every signal, a scored live test and live meters — "
                                     "saved to this person's profile.", lambda: self.main.go("calib")), 1)
        choices.addWidget(ChoiceCard("📖", "Use the viewer",
                                     "Read PDFs, listen to audio and watch video hands-free, "
                                     "with this person's settings.", lambda: self.main.go("viewer")), 1)
        outer.addLayout(choices)
        outer.addStretch(1)
        foot = QHBoxLayout()
        self.data_label = QLabel(objectName="hint")
        foot.addWidget(self.data_label, 1)
        open_btn = QPushButton("Open data folder")
        open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(data_dir()))))
        foot.addWidget(open_btn)
        outer.addLayout(foot)

        main.bridge.streams.connect(self.got_streams)

    def activate(self):
        names = list_profiles()
        if self.main.profile.name not in names:
            names.append(self.main.profile.name)
        self.profile_box.clear()
        self.profile_box.addItems(sorted(names, key=str.lower))
        self.profile_box.setCurrentText(self.main.profile.name)
        cal = self.main.profile.calibration
        if cal:
            counts = {v: sum(r["verdict"] == v for r in cal.values()) for v in ("Good", "OK", "Poor")}
            last = max(r.get("date", "") for r in cal.values())
            good = list(dict.fromkeys(r["item"] for r in cal.values() if r["verdict"] == "Good"))
            self.profile_info.setText(f"Last calibrated {last[:16].replace('T', ' ')} · "
                                      f"{counts['Good']} Good, {counts['OK']} OK, {counts['Poor']} Poor"
                                      + (f"<br>Works well: {', '.join(good)}" if good else ""))
        else:
            self.profile_info.setText("Not calibrated yet. Default settings are used until you calibrate.")
        self.data_label.setText(f"Profiles and trials are saved in {data_dir()}")
        self.rb_fake.setChecked(bool(self.main.engine.status.get("fake")))
        self.rb_real.setChecked(not self.main.engine.status.get("fake"))

    def deactivate(self):
        pass

    def new_profile(self):
        name, ok = QInputDialog.getText(self, "New profile", "Wearer's name:")
        if ok and name.strip():
            self.main.set_profile(name.strip(), create=True)
            self.activate()

    def search(self):
        self.search_btn.setEnabled(False)
        self.search_btn.setText("Searching…")
        import threading
        threading.Thread(target=lambda: self.main.bridge.streams.emit(find_streams(2.0)), daemon=True).start()

    def got_streams(self, streams):
        self.search_btn.setEnabled(True)
        self.search_btn.setText("Search")
        self.stream_box.clear()
        self.stream_box.addItem("Best EEG stream found", None)
        for s in streams:
            if s["name"] == FAKE_NAME:
                continue
            self.stream_box.addItem(f"{s['name']} · {s['type']} · {s['channels']} ch · {s['rate']:g} Hz", s["name"])
        self.conn_detail.setText(f"Found {sum(s['name'] != FAKE_NAME for s in streams)} stream(s).")

    def connect(self):
        if self.rb_fake.isChecked():
            self.main.connect_fake()
        else:
            self.main.connect_real(self.stream_box.currentData())

    def on_event(self, m):
        if m["type"] != "status":
            return
        self.conn.setText(status_html(m))
        self.conn_pill.setText(status_html(m, compact=True))
        if m.get("connected"):
            names = "" if m.get("names_from_stream", True) else \
                " · channel names missing, assuming the standard eego 24 layout"
            missing = (f"<br><span style='color:{WARN}'>Not found: {', '.join(m['missing'])} — the inputs that "
                       f"need them will work poorly. Check the channel names with ANT staff.</span>"
                       if m.get("missing") else "")
            self.conn_detail.setText(f"{m.get('stream_type', 'EEG')} stream · {m.get('n_channels', '?')} channels "
                                     f"at {m.get('rate', 0):g} Hz{names}{missing}")

    def handle_key(self, e):
        return False

    def on_frame(self):
        pass


# =========================================================================== main window
class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("Hands-Free Viewer")
        self.resize(1320, 860)
        self.setStyleSheet(STYLE)
        state = load_state()
        self.profile = Profile.load(args.profile or state.get("last_profile") or "default")
        Flicker.strength, Flicker.style = self.profile.settings["flicker_strength"], self.profile.settings["flicker_style"]
        self.fake = None
        self.bridge = Bridge()
        self.bridge.event.connect(self.on_event)
        self.engine = LiveEngine(self.safe_emit, self.profile.settings)
        self.beep_sound = self.make_beep()
        self.tts = None
        if QTextToSpeech is not None:
            try:
                engine = os.environ.get("HFV_TTS_ENGINE")  # "mock" = silent, for tests
                self.tts = QTextToSpeech(engine, self) if engine else QTextToSpeech(self)
                self.tts.setRate(0.0)  # normal speaking pace
            except Exception:
                self.tts = None

        self.pages = {"start": StartPage(self), "calib": CalibPage(self), "viewer": ViewerPage(self, args.pdf)}
        self.stack = QStackedWidget()
        for p in self.pages.values():
            self.stack.addWidget(p)
        self.setCentralWidget(self.stack)
        self.current = "start"

        hz = (self.screen().refreshRate() if self.screen() else 60) or 60
        self.frame = QTimer(self, interval=max(4, int(1000 / hz)), timerType=Qt.PreciseTimer,
                            timeout=lambda: self.pages[self.current].on_frame())
        self.frame.start()
        QApplication.instance().installEventFilter(self)

        if args.fake:
            self.connect_fake()
        else:
            self.connect_real(args.stream)
        self.go("start")

    # ---------- headset
    def connect_fake(self):
        if not self.fake:
            self.fake = FakeHeadset()
            self.fake.begin()
        self.engine.connect(FAKE_NAME, fake=True, source_id=self.fake.source_id)

    def connect_real(self, stream_name=None):
        self.engine.connect(stream_name, fake=False)

    def safe_emit(self, msg):
        try:
            self.bridge.event.emit(msg)
        except RuntimeError:  # window closing
            pass

    def on_event(self, m):
        for p in self.pages.values():
            p.on_event(m)

    # ---------- profiles
    def set_profile(self, name, create=False):
        self.profile.save()
        self.profile = Profile.load(name)
        if create:
            self.profile.save()
        save_state(last_profile=self.profile.name)
        self.apply_profile()

    def apply_profile(self):
        s = self.profile.settings
        Flicker.strength, Flicker.style = s["flicker_strength"], s["flicker_style"]
        self.engine.apply_settings(s)

    # ---------- navigation
    def go(self, name):
        self.pages[self.current].deactivate()
        self.current = name
        self.stack.setCurrentWidget(self.pages[name])
        self.pages[name].activate()
        save_state(last_profile=self.profile.name)

    def make_beep(self):
        path = data_dir() / "beep.wav"
        if not path.exists():
            rate, t = 44100, np.arange(int(0.15 * 44100)) / 44100
            sig = 0.4 * np.sin(2 * np.pi * 880 * t) * np.minimum(1, np.minimum(t, 0.15 - t) / 0.01)
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes((sig * 32767).astype(np.int16).tobytes())
        s = QSoundEffect(self)
        s.setSource(QUrl.fromLocalFile(str(path)))
        s.setVolume(0.5)
        return s

    def beep(self):
        self.beep_sound.play()

    def say(self, text):
        self.tts.say(text)

    # ---------- keyboard
    def eventFilter(self, obj, e):
        if e.type() not in (QEvent.KeyPress, QEvent.KeyRelease):
            return False
        if QApplication.activeModalWidget() or isinstance(QApplication.focusWidget(), QLineEdit):
            return False
        return self.pages[self.current].handle_key(e)

    def closeEvent(self, e):
        cal = self.pages["calib"]
        if cal.rec:  # closing mid-session: keep the data rather than lose it
            cal.step_timer.stop()
            cal.close_step()
            cal.finish(partial=True)
        self.pages[self.current].deactivate()
        self.profile.save()
        self.engine.stop()
        super().closeEvent(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="start with the practice (fake) headset connected")
    ap.add_argument("--stream", help="LSL stream name for the real headset (default: first EEG stream)")
    ap.add_argument("--profile", help="profile to open (default: the last one used)")
    ap.add_argument("--pdf", default=str(HERE / "media" / "sample.pdf"))
    ap.add_argument("--no-warning", action="store_true", help="skip the flashing warning (for testing)")
    args = ap.parse_args()
    if not Path(args.pdf).exists():
        raise SystemExit(f"No PDF at {args.pdf}. Run: uv run python bci_demo/make_media.py  (or pass --pdf FILE)")

    app = QApplication(sys.argv)
    if (HERE / "icon.ico").exists():
        app.setWindowIcon(QIcon(str(HERE / "icon.ico")))
    if not args.no_warning:
        ok = QMessageBox.warning(
            None, "Hands-Free Viewer",
            "Flashing warning: parts of this app flicker 12–15 times a second. Don't use it if you or anyone "
            "watching has photosensitive epilepsy.",
            QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok)
        if ok != QMessageBox.Ok:
            return
    win = MainWindow(args)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
