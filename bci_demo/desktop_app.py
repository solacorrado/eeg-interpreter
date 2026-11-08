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
from collections import deque
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from pylsl import local_clock
from PySide6.QtCore import QEvent, QObject, QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QImage, QLinearGradient, QPainter, QPen, QPolygonF, QRadialGradient
from PySide6.QtCore import QPoint, QPointF
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QSoundEffect
from PySide6.QtMultimediaWidgets import QVideoWidget
try:
    from PySide6.QtTextToSpeech import QTextToSpeech
except ImportError:  # speech is optional
    QTextToSpeech = None
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
                               QMenu, QWidgetAction, QSizePolicy,
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
# Design tokens, from the team palette: charcoal #2C2B30, graphite #4F4F51, light grey #D6D6D6,
# blush #F2C4CE and coral #F58F7C. Panels are small steps between charcoal and graphite. The flicker
# targets stay white glows (the brightest thing on screen) with coral arrows.
BG, PANEL, PANEL2, LINE, TEXT, MUTED = "#2C2B30", "#333238", "#3A393F", "#4F4F51", "#D6D6D6", "#9C9AA2"
ACCENT, WARN, BAD, ARROW = "#F58F7C", "#F2C27C", "#E8606F", "#F58F7C"
GOOD = "#8FD1A4"
ACCENT_SOFT, ACCENT_HI, ON_ACCENT = "#4A3A3D", "#F2C4CE", "#2C2B30"
# Atkinson Hyperlegible: made by the Braille Institute for low-vision readers. IBM Plex Mono: readouts.
FONT, FONT_DISPLAY, MONO = "Atkinson Hyperlegible", "Atkinson Hyperlegible", "IBM Plex Mono"
VERDICT_COLOR = {"Good": GOOD, "OK": WARN, "Poor": BAD}
GLOW_STOPS = [(0.0, 1.0), (0.35, 0.95), (0.6, 0.65), (0.8, 0.3), (1.0, 0.0)]  # soft edge, wide bright core
MINT_GRADIENT = ACCENT  # (solid accent fill; no gradients)

STYLE = f"""
QWidget {{ background:{BG}; color:{TEXT}; font-family:"{FONT}", "Segoe UI"; font-size:10.5pt; }}
QPushButton#primary, QPushButton#bigopen {{ letter-spacing:0.2px; }}
QToolTip {{ background:{PANEL2}; color:{TEXT}; border:1px solid {LINE}; padding:6px 8px; }}
#bar {{ background:{BG}; border-bottom:1px solid {LINE}; }}
#bar QLabel, #bar QSlider {{ background:transparent; }}
#panel {{ background:{PANEL}; border:1px solid {LINE}; border-radius:14px; }}
#panel QLabel, #panel QCheckBox, #panel QRadioButton {{ background:transparent; }}
QPushButton {{ background:{PANEL2}; border:1px solid {LINE}; color:{TEXT}; padding:8px 16px; border-radius:8px; }}
QPushButton:hover {{ background:#444349; border-color:#6A6A6E; }}
QPushButton:pressed {{ background:#262529; }}
QPushButton:checked {{ background:{ACCENT_SOFT}; border-color:{ACCENT}; color:#F2C4CE; }}
QPushButton:disabled {{ color:#6E6D73; background:#303034; border-color:#3E3E42; }}
QPushButton#primary {{ background:{MINT_GRADIENT}; color:{ON_ACCENT}; border:none; font-weight:700; padding:10px 18px; }}
QPushButton#primary:hover {{ background:{ACCENT_HI}; }}
QPushButton#primary:pressed {{ background:#E07A67; }}
QPushButton#primary:disabled {{ background:#3A393F; color:#6E6D73; }}
QComboBox {{ background:{PANEL2}; border:1px solid {LINE}; border-radius:10px; padding:7px 12px; }}
QComboBox:hover {{ border-color:#6A6A6E; }}
QComboBox::drop-down {{ border:none; width:26px; }}
QComboBox QAbstractItemView {{ background:{PANEL2}; border:1px solid {LINE}; selection-background-color:{ACCENT_SOFT};
                               selection-color:{TEXT}; outline:none; padding:4px; }}
QSlider::groove:horizontal {{ height:6px; background:{PANEL2}; border-radius:3px; }}
QSlider::sub-page:horizontal {{ background:{MINT_GRADIENT}; border-radius:3px; }}
QSlider::handle:horizontal {{ background:#F2C4CE; border:2px solid {ACCENT}; width:14px; margin:-6px 0; border-radius:9px; }}
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
QScrollBar::handle:vertical {{ background:#5A5A5E; border-radius:4px; min-height:30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height:0; width:0; }}
QCheckBox {{ spacing:10px; }}
QRadioButton {{ spacing:10px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width:16px; height:16px; border:1.5px solid #77767C; background:{BG}; }}
QCheckBox::indicator {{ border-radius:5px; }}
QRadioButton::indicator {{ border-radius:9px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{ background:{ACCENT}; border-color:{ACCENT}; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color:{ACCENT}; }}
QCheckBox::indicator:disabled {{ border-color:{LINE}; }}
#toast {{ background:rgba(51, 50, 56, 245); border:1px solid {ACCENT}; border-radius:14px; padding:10px 22px;
          font-size:13pt; color:{TEXT}; }}
#big_icon {{ font-size:54pt; color:{ACCENT}; background:{ACCENT_SOFT}; border:2px solid {ACCENT};
             border-radius:90px; min-width:180px; max-width:180px; min-height:180px; max-height:180px; }}
#hint {{ color:{MUTED}; background:transparent; }}
#h1 {{ font-family:"{FONT_DISPLAY}", "Segoe UI"; font-size:26pt; font-weight:700; background:transparent; }}
#h2 {{ font-family:"{FONT_DISPLAY}", "Segoe UI"; font-size:13pt; font-weight:600; background:transparent; }}
#panel #badge, #badge {{ background:{ACCENT_SOFT}; color:{ACCENT}; border-radius:14px; font-weight:700; font-size:11pt;
          min-width:28px; max-width:28px; min-height:28px; max-height:28px; }}
#bar #pill, #pill {{ background:{PANEL2}; border:1px solid {LINE}; border-radius:14px; padding:5px 14px;
          font-family:"{MONO}", "Cascadia Mono", monospace; font-size:9.5pt; }}
#card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:14px; }}
#card:hover {{ border:1px solid {ACCENT}; background:#3B3437; }}
#cardkicker {{ color:{ACCENT}; font-family:"{MONO}", monospace; font-size:9pt; letter-spacing:1.5px; }}
#chip {{ background:{PANEL2}; border:1px solid {LINE}; border-radius:13px; padding:4px 12px; font-size:10.5pt; }}
#card QLabel {{ background:transparent; }}
#cardtitle {{ font-family:"{FONT_DISPLAY}", "Segoe UI"; font-size:21pt; font-weight:700; }}
#cardicon {{ font-size:22pt; color:{ACCENT}; border:1.5px solid {ACCENT}; border-radius:28px;
              min-width:56px; max-width:56px; min-height:56px; max-height:56px; qproperty-alignment:AlignCenter; }}
#carddesc {{ color:{MUTED}; font-size:11pt; }}
#cardarrow {{ color:{ACCENT}; font-size:26pt; font-weight:700; }}
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
        calib = " · <b>settling…</b>" if m.get("calibrating") else ""
        jaw = f.get("jaw")
        jaw_txt = (f" · jaw <b style='color:{WARN if m.get('clenching') else TEXT}'>{jaw:.1f}×</b>"
                   if jaw is not None else "")
        name = short_stream(m["stream"])
        name = name if len(name) <= 14 else name[:13] + "…"
        return (f"<span style='color:{GOOD}'>●</span> {name}{' (sim)' if m.get('fake') else ''}{calib}{jaw_txt}")
    jaw = f.get("jaw")
    jaw_txt = f"   jaw <b style='color:{WARN if m.get('clenching') else TEXT}'>{jaw:.1f}×</b>" if jaw is not None else ""
    scores = "  ".join(f"{k} Hz {v:.2f}" for k, v in m.get("scores", {}).items())
    return (f"<span style='color:{GOOD}'>●</span> {short_stream(m['stream'])}{' (practice)' if m.get('fake') else ''}{calib}"
            f"   <span style='color:{MUTED}'>{scores}</span>{jaw_txt}")


class SignalStrip(QWidget):
    """A small live trace of the jaw-muscle signal from the headset: it moves whenever data is
    arriving (and jumps on a clench), so anyone can see at a glance that the EEG is live."""

    def __init__(self, width=150, big=False):
        super().__init__()
        self.big = big
        if big:  # wide band across the start screen, drawn like EEG chart paper
            self.setFixedHeight(86)
            self.setMinimumWidth(200)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        else:
            self.setFixedSize(width, 34)
        self.vals = deque(maxlen=150 if big else 60)  # 15 s / 6 s at 10 updates a second
        self.live, self.clenching, self.jaw = False, False, None
        self.setToolTip("Live signal from the headset (jaw muscle). It moves when data is arriving.")

    def push(self, m):
        if m.get("type") != "status":
            return
        self.live = bool(m.get("connected")) and not m.get("stalled")
        self.clenching = bool(m.get("clenching"))
        jaw = (m.get("features") or {}).get("jaw")
        self.jaw = jaw if self.live else None
        if self.live and jaw:
            self.vals.append(math.log10(max(0.3, min(jaw, 300.0))))
        elif not self.live:
            self.vals.clear()
        self.update()

    def paint_big(self, p):
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(LINE), 1))
        p.setBrush(QColor(PANEL))
        p.drawRoundedRect(r, 14, 14)
        inner = r.adjusted(16, 30, -16, -12)
        n = self.vals.maxlen
        step = inner.width() / (n - 1)
        grid = QColor("#3E3D43")
        for k in range(0, n, 10):  # one line per second, like EEG paper
            x = inner.left() + k * step
            p.setPen(QPen(grid, 1))
            p.drawLine(QPointF(x, inner.top()), QPointF(x, inner.bottom()))
        for k in range(5):
            y = inner.top() + inner.height() * k / 4
            p.setPen(QPen(grid, 1, Qt.SolidLine if k in (0, 4) else Qt.DotLine))
            p.drawLine(QPointF(inner.left(), y), QPointF(inner.right(), y))
        live = self.live and len(self.vals) >= 2
        p.setFont(QFont(MONO, 8.5, QFont.Medium))
        cap = r.adjusted(16, 8, -16, 0)
        p.setPen(QColor(GOOD if live else MUTED))
        p.drawText(cap, Qt.AlignLeft | Qt.AlignTop,
                   "●  LIVE · jaw muscle (30–100 Hz) · 1 s grid" if live else "○  NO SIGNAL · connect the headset below")
        if live and self.jaw is not None:
            p.setPen(QColor(WARN if self.clenching else TEXT))
            p.drawText(cap, Qt.AlignRight | Qt.AlignTop,
                       f"{'CLENCH  ' if self.clenching else ''}{self.jaw:.1f}× rest")
        if not live:
            p.setPen(QPen(QColor(MUTED), 1, Qt.DashLine))
            p.drawLine(QPointF(inner.left(), inner.center().y()), QPointF(inner.right(), inner.center().y()))
            return
        lo, hi = math.log10(0.3), math.log10(300.0)
        pts = [QPointF(inner.left() + step * (i + n - len(self.vals)),
                       inner.bottom() - inner.height() * (v - lo) / (hi - lo)) for i, v in enumerate(self.vals)]
        col = QColor(WARN if self.clenching else ACCENT)
        fill = QLinearGradient(0, inner.top(), 0, inner.bottom())
        c1, c2 = QColor(col), QColor(col)
        c1.setAlpha(70)
        c2.setAlpha(0)
        fill.setColorAt(0, c1)
        fill.setColorAt(1, c2)
        area = QPolygonF([QPointF(pts[0].x(), inner.bottom())] + pts + [QPointF(pts[-1].x(), inner.bottom())])
        p.setPen(Qt.NoPen)
        p.setBrush(fill)
        p.drawPolygon(area)
        pen = QPen(col, 2.0)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPolyline(QPolygonF(pts))
        p.setPen(Qt.NoPen)
        halo = QColor(col)
        halo.setAlpha(60)
        p.setBrush(halo)
        p.drawEllipse(pts[-1], 7, 7)
        p.setBrush(col)
        p.drawEllipse(pts[-1], 3.4, 3.4)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self.big:
            return self.paint_big(p)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.setPen(QPen(QColor(LINE), 1))
        p.setBrush(QColor(PANEL2))
        p.drawRoundedRect(r, 10, 10)
        inner = r.adjusted(10, 7, -10, -7)
        if not self.live or len(self.vals) < 2:
            p.setPen(QPen(QColor(MUTED), 1, Qt.DashLine))
            p.drawLine(QPointF(inner.left(), inner.center().y()), QPointF(inner.right(), inner.center().y()))
            return
        lo, hi = math.log10(0.3), math.log10(300.0)
        n = self.vals.maxlen
        pts = [QPointF(inner.left() + inner.width() * (i + n - len(self.vals)) / (n - 1),
                       inner.bottom() - inner.height() * (v - lo) / (hi - lo)) for i, v in enumerate(self.vals)]
        pen = QPen(QColor(WARN if self.clenching else ACCENT), 1.6)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.drawPolyline(QPolygonF(pts))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(WARN if self.clenching else ACCENT))
        p.drawEllipse(pts[-1], 2.6, 2.6)


class Bridge(QObject):
    """Carries events from background threads to the screen (Qt queues them safely)."""
    event = Signal(dict)
    streams = Signal(list)


# =========================================================================== viewer pieces
GLYPH = {"up": "▲", "right": "▶", "down": "▼", "left": "◀", "back": "◀◀ 15", "fwd": "15 ▶▶"}
CAPTION = {"up": "Up", "down": "Down", "left": "Left", "right": "Right", "back": "Back", "fwd": "Forward"}
# Defaults = the best, well-spaced rates on the first real recording (wearer1: 92-100% each).
# Each profile can override them with settings["viewer_rates"].
FREQ = {"up": 13.0, "right": 9.0, "down": 15.0, "left": 11.0, "back": 9.0, "fwd": 13.0}
TABS = ["pdf", "audio", "video"]
TAB_ACTS = {"pdf": ["up", "right", "down", "left"], "audio": [], "video": ["back", "fwd"]}
PDF_KEYS = {Qt.Key_Up: "up", Qt.Key_Right: "right", Qt.Key_Down: "down", Qt.Key_Left: "left"}
VIDEO_KEYS = {Qt.Key_Left: "back", Qt.Key_Right: "fwd"}
TAB_KEYS = {Qt.Key_1: "pdf", Qt.Key_2: "audio", Qt.Key_3: "video"}
DOUBLE_MS, HOLD_MS, ZOOM_IN = 1500, 1000, 2.5
SEQ_GAP_MS = 1000  # Listen/Watch: how long to wait for another clench before acting


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
        if Flicker.style == "box":  # classic square: the whole tile flashes
            v = int(255 * level) if not self.off else 12
            p.setPen(QPen(QColor(LINE), 1))
            p.setBrush(QColor(v, v, v))
            p.drawRoundedRect(r, 18, 18)
        elif self.off:  # glow switched off: only a faint outline marks where it sits
            p.setPen(QPen(QColor(LINE), 1.2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(r.adjusted(r.width() * 0.08, r.height() * 0.08, -r.width() * 0.08, -r.height() * 0.08))
        if not self.off and Flicker.style == "glow":
            # stretch the soft glow into an oval that fills the whole tile: much more flickering
            # area (= stronger brain response) than a circle limited by the tile's short side
            rx, ry = r.width() * 0.5, r.height() * 0.5
            rad = min(rx, ry)
            p.save()
            p.translate(r.center())
            p.scale(rx / rad, ry / rad)
            paint_glow(p, QPointF(0, 0), rad, level)
            p.restore()
        if now < self.hit_until:  # selected: an accent ring (glow) or outline (box)
            p.setPen(QPen(QColor(ACCENT), 3))
            p.setBrush(QColor(245, 143, 124, 45))
            if Flicker.style == "box":
                p.drawRoundedRect(r, 18, 18)
            else:
                p.drawEllipse(r.adjusted(4, 4, -4, -4))
        arrow = QColor(ARROW)
        arrow.setAlphaF(0.25 if self.off else 1.0)
        p.setPen(arrow)
        p.setFont(QFont(FONT_DISPLAY, 34 if len(GLYPH[self.act]) == 1 else 21, QFont.Bold))
        p.drawText(r, Qt.AlignCenter, GLYPH[self.act])
        p.setFont(QFont(FONT, 11, QFont.DemiBold))
        p.setPen(QColor(255, 255, 255, 70 if self.off else 200))
        if r.height() < 200:  # wide, short tile: word to the right of the arrow
            p.drawText(QRectF(r.center().x() + 34, r.top(), 140, r.height()), Qt.AlignLeft | Qt.AlignVCenter,
                       CAPTION[self.act])
        else:
            p.drawText(r.adjusted(0, 70, 0, 0), Qt.AlignHCenter | Qt.AlignVCenter, CAPTION[self.act])
        # small rate label in a pill at the bottom
        label = "off" if self.off else f"{self.freq:g} Hz"
        p.setFont(QFont(MONO, 8, QFont.Medium))
        w = p.fontMetrics().horizontalAdvance(label) + 16
        pill = QRectF(r.center().x() - w / 2, r.bottom() - 26, w, 18)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(36, 35, 40, 200))
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
        self.name = Path(path).name
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
        self.name = Path(path).name
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
        x, y, s = self.shown
        if s <= 0:
            self.shown = [self.x, self.y, self.scale()]
            x, y, s = self.shown
        img = self.image(self.page)
        M, (iw, ih) = self.M, self.inner()
        page = QRectF(M - x, M - y, img.width() * s, img.height() * s)
        view = QRectF(M, M, iw, ih)
        shown = page.intersected(view)
        for i, a in ((18, 14), (10, 24), (5, 38), (2, 60)):  # soft shadow: the page floats
            p.setBrush(QColor(0, 0, 0, a))
            p.drawRoundedRect(shown.adjusted(-i, -i + 3, i, i + 3), 10, 10)
        p.save()
        p.setClipRect(view)
        p.drawImage(page, img)
        p.restore()
        name = self.name if len(self.name) <= 32 else self.name[:29] + "…"
        info = (f"{name}  ·  page {self.page + 1}/{self.pages}  ·  {'fit' if self.zoom == 1 else f'{self.zoom:g}×'}"
                f"  ·  arrows {'on' if self.arrows_on else 'off'}")
        p.setFont(QFont(MONO, 9, QFont.Medium))
        w = p.fontMetrics().horizontalAdvance(info) + 24
        box = QRectF(self.width() - w - M - 8, self.height() - M - 36, w, 26)
        p.setBrush(QColor(36, 35, 40, 225))
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
#segment {{ background:transparent; border:none; }}
QPushButton#seg {{ background:transparent; border:none; border-bottom:3px solid transparent; border-radius:0;
                   font-size:15pt; font-weight:700; padding:8px 14px 6px 14px; min-height:40px; color:{MUTED}; }}
QPushButton#seg:hover {{ color:{TEXT}; background:transparent; border-bottom:3px solid {LINE}; }}
QPushButton#seg:checked {{ background:transparent; color:{TEXT}; border-bottom:3px solid {ACCENT}; }}
QPushButton#bigctl {{ font-size:13.5pt; padding:8px 16px; border-radius:14px; min-height:48px;
                      background:{PANEL2}; border:1px solid {LINE}; }}
QPushButton#bigctl:hover {{ border-color:{ACCENT}; background:#444349; }}
QPushButton#bigctl:pressed {{ background:{ACCENT_SOFT}; }}
QPushButton#bigopen {{ font-size:13pt; padding:6px 18px; border-radius:10px; min-height:34px;
                       background:{MINT_GRADIENT}; color:{ON_ACCENT}; border:none; font-weight:700; }}
QPushButton#bigopen:hover {{ background:{ACCENT_HI}; }}
QPushButton#bighome {{ font-size:13pt; padding:10px 16px; border-radius:12px; min-height:40px;
                       background:transparent; border:1px solid transparent; color:{MUTED}; }}
QPushButton#bighome:hover {{ background:{PANEL2}; border:1px solid {LINE}; color:{TEXT}; }}
QPushButton#bighome:pressed, QPushButton#bighome:checked {{ background:{ACCENT_SOFT}; color:{TEXT}; }}
QPushButton#bighome::menu-indicator {{ width:0; }}
#bar QLabel#pill {{ background:transparent; border:none; color:{MUTED}; padding:4px 6px; }}
QLabel#filename {{ font-size:12.5pt; color:{TEXT}; background:transparent; padding:2px 4px; font-weight:600; }}
"""


def kind_of(path):
    ext = Path(path).suffix.lower()
    return "pdf" if ext in PDF_EXT else "audio" if ext in AUDIO_EXT else "video" if ext in VIDEO_EXT else None


HELP = {
    "pdf": ("Reading a PDF", [
        ("👀", "Look at a glowing arrow for 1–2 seconds", "The page scrolls that way. Past the edge, it turns the page."),
        ("😬", "Clench your jaw twice, quickly", "Turns the arrows off (read without flicker) — or back on."),
        ("⏱", "Hold a jaw clench for about 1 second", "Zooms in. Hold again to zoom out."),
        ("😌", "Relax your face while reading", "Frowning or tensing hard can count as a clench."),
    ]),
    "audio": ("Listening", [
        ("😬", "Clench your jaw once", "Play or pause (it waits about 1 second to see if more clenches follow)."),
        ("⏪", "Clench twice", "Back 15 seconds. Hold the 2nd clench to keep going back, every second."),
        ("⏩", "Clench three times", "Forward 15 seconds. Hold the 3rd clench to keep going forward."),
        ("😌", "Relax your face between clenches", "Tensing hard can count as a clench."),
    ]),
    "video": ("Watching", [
        ("👀", "Look at the glowing ⏪ (left) or ⏩ (right) for 1–2 s", "Skips back or forward 15 seconds."),
        ("😬", "Clench your jaw once", "Play or pause (it waits about 1 second to see if more clenches follow)."),
        ("⏪", "Clench twice", "Back 15 seconds. Hold the 2nd clench to keep going back, every second."),
        ("⏩", "Clench three times", "Forward 15 seconds. Hold the 3rd clench to keep going forward."),
    ]),
}
HELP_BLINK = {
    "pdf": ("Reading a PDF", [
        ("👀", "Look at a glowing arrow for 1–2 seconds", "The page scrolls that way. Past the edge, it turns the page."),
        ("😉", "Blink twice, quickly", "Turns the arrows off (read without flicker) — or back on."),
        ("😌", "Close your eyes for about 1 second", "Zooms in. Do it again to zoom out."),
        ("ℹ", "Normal blinking is fine", "Only two quick, deliberate blinks count (natural double blinks can occasionally trigger it)."),
    ]),
    "audio": ("Listening", [
        ("😉", "Blink twice, quickly", "Play or pause."),
        ("😌", "Close your eyes for about 1 second", "Go back 15 seconds."),
    ]),
    "video": ("Watching", [
        ("👀", "Look at the glowing ⏪ (left) or ⏩ (right) for 1–2 s", "Skips back or forward 15 seconds."),
        ("😉", "Blink twice, quickly", "Play or pause."),
        ("😌", "Close your eyes for about 1 second", "Go back 15 seconds."),
    ]),
}
HELP_ROLL = {
    "pdf": ("Reading a PDF (eye roll, experimental)", [
        ("👀", "Look at a glowing arrow for 1–2 seconds", "The page scrolls that way. Past the edge, it turns the page."),
        ("🙄", "Roll your eyes in one full circle", "Turns the arrows off (read without flicker) — or back on."),
        ("😌", "Close your eyes for about 1 second", "Zooms in. Do it again to zoom out."),
    ]),
    "audio": ("Listening (eye roll, experimental)", [
        ("🙄", "Roll your eyes in one full circle", "Play or pause."),
        ("😌", "Close your eyes for about 1 second", "Go back 15 seconds."),
    ]),
    "video": ("Watching (eye roll, experimental)", [
        ("👀", "Look at the glowing ⏪ (left) or ⏩ (right) for 1–2 s", "Skips back or forward 15 seconds."),
        ("🙄", "Roll your eyes in one full circle", "Play or pause."),
        ("😌", "Close your eyes for about 1 second", "Go back 15 seconds."),
    ]),
}
HELP_COMMON = [
    ("🖱", "Click instead", "Click any glowing arrow tile — for helpers, or as a backup."),
    ("📂", "Open files (or drag files / a folder onto the window)",
     "Bring in as many PDFs, audio files and videos as you like; pick one from the list at the top. "
     "Audio and video move on to the next file by themselves."),
    ("⌨", "Keyboard backup", "1 / 2 / 3 = Read / Listen / Watch · Arrow keys = look at an arrow · "
                                 "J = jaw (tap 1, 2 or 3 times; hold)."),
]


class Toast(QLabel):
    """A short message in a floating pill. It is its own little window, so it is drawn above
    video (which Windows puts on its own layer); the pill is painted here because style-sheet
    backgrounds do not show on a see-through window."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet(f"background:transparent; border:none; color:{TEXT}; font-size:13pt;"
                           f"padding:11px 24px;")

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(ACCENT), 1.5))
        p.setBrush(QColor(51, 50, 56, 245))
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        p.end()
        super().paintEvent(e)


class HelpOverlay(QFrame):
    """Large, easy-to-read instructions for the current tab, over a dimmed screen."""
    shown = Signal(bool)  # the video hides while this is open (Windows draws video above widgets)

    def showEvent(self, e):
        super().showEvent(e)
        self.shown.emit(True)

    def hideEvent(self, e):
        super().hideEvent(e)
        self.shown.emit(False)

    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet(f"""
            HelpOverlay {{ background: rgba(24, 23, 27, 225); }}
            #helpcard {{ background:{PANEL}; border:1px solid {LINE}; border-radius:18px; }}
            #helpcard QLabel {{ background:transparent; }}
            #helptitle {{ font-family:"{FONT_DISPLAY}", "Segoe UI"; font-size:20pt; font-weight:700; }}
            #helpicon {{ font-family:"{MONO}"; font-size:15pt; color:{ACCENT}; border:1.5px solid {LINE};
                         border-radius:20px; min-width:40px; max-width:40px; min-height:40px; max-height:40px;
                         qproperty-alignment:AlignCenter; }}
            #helpdo {{ font-size:14pt; font-weight:600; color:{TEXT}; }}
            #helpwhat {{ font-size:12pt; color:{MUTED}; }}
            #helpsep {{ background:{LINE}; max-height:1px; min-height:1px; }}
            #helpbody {{ background:transparent; }}
        """)
        outer = QVBoxLayout(self)
        outer.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        self.card = QFrame(objectName="helpcard")
        self.card.setMaximumWidth(820)
        self.card_lay = QVBoxLayout(self.card)
        self.card_lay.setContentsMargins(34, 28, 34, 26)
        self.body, self.shown_for = None, None
        row.addWidget(self.card, 6)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)

    def mousePressEvent(self, e):
        if not self.card.geometry().contains(e.position().toPoint()):
            self.hide()  # click outside the card to close

    def _clear(self):
        """Throw away the old instructions in one piece (rows hold nested layouts, so deleting
        item by item used to leave stray labels piled on top of the new ones)."""
        if self.body is not None:
            self.card_lay.removeWidget(self.body)
            self.body.hide()
            self.body.deleteLater()
        self.body = QWidget(objectName="helpbody")
        self.lay = QVBoxLayout(self.body)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(12)
        self.card_lay.addWidget(self.body)

    SYMBOL = {"👀": "◉", "😬": "J", "⏱": "◷", "😌": "~", "⏪": "«", "⏩": "»", "😉": "◡", "🙄": "↻",
              "🖱": "↖", "📂": "+", "⌨": "⌨", "ℹ": "i"}

    def _row(self, icon, do, what):
        icon = self.SYMBOL.get(icon, icon)
        h = QHBoxLayout()
        h.setSpacing(16)
        h.addWidget(QLabel(icon, objectName="helpicon"), alignment=Qt.AlignTop)
        v = QVBoxLayout()
        v.setSpacing(2)
        v.addWidget(QLabel(do, objectName="helpdo", wordWrap=True))
        v.addWidget(QLabel(what, objectName="helpwhat", wordWrap=True))
        h.addLayout(v, 1)
        self.lay.addLayout(h)

    def set_tab(self, tab, mode="jaw"):
        if self.shown_for == (tab, mode):  # already showing these instructions
            return
        self.shown_for = (tab, mode)
        self._clear()
        title, rows = {"blink": HELP_BLINK, "roll": HELP_ROLL}.get(mode, HELP)[tab]
        self.lay.addWidget(QLabel(f"How to use — {title}", objectName="helptitle"))
        for r in rows:
            self._row(*r)
        self.lay.addWidget(QFrame(objectName="helpsep"))
        for r in HELP_COMMON:
            self._row(*r)
        ok = QPushButton("Got it", objectName="primary")
        ok.setMinimumHeight(48)
        ok.setStyleSheet("font-size:14pt;")
        ok.setFocusPolicy(Qt.NoFocus)
        ok.clicked.connect(self.hide)
        self.lay.addSpacing(6)
        self.lay.addWidget(ok)


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
        self.library = {k: [v] for k, v in self.files.items()}  # every file brought in, per tab
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
        for i, label in enumerate(["Read", "Listen", "Watch"]):
            b = QPushButton(label, objectName="seg", checkable=True)
            self.tab_buttons.addButton(b, i)
            sl.addWidget(b)
        top.addWidget(segment)
        self.tab_buttons.button(0).setChecked(True)
        self.tab_buttons.idClicked.connect(lambda i: self.set_tab(TABS[i]))
        top.addSpacing(8)
        open_btn = QPushButton("+  Open files", objectName="bigopen")
        open_btn.setToolTip("Open as many PDFs, audio files and videos as you like "
                            "(or drag files or a whole folder onto this window)")
        open_btn.clicked.connect(self.open_dialog)
        top.addWidget(open_btn)
        self.lib_box = QComboBox()  # this tab's files
        self.lib_box.setMinimumWidth(170)
        self.lib_box.setMaximumWidth(300)
        self.lib_box.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.lib_box.setToolTip("Files on this tab — pick one to open it")
        self.lib_box.setFocusPolicy(Qt.NoFocus)
        self.lib_box.activated.connect(lambda i: self.pick_file(i))
        top.addWidget(self.lib_box)
        help_btn = QPushButton("?  How to use", objectName="bighome")
        help_btn.clicked.connect(lambda: self.show_help(not self.help.isVisible()))
        top.addWidget(help_btn)
        top.addStretch(1)
        self.strip = SignalStrip()
        top.addWidget(self.strip, alignment=Qt.AlignVCenter)
        self.status = QLabel(objectName="pill")
        self.status.setMinimumWidth(1)
        self.status.setMaximumWidth(420)
        top.addWidget(self.status, alignment=Qt.AlignVCenter)
        self.gear = QPushButton("Settings", objectName="bighome")
        self.gear.setToolTip("Flicker settings")
        top.addWidget(self.gear)
        for w in bar.findChildren(QPushButton):
            w.setFocusPolicy(Qt.NoFocus)

        self.targets = {a: FlickerTarget(a) for a in FREQ}
        for t in self.targets.values():
            t.clicked.connect(self.act)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.build_pdf(pdf_path))
        self.stack.addWidget(self.build_audio())
        self.stack.addWidget(self.build_video())

        # ---- flicker comfort settings live in the gear menu (out of the way, more room for content)
        settings = QWidget()
        settings.setStyleSheet(f"background:{PANEL2};")
        sv = QVBoxLayout(settings)
        sv.setContentsMargins(14, 10, 14, 10)
        sv.setSpacing(10)
        al = QHBoxLayout()
        al.addWidget(QLabel("Action input"))
        self.mode_box = QComboBox()
        self.mode_box.addItems(["Jaw clench", "Blinks", "Eye roll (experimental)"])
        self.mode_box.setToolTip("Blinks: double blink = the main action, close eyes ~1 s = the 'hold' action")
        self.mode_box.setFocusPolicy(Qt.NoFocus)
        self.mode_box.currentIndexChanged.connect(lambda i: self.set_mode(["jaw", "blink", "roll"][i]))
        al.addWidget(self.mode_box, 1)
        sv.addLayout(al)
        fl = QHBoxLayout()
        sv.addLayout(fl)
        self.hint = QLabel(objectName="hint")  # (instructions now live in the How-to-use card)
        fl.addWidget(QLabel("Flicker strength"))
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
        menu = QMenu(self)
        act = QWidgetAction(menu)
        act.setDefaultWidget(settings)
        menu.addAction(act)
        self.gear.setMenu(menu)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(bar)
        lay.addWidget(self.stack, 1)

        self.help = HelpOverlay(self)
        self.help.hide()
        self.help.shown.connect(lambda on: self.vw.setVisible(not on))
        self.help_shown_once = False
        self.mode = "jaw"

        self.toast_label = Toast(self)
        self.toast_label.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.toast_label.setAttribute(Qt.WA_TranslucentBackground)
        self.toast_label.setAttribute(Qt.WA_ShowWithoutActivating)
        self.toast_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.toast_label.hide()
        self.toast_timer = QTimer(self, singleShot=True, timeout=self.toast_label.hide)
        self.single_timer = QTimer(self, singleShot=True, timeout=self.audio_back)
        self.hold_timer = QTimer(self, singleShot=True, timeout=self.j_hold_fired)
        self.seq_n, self.seq_last_end, self.seq_holding, self.repeat_dir = 0, -1e9, False, 0
        self.seq_timer = QTimer(self, singleShot=True, timeout=self.seq_commit)
        self.repeat_timer = QTimer(self, interval=1000, timeout=lambda: self.media_skip(self.repeat_dir))

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
        v.setContentsMargins(10, 8, 10, 10)
        v.setSpacing(8)
        self.pdf_name = QLabel()  # the file name now shows in the page's info pill
        g = QGridLayout()
        g.setSpacing(8)
        self.pdf = PdfView(pdf_path)

        # ◀ / ▶ tall side tiles (the video's size); ▲ / ▼ wide tiles on top and bottom
        def wide(tile):
            tile.setFixedHeight(140)
            tile.setMinimumWidth(420)
            tile.setMaximumWidth(700)
            box = QWidget()
            h = QHBoxLayout(box)
            h.setContentsMargins(0, 0, 0, 0)
            h.addStretch(1)
            h.addWidget(tile, 4)
            h.addStretch(1)
            return box

        for t in ("left", "right"):
            self.targets[t].setFixedWidth(160)
        g.addWidget(wide(self.targets["up"]), 0, 1)
        g.addWidget(self.targets["left"], 1, 0)
        g.addWidget(self.pdf, 1, 1)
        g.addWidget(self.targets["right"], 1, 2)
        g.addWidget(wide(self.targets["down"]), 2, 1)
        g.setRowStretch(1, 1)
        g.setColumnStretch(1, 1)
        v.addLayout(g, 1)
        self.btn_arrows = QPushButton()  # (the button row under the PDF was removed to give the page more room)
        return w

    def build_audio(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(12, 8, 12, 12)
        self.audio_name = QLabel(objectName="filename", alignment=Qt.AlignCenter)
        self.audio_name.setStyleSheet("font-size:20pt; font-weight:700;")
        v.addStretch(1)
        self.a_icon = QLabel("▶", objectName="big_icon", alignment=Qt.AlignCenter)
        self.a_time = QLabel("0:00", alignment=Qt.AlignCenter)
        self.a_time.setStyleSheet(f"font-family:'{MONO}'; font-size:15pt; color:{MUTED};")
        self.a_bar = QSlider(Qt.Horizontal, enabled=False)
        self.a_bar.setFixedWidth(560)
        v.addWidget(self.a_icon, alignment=Qt.AlignCenter)
        v.addSpacing(22)
        v.addWidget(self.audio_name, alignment=Qt.AlignCenter)
        v.addSpacing(14)
        v.addWidget(self.a_bar, alignment=Qt.AlignCenter)
        v.addWidget(self.a_time)
        v.addStretch(1)
        self.btn_audio_play = QPushButton()  # (button row removed: jaw / keyboard control playback)
        self.audio = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.audio.setAudioOutput(self.audio_out)
        self.audio.positionChanged.connect(self.audio_progress)
        self.audio.playbackStateChanged.connect(self.audio_state)
        self.audio.mediaStatusChanged.connect(
            lambda st: st == QMediaPlayer.EndOfMedia and self.next_file("audio"))
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
        keep = vw.sizePolicy()
        keep.setRetainSizeWhenHidden(True)  # hiding it (under the help card) must not reflow the tab
        vw.setSizePolicy(keep)
        self.vw = vw
        self.video = QMediaPlayer(self)
        self.video_out = QAudioOutput(self)
        self.video.setAudioOutput(self.video_out)
        self.video.setVideoOutput(vw)
        self.video.playbackStateChanged.connect(self.video_state)
        self.video.mediaStatusChanged.connect(
            lambda st: st == QMediaPlayer.EndOfMedia and self.next_file("video"))
        h.addWidget(self.targets["back"])
        h.addWidget(vw, 1)
        h.addWidget(self.targets["fwd"])
        v.addLayout(h, 1)
        self.btn_video_play = QPushButton()  # (button row removed: flicker / jaw / keyboard control playback)
        self.load_video(self.files["video"])
        return w

    # ---------- files: each tab keeps a list; open any number at once
    def open_dialog(self):
        start = str(Path(self.files.get(self.tab) or Path.home()).parent)
        paths, _ = QFileDialog.getOpenFileNames(self, "Open PDFs, audio or video (select as many as you like)",
                                                start, OPEN_FILTER)
        if paths:
            self.open_paths(paths)

    def open_paths(self, paths):
        """Add many files (and folders) to the right lists, open the first new one."""
        found = []
        for p in paths:
            p = Path(p)
            if p.is_dir():
                found += sorted(str(f) for f in p.rglob("*") if f.is_file() and kind_of(f))
            elif kind_of(p):
                found.append(str(p))
        if not found:
            QMessageBox.warning(self, "Nothing to open", "None of those are PDF, audio or video files.")
            return
        added = {"pdf": 0, "audio": 0, "video": 0}
        for f in found:
            lst = self.library[kind_of(f)]
            if f not in lst:
                lst.append(f)
                added[kind_of(f)] += 1
        self.save_library()
        self.open_path(found[0])
        summary = ", ".join(f"{n} {k if k != 'pdf' else 'PDF'}{'s' if n != 1 and k == 'pdf' else ''}"
                            for k, n in added.items() if n)
        if len(found) > 1:
            self.toast(f"Added {summary}" if summary else "Already in your lists")

    def save_library(self):
        self.main.profile.settings["library"] = {k: list(v) for k, v in self.library.items()}
        self.main.profile.save()

    def refresh_lib_box(self):
        lst, cur = self.library[self.tab], self.files.get(self.tab)
        self.lib_box.blockSignals(True)
        self.lib_box.clear()
        for i, f in enumerate(lst, 1):
            name = Path(f).name
            name = name if len(name) <= 26 else name[:16] + "…" + name[-8:]
            self.lib_box.addItem(f"{i}/{len(lst)}  {name}", f)
        if cur in lst:
            self.lib_box.setCurrentIndex(lst.index(cur))
        self.lib_box.setVisible(bool(lst))
        self.lib_box.blockSignals(False)

    def pick_file(self, i):
        lst = self.library[self.tab]
        if 0 <= i < len(lst):
            self.open_path(lst[i])

    def next_file(self, kind):
        """Audio / video: when one file ends, play the next one in the list."""
        lst, cur = self.library[kind], self.files.get(kind)
        if cur in lst and lst.index(cur) + 1 < len(lst):
            nxt = lst[lst.index(cur) + 1]
            {"audio": self.load_audio, "video": self.load_video}[kind](nxt)
            self.main.profile.settings.setdefault("last_files", {})[kind] = nxt
            (self.audio if kind == "audio" else self.video).play()
            if self.tab == kind:
                self.refresh_lib_box()
            self.toast(f"Next: {Path(nxt).name}")

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
        if str(path) not in self.library[kind]:
            self.library[kind].append(str(path))
            self.save_library()
        if remember:
            self.main.profile.settings.setdefault("last_files", {})[kind] = str(path)
            self.main.profile.save()
        self.set_tab(kind)
        self.toast(f"Opened {Path(path).name}")
        return True

    def load_pdf(self, path):
        self.pdf.load(path)
        self.files["pdf"] = str(path)
        self.pdf_name.setText(Path(path).name)

    def load_audio(self, path):
        self.audio.stop()
        self.audio.setSource(QUrl.fromLocalFile(str(path)))
        self.audio.setLoops(QMediaPlayer.Infinite if "sample_audio" in str(path) else 1)
        self.files["audio"] = str(path)
        self.audio_name.setText(Path(path).name)
        if hasattr(self, "lib_box") and self.tab == "audio":
            self.refresh_lib_box()

    def load_video(self, path):
        self.video.stop()
        self.video.setSource(QUrl.fromLocalFile(str(path)))
        self.files["video"] = str(path)
        self.video_name.setText(Path(path).name)
        if hasattr(self, "lib_box") and self.tab == "video":
            self.refresh_lib_box()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(kind_of(u.toLocalFile()) or Path(u.toLocalFile()).is_dir()
                                          for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.open_paths([u.toLocalFile() for u in e.mimeData().urls()])

    # ---------- page life
    def apply_rates(self):
        """Flicker rates come from the profile (defaults = what worked best on real data)."""
        for act, f in self.main.profile.settings.get("viewer_rates", {}).items():
            if act in FREQ:
                FREQ[act] = float(f)
                self.targets[act].freq = float(f)
                self.targets[act].update()

    def set_mode(self, mode, announce=True):
        """Action input: 'jaw' (clench) or 'blink' (double blink + eyes closed ~1 s)."""
        self.mode = mode
        self.main.profile.settings["action_input"] = mode
        # only detect what this mode uses, so other detectors can't steal or invent events
        self.main.engine.set_inputs({"jaw": {"ssvep", "jaw"}, "blink": {"ssvep", "blink", "eyes_closed"},
                                     "roll": {"ssvep", "eye_roll", "eyes_closed"}}[mode])
        self.mode_box.blockSignals(True)
        self.mode_box.setCurrentIndex(["jaw", "blink", "roll"].index(mode))
        self.mode_box.blockSignals(False)
        if self.help.isVisible():
            self.help.set_tab(self.tab, mode)
        if announce:
            self.toast({"jaw": "Action input: jaw clench",
                        "blink": "Action input: blinks (double blink · close eyes 1 s)",
                        "roll": "Action input: eye roll, experimental (roll · close eyes 1 s)"}[mode])

    def activate(self):
        s = self.main.profile.settings
        self.set_mode(s.get("action_input", "jaw"), announce=False)
        self.apply_rates()
        saved = s.get("library") or {}
        for kind in self.library:
            keep = [f for f in saved.get(kind, []) if Path(f).exists()]
            for f in keep:
                if f not in self.library[kind]:
                    self.library[kind].append(f)
        for kind, path in (s.get("last_files") or {}).items():
            if path and Path(path).exists() and path != self.files.get(kind):
                self.open_path(path, remember=False)
        self.pdf_name.setText(Path(self.files['pdf']).name)
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(s["flicker_strength"] * 100)))
        self.slider.blockSignals(False)
        self.strength_label.setText(f"{self.slider.value()}%")
        self.style_box.blockSignals(True)
        self.style_box.setCurrentIndex(0 if s["flicker_style"] == "glow" else 1)
        self.style_box.blockSignals(False)
        self.set_tab(self.tab)
        if not self.help_shown_once:  # first time the viewer opens: show the instructions
            self.help_shown_once = True
            self.show_help(True)

    def show_help(self, on):
        self.help.set_tab(self.tab, self.mode)
        self.help.setGeometry(self.rect())
        self.help.setVisible(on)
        if on:
            self.help.raise_()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.help.setGeometry(self.rect())

    def deactivate(self):
        self.toast_label.hide()
        self.main.engine.set_inputs(None)
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
        # it is a floating window, so only show it while this app is the one in front
        if not self.isVisible() or not self.window().isActiveWindow():
            return
        self.toast_label.setText(msg)
        self.toast_label.adjustSize()
        # over the page / video itself, never over a flicker target (that would disturb its signal)
        area = {"pdf": self.pdf, "video": self.vw}.get(self.tab, self.stack)
        if not area.isVisible():
            area = self.stack
        top = 30 if area is self.pdf else 16
        pos = area.mapToGlobal(QPoint((area.width() - self.toast_label.width()) // 2, top))
        self.toast_label.move(pos)
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
            "audio": "Jaw: clench once = play / pause · hold 1 s = back 15 s",
            "video": "Look at a glowing arrow to skip 15 s · jaw: clench = play / pause, hold = back 15 s"}[name])
        if self.mode == "blink":
            self.hint.setText("Blinks: blink twice = main action · close eyes ~1 s = hold action")
        elif self.mode == "roll":
            self.hint.setText("Eye roll: roll = main action · close eyes ~1 s = hold action")
        self.seq_timer.stop()
        self.repeat_timer.stop()
        self.seq_n = 0
        self.refresh_lib_box()
        if self.help.isVisible():
            self.help.set_tab(name, self.mode)
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
        elif self.tab == "audio":  # one clench = play / pause
            self.toast(f"Clench → {self.toggle(self.audio)}")
        else:
            self.toast(f"Clench → {self.toggle(self.video)}")

    def blink_action(self, label="Double blink"):
        """Blink / eye-roll mode: the main action (what one clench / a double clench does)."""
        if self.tab == "pdf":
            self.set_arrows(not self.pdf.arrows_on)
        elif self.tab == "audio":
            self.toast(f"{label} → {self.toggle(self.audio)}")
        else:
            self.toast(f"{label} → {self.toggle(self.video)}")

    # ---------- Listen / Watch jaw patterns: 1 = play/pause, 2 = back 15 s, 3 = forward 15 s,
    #            holding the 2nd / 3rd clench keeps skipping every second until you let go
    def media_player(self):
        return self.audio if self.tab == "audio" else self.video

    def media_skip(self, d):
        self.skip(self.media_player(), 15000 * d)
        self.toast("⏩  Forward 15 s" if d > 0 else "⏪  Back 15 s")

    def media_start(self):
        now = time.monotonic()
        if now - self.seq_last_end > SEQ_GAP_MS / 1000:
            self.seq_n = 0
        self.seq_timer.stop()
        self.seq_n += 1
        self.seq_holding = False

    def media_hold(self):
        if self.seq_n >= 2:  # holding the 2nd (back) or 3rd (forward) clench: keep skipping
            self.seq_holding = True
            self.repeat_dir = -1 if self.seq_n == 2 else 1
            self.media_skip(self.repeat_dir)
            self.repeat_timer.start()

    def media_release(self):
        self.seq_last_end = time.monotonic()
        if self.seq_holding:
            self.repeat_timer.stop()
            self.seq_holding, self.seq_n = False, 0
            return
        if self.seq_n >= 3:
            self.seq_n = 0
            self.media_skip(+1)
            return
        self.seq_timer.start(SEQ_GAP_MS)  # wait: is another clench coming?
        self.toast("1 clench…  (2 = back 15 s · 3 = forward 15 s)" if self.seq_n == 1 else
                   "2 clenches…  (one more = forward 15 s)")

    def seq_commit(self):
        n, self.seq_n = self.seq_n, 0
        if n == 1:
            self.toast(f"Clench → {self.toggle(self.media_player())}")
        elif n == 2:
            self.media_skip(-1)

    def blink_hold(self):
        """Blink mode, eyes closed ~1 s = the 'hold' action."""
        if self.tab == "pdf":
            self.toast("Eyes closed → zoom in" if self.pdf.toggle_zoom() else "Eyes closed → zoom out")
        elif self.tab == "audio":
            self.audio_back()
        else:
            self.skip(self.video, -15000)
            self.toast("Eyes closed → back 15 s")

    def hold(self):
        self.last_clench = 0.0
        if self.tab == "pdf":
            self.toast("Held clench → zoom in" if self.pdf.toggle_zoom() else "Held clench → zoom out")
        elif self.tab == "audio":
            self.skip(self.audio, -15000)
            self.toast("Held clench → back 15 s")
        else:
            self.skip(self.video, -15000)
            self.toast("Held clench → back 15 s")

    def on_event(self, m):
        t = m["type"]
        if t == "status":
            self.status.setText(status_html(m, compact=True))
            self.strip.push(m)
        elif not self.isVisible():
            return
        elif self.mode == "jaw" and self.tab in ("audio", "video") and t in (
                "clench_start", "hold", "clench", "clench_end"):
            {"clench_start": self.media_start, "hold": self.media_hold,
             "clench": self.media_release, "clench_end": self.media_release}[t]()
        elif t == "clench" and self.mode == "jaw":
            self.clench()
        elif t == "hold" and self.mode == "jaw":
            self.hold()
        elif t == "double_blink" and self.mode == "blink":
            self.blink_action()
        elif t == "eye_roll" and self.mode == "roll":
            self.blink_action("Eye roll")
        elif t == "eyes_closed" and self.mode in ("blink", "roll"):
            self.blink_hold()
        elif t == "ssvep":
            for a in TAB_ACTS[self.tab]:
                if FREQ[a] == m["freq"]:
                    self.act(a)
                    self.toast(f"Looked at {m['freq']:g} Hz")

    def handle_key(self, e):
        press = e.type() == QEvent.KeyPress
        key, repeat = e.key(), e.isAutoRepeat()
        if self.help.isVisible() and key in (Qt.Key_Escape, Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            if press:
                self.show_help(False)
            return True
        if key in TAB_KEYS:  # 1 / 2 / 3 = Read / Listen / Watch
            if press and not repeat and TAB_KEYS[key] != self.tab:
                self.set_tab(TAB_KEYS[key])
            return True
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
                    if self.tab in ("audio", "video"):
                        self.media_start()
                    self.hold_timer.start(HOLD_MS)
            elif not press and self.j_down:
                self.j_down = False
                if self.j_via_eeg:
                    fake.stop("clench")
                else:
                    self.hold_timer.stop()
                    if self.tab in ("audio", "video"):
                        self.media_release()
                    elif not self.j_held:
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
            if self.tab in ("audio", "video"):
                self.media_hold()
            else:
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
        bg.setColorAt(0, QColor("#333238"))
        bg.setColorAt(1, QColor("#252429"))
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
            chip_color = QColor(ACCENT if (self.phase == "action" and not self.listening) else "#F2C4CE" if self.listening else MUTED)
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
        p.setFont(QFont(MONO, 12))
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
                p.setBrush(QColor(ACCENT if over else "#5F5E64"))
                p.drawRoundedRect(QRectF(min(zero, xv), y + 7, max(2.0, abs(xv - zero)), 6), 3, 3)
                p.setPen(QColor(TEXT))
                p.drawText(QRectF(x0 + w + 6, y, 54, 20), Qt.AlignVCenter | Qt.AlignLeft,
                           f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}")
            p.setPen(QPen(QColor(TEXT), 2))
            for t in thrs:
                xt = self._pos(t, scale, lo, hi, x0, w)
                p.drawLine(QPointF(xt, y + 3), QPointF(xt, y + 17))


EVENT_TILES = [("clench", "Jaw clench"), ("hold", "Jaw hold"), ("brow", "Eyebrows"), ("blink", "Blink"),
               ("double_blink", "Double blink"), ("glance_left", "Glance ←"), ("glance_right", "Glance →"),
               ("eyes_closed", "Eyes closed"), ("fist_right", "Right fist"), ("fist_left", "Left fist")] + \
              [("eye_roll", "Eye roll (exp.)")] + [(f"ssvep_{f:g}", f"Flicker {f:g}") for f in C.SSVEP_FREQS]


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
            f"background:{ACCENT if lit else PANEL}; color:{ON_ACCENT if lit else TEXT}; border:1px solid {LINE};"
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
        self.status.setMinimumWidth(1)
        top.addWidget(self.status)

        # ---- left: what to run
        left = QFrame(objectName="panel")
        left.setFixedWidth(300)
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("Inputs to include", objectName="h2"))
        self.checks = {}
        for key, (label, secs) in C.INPUTS.items():
            cb = QCheckBox(f"{label}   ~{secs} s")
            cb.setChecked(key != "eye_roll")  # experimental: only when asked for
            cb.setFocusPolicy(Qt.NoFocus)
            cb.toggled.connect(self.update_total)
            self.checks[key] = cb
            lv.addWidget(cb)
        self.total = QLabel(objectName="hint")
        lv.addWidget(self.total)
        lv.addSpacing(8)
        self.btn_quick = QPushButton("Quick: flicker · blink · jaw", objectName="primary")
        self.btn_quick.setToolTip("Only the flicker rates the viewer uses, then blinks, then jaw clench "
                                  "(+ eye roll when that's the action input), in that order (~4 min)")
        self.btn_quick.clicked.connect(self.start_quick)
        self.btn_cal = QPushButton("Full calibration (ticked inputs)")
        self.btn_roll = QPushButton("Eye roll calibration (experimental)")
        self.btn_roll.setToolTip("Calibrates only the experimental eye-roll input (~1 min). "
                                 "It isn't used by the viewer yet.")
        self.btn_roll.clicked.connect(self.start_roll)
        self.btn_redo = QPushButton("Redo Poor inputs only")
        self.btn_redo.setToolTip("Tick only the inputs whose latest calibration had a Poor result, then calibrate them")
        self.btn_test = QPushButton("Run live test")
        self.btn_stop = QPushButton("Stop")
        for b in (self.btn_quick, self.btn_cal, self.btn_roll, self.btn_redo, self.btn_test, self.btn_stop):
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
        self.main.engine.set_inputs(None)  # calibration and the live test use every input
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
        self.btn_quick.setEnabled(not self.rec)
        self.btn_roll.setEnabled(not self.rec)

    def poor_inputs(self):
        """Inputs whose most recent CALIBRATION had any Poor item."""
        return sorted({r["input"] for r in self.main.profile.calibration.values()
                       if r.get("kind", "").startswith("calibration") and "partial" not in r.get("kind", "")
                       and r.get("verdict") == "Poor" and r.get("input") in C.INPUTS})

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
    def viewer_rates(self):
        return sorted({float(f) for f in self.main.profile.settings.get("viewer_rates", {}).values()})

    def start_roll(self):
        """Experimental: calibrate only the eye roll."""
        for k, cb in self.checks.items():
            cb.setChecked(k == "eye_roll")
        self.start_session("calibration")

    def quick_inputs(self):
        extra = ["eye_roll"] if self.main.profile.settings.get("action_input") == "roll" else []
        return ["ssvep", "blink", "jaw"] + extra

    def start_quick(self):
        """Only what the viewer uses: its flicker rates, then blinks, then jaw (+ eye roll if that's
        the chosen action input), in that order."""
        for k, cb in self.checks.items():
            cb.setChecked(k in self.quick_inputs())
        self.start_session("calibration", quick=True)

    def start_session(self, kind, quick=False):
        eng = self.main.engine
        if not eng.status.get("connected"):
            QMessageBox.warning(self, "No headset", "Connect a headset on the start screen first.")
            return
        inputs = self.quick_inputs() if quick else self.selected()
        self.kind, self.quick = kind, quick
        self.seed = random.randrange(1_000_000)  # saved with the trial, so the order can be reproduced
        freqs = self.viewer_rates() if quick else C.SSVEP_FREQS
        self.steps = (C.build_calibration(inputs, seed=self.seed, ssvep_freqs=freqs, keep_order=quick)
                      if kind == "calibration" else C.build_test(inputs, reps=3, seed=self.seed))
        self.total_s = C.total_seconds(self.steps)
        self.rec = TrialRecorder(f"{kind}_quick" if quick else kind, self.main.profile.name, eng.info)
        eng.set_freqs(freqs)
        eng.set_recorder(self.rec)
        self.i, self.cur_t0, self.elapsed_before = -1, None, 0.0
        self.voice_warning = ""
        for b, on in ((self.btn_quick, False), (self.btn_cal, False), (self.btn_roll, False), (self.btn_redo, False),
                      (self.btn_test, False), (self.btn_stop, True)):
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
                                        "quick": getattr(self, "quick", False),
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

    def __init__(self, kicker, title, desc, on_click):
        super().__init__(objectName="card")
        self.on_click = on_click
        self.setAttribute(Qt.WA_Hover)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(140)
        h = QHBoxLayout(self)
        h.setContentsMargins(28, 22, 28, 22)
        h.setSpacing(20)
        v = QVBoxLayout()
        v.setSpacing(6)
        v.addWidget(QLabel(kicker.upper(), objectName="cardkicker"))
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
        tag.setWordWrap(True)
        titles.addWidget(tag)
        header.addLayout(titles)
        header.addStretch(1)
        self.strip = SignalStrip(big=True)
        self.conn_pill = QLabel(objectName="pill")
        self.conn_pill.setMinimumWidth(1)      # long stream names must never widen the window
        self.conn_pill.setMaximumWidth(460)
        header.addWidget(self.conn_pill, alignment=Qt.AlignVCenter)
        outer.addLayout(header)
        outer.addWidget(self.strip)

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
        self.chips = QGridLayout()
        self.chips.setHorizontalSpacing(8)
        self.chips.setVerticalSpacing(8)
        pv.addLayout(self.chips)
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
        self.conn.setMinimumWidth(1)
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
        choices.addWidget(ChoiceCard("First, for each person", "Calibrate & test",
                                     "Guided calibration of every signal, a scored live test and live meters — "
                                     "saved to this person's profile.", lambda: self.main.go("calib")), 1)
        choices.addWidget(ChoiceCard("Read · Listen · Watch", "Use the viewer",
                                     "Read PDFs, listen to audio and watch video hands-free, "
                                     "with this person's settings.", lambda: self.main.go("viewer")), 1)
        outer.addLayout(choices)
        outer.addStretch(1)
        foot = QHBoxLayout()
        self.data_label = QLabel(objectName="hint", wordWrap=True)
        self.data_label.setMinimumWidth(1)
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
            self.profile_info.setText(f"Calibrated {last[:16].replace('T', ' ')} · "
                                      f"{counts['Good']} good · {counts['OK']} ok · {counts['Poor']} poor"
                                      f" — best signals:")
            best = {}
            for r in cal.values():  # one chip per item, its best score
                if r["item"] not in best or r.get("accuracy", 0) > best[r["item"]].get("accuracy", 0):
                    best[r["item"]] = r
            ranked = sorted(best.values(), key=lambda r: -r.get("accuracy", 0))[:8]
        else:
            self.profile_info.setText("Not calibrated yet. Default settings are used until you calibrate.")
            ranked = []
        while self.chips.count():
            w = self.chips.takeAt(0).widget()
            if w:
                w.deleteLater()
        for i, r in enumerate(ranked):
            c = QLabel(f"<span style='color:{VERDICT_COLOR.get(r['verdict'], MUTED)}'>●</span>&nbsp; {r['item']}"
                       f"&nbsp; <b>{r.get('accuracy', 0) * 100:.0f}%</b>", objectName="chip")
            c.setToolTip(f"{r['verdict']} — {r.get('kind', 'calibration')} on {r.get('date', '')[:10]}")
            self.chips.addWidget(c, i // 2, i % 2)
        self.data_label.setText(f"Profiles and recordings are saved in Documents › {data_dir().name}")
        self.data_label.setToolTip(str(data_dir()))
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
        self.strip.push(m)
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


def load_fonts():
    """Bundled open-source fonts (fonts/*.ttf, OFL licence), so the app looks the same offline."""
    from PySide6.QtGui import QFontDatabase
    for f in sorted((HERE / "fonts").glob("*.ttf")):
        QFontDatabase.addApplicationFont(str(f))
    QApplication.setFont(QFont(FONT, 10))


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
    load_fonts()
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
