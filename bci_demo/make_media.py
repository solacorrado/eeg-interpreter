"""Create the demo's sample PDF, audio and video (run once).

  uv run python bci_demo/make_media.py

PDF: 5 pages of large text, to practise scrolling, zooming and page turns
(skipped if bci_demo/media/sample.pdf already exists; any PDF works, or pass
--pdf to the server). Audio: a 40-second looping tune (WAV). Video: 2 minutes
with a big timer on screen, so the 15-second skips are easy to see. Replace
them with your own .wav and .mp4 files (same names in static/media/).
"""
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "static" / "media"
OUT.mkdir(parents=True, exist_ok=True)
PDF_OUT = Path(__file__).parent / "media" / "sample.pdf"


def make_pdf(path, pages=5, size=(1240, 1754)):  # A4 at 150 dpi
    title = ImageFont.load_default(size=72)
    head = ImageFont.load_default(size=44)
    body = ImageFont.load_default(size=30)
    text = ("Look at a glowing arrow to scroll. Hold a jaw clench to zoom in or out. "
            "Clench twice quickly to turn the arrows off while you read. ")
    images = []
    for p in range(pages):
        img = Image.new("RGB", size, "white")
        d = ImageDraw.Draw(img)
        d.text((100, 120), "Hands-Free Viewer", font=title, fill=(23, 32, 44))
        d.text((100, 230), f"Sample page {p + 1} of {pages}", font=head, fill=(214, 58, 44))
        y = 340
        for para in range(12):
            words, line = (f"{para + 1}. " + text * 2).split(), ""
            for w in words:
                if d.textlength(line + w, font=body) > size[0] - 200:
                    d.text((100, y), line, font=body, fill=(60, 66, 78))
                    y, line = y + 42, ""
                line += w + " "
            d.text((100, y), line, font=body, fill=(60, 66, 78))
            y += 70
            if y > size[1] - 150:
                break
        d.text((size[0] - 300, size[1] - 80), f"page {p + 1}", font=body, fill=(150, 150, 150))
        images.append(img)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], resolution=150)


def make_audio(path, seconds=40, rate=44100):
    notes = [261.6, 329.6, 392.0, 523.3, 392.0, 329.6, 293.7, 349.2, 440.0, 349.2, 293.7, 246.9]
    beat = 0.5
    t = np.arange(int(beat * rate)) / rate
    env = np.exp(-4 * t)
    tones = [env * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) for f in notes]
    n_beats = int(seconds / beat)
    sig = np.concatenate([tones[i % len(tones)] for i in range(n_beats)]) * 0.3
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((sig * 32767).astype(np.int16).tobytes())


def make_video(path, seconds=120, fps=24, size=(960, 540)):
    big = ImageFont.load_default(size=150)
    small = ImageFont.load_default(size=32)
    writer = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec="libx264",
                                         pix_fmt_out="yuv420p", macro_block_size=8)
    writer.send(None)
    for i in range(seconds * fps):
        s = i / fps
        hue = int(s / seconds * 255)
        img = Image.new("RGB", size, (20 + hue // 4, 30, 60 + hue // 3))
        d = ImageDraw.Draw(img)
        x = int((s * 120) % (size[0] + 100)) - 50  # moving dot so motion is visible
        d.ellipse([x - 25, 430, x + 25, 480], fill=(255, 200, 60))
        d.text((size[0] / 2, 220), f"{int(s // 60)}:{int(s % 60):02d}", font=big, fill="white", anchor="mm")
        d.text((size[0] / 2, 350), "Sample video - clench to pause, look at arrows to skip 15 s",
               font=small, fill=(200, 200, 220), anchor="mm")
        writer.send(img.tobytes())
    writer.close()


if __name__ == "__main__":
    if PDF_OUT.exists():
        print("kept existing", PDF_OUT)
    else:
        make_pdf(PDF_OUT)
        print("wrote", PDF_OUT)
    make_audio(OUT / "sample_audio.wav")
    print("wrote", OUT / "sample_audio.wav")
    make_video(OUT / "sample_video.mp4")
    print("wrote", OUT / "sample_video.mp4")
