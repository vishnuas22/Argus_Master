"""Tier 1 — Degradation triage (docs 2.2 Tier 1, 8.2).

Estimates the degradation state vector d before any evidence module runs:
- JPEG quality from quantization tables (Pillow)
- recompression generations via JPEG-ghost minima (double-JPEG heuristic)
- resize detection via Gallagher-style second-difference residual FFT peaks
- screenshot heuristic (container/aspect/border rules)
All deterministic, CPU, tens of milliseconds.
"""
import io

import numpy as np
from PIL import Image

from base import ImageContext
from schemas import DegradationState

TRIAGE_VERSION = "0.1.0"

# Standard IJG luminance quantization table (natural order).
_STD_LUM = np.array(
    [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
     14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
     18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
     49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
    dtype=np.float64,
)

_COMMON_SCREEN_DIMS = {
    (1920, 1080), (1366, 768), (1536, 864), (1440, 900), (2560, 1440),
    (1280, 720), (3840, 2160), (1280, 800), (1600, 900), (2880, 1800),
    (750, 1334), (828, 1792), (1080, 1920), (1080, 2340), (1125, 2436),
    (1170, 2532), (1242, 2688), (1284, 2778), (720, 1280), (1440, 2560),
}
_SCREEN_ASPECTS = (16 / 9, 9 / 16, 16 / 10, 10 / 16, 19.5 / 9, 9 / 19.5)


def _scaled_std_sum(q: int) -> float:
    s = 5000.0 / q if q < 50 else 200.0 - 2.0 * q
    t = np.clip(np.floor((_STD_LUM * s + 50.0) / 100.0), 1, 255)
    return float(t.sum())


def estimate_jpeg_quality(pil: Image.Image):
    qt = getattr(pil, "quantization", None)
    if not qt or 0 not in qt:
        return None
    total = float(np.sum(np.asarray(qt[0], dtype=np.float64)))
    best_q, best_err = None, float("inf")
    for q in range(1, 101):
        err = abs(_scaled_std_sum(q) - total)
        if err < best_err:
            best_q, best_err = q, err
    return best_q


def ghost_curve(pil_rgb: Image.Image, max_side: int = 512):
    """Recompress sweep q=50..100; MSE dips at prior compression qualities."""
    img = pil_rgb
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(8, int(img.width * scale)), max(8, int(img.height * scale))), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float64)
    qs = list(range(50, 101, 5))
    curve = []
    for q in qs:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q)
        buf.seek(0)
        rec = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float64)
        curve.append(float(np.mean((arr - rec) ** 2)))
    minima = []
    for i in range(1, len(qs) - 1):
        dip = (curve[i - 1] + curve[i + 1]) / 2.0 - curve[i]
        if curve[i] < curve[i - 1] and curve[i] <= curve[i + 1] and dip > 0.03 * (curve[i] + 1e-9):
            minima.append(qs[i])
    return qs, curve, minima


def detect_resize(gray: np.ndarray, is_jpeg: bool):
    """Gallagher-style resampling detection. Returns (factor_est, peak_ratio).
    factor_est=1.0 means no resampling detected. Estimate maps the strongest
    residual-spectrum peak frequency f to factor 1/f (upsampling assumption,
    lc-v0 simplification — see DECISIONS.md)."""
    h, w = gray.shape
    g = gray[: min(h, 1024), : min(w, 1024)]
    best = None
    for axis in (0, 1):
        d2 = np.abs(np.diff(g, n=2, axis=axis))
        prof = d2.mean(axis=1 - axis)
        if len(prof) < 64:
            continue
        prof = prof - prof.mean()
        spec = np.abs(np.fft.rfft(prof * np.hanning(len(prof))))
        freqs = np.fft.rfftfreq(len(prof))
        med = float(np.median(spec[1:])) + 1e-9
        for i in range(2, len(spec) - 2):
            f = float(freqs[i])
            if not 0.04 < f < 0.46:
                continue
            if is_jpeg and min(abs(f - k / 8.0) for k in (1, 2, 3)) < 0.012:
                continue  # JPEG blocking harmonics, not resampling
            ratio = spec[i] / med
            if ratio > 6.0 and spec[i] >= spec[i - 1] and spec[i] >= spec[i + 1]:
                if best is None or ratio > best[0]:
                    best = (float(ratio), f)
    if best is None:
        return 1.0, 0.0
    factor = float(np.clip(1.0 / best[1], 1.05, 4.0))
    return round(factor, 2), round(best[0], 1)


def screenshot_probability(pil: Image.Image, fmt: str) -> float:
    w, h = pil.size
    p = 0.0
    if fmt == "png":
        p += 0.30
    try:
        has_exif = len(pil.getexif()) > 0
    except Exception:
        has_exif = False
    if not has_exif:
        p += 0.15 if fmt == "png" else 0.05
    if (w, h) in _COMMON_SCREEN_DIMS:
        p += 0.35
    else:
        ar = w / h
        if any(abs(ar - a) < 0.01 for a in _SCREEN_ASPECTS):
            p += 0.15
    arr = np.asarray(pil.convert("L"), dtype=np.float64)
    if arr.shape[0] > 12 and (arr[:3].std() < 2.0 or arr[-3:].std() < 2.0):
        p += 0.15
    return round(min(1.0, p), 2)


class DegradationEstimator:
    version = TRIAGE_VERSION

    def estimate(self, ctx: ImageContext) -> DegradationState:
        pil = ctx.pil
        fmt = ctx.fmt
        rgb = pil.convert("RGB")
        gray = np.asarray(rgb.convert("L"), dtype=np.float64)

        q_est = estimate_jpeg_quality(pil) if fmt == "jpeg" else None
        _, _, ghost_minima = ghost_curve(rgb)

        if fmt == "jpeg":
            extra = [m for m in ghost_minima if q_est is None or abs(m - q_est) > 7]
            n_gen = 1 + min(2, len(extra))
        else:
            n_gen = 1 if ghost_minima else 0

        resize_factor, _peak = detect_resize(gray, is_jpeg=(fmt == "jpeg"))
        p_screen = screenshot_probability(pil, fmt)
        eff_res = int(min(pil.size) / max(1.0, resize_factor))

        penalty = 0
        if q_est is not None:
            penalty += (q_est < 85) + (q_est < 70) + (q_est < 55)
        penalty += (n_gen >= 2) + (n_gen >= 3)
        penalty += abs(resize_factor - 1.0) > 0.08
        penalty += p_screen > 0.6
        penalty += (min(pil.size) < 500) + (min(pil.size) < 250)
        capacity = "HIGH" if penalty <= 1 else ("MODERATE" if penalty <= 3 else "LOW")

        return DegradationState(
            jpeg_quality_est=q_est,
            recompression_generations=int(n_gen),
            resize_factor_est=float(resize_factor),
            screenshot_probability=float(p_screen),
            effective_resolution=eff_res,
            evidence_capacity=capacity,
        )
