"""Reliability curves lc-v0 — hand-authored monotonic lookup tables.

r_m(d) = probability the module's evidence DIRECTION is correct given the
degradation state d (docs 4.2). These v0 tables are hand-authored from
forensic priors and are monotonic non-increasing in degradation severity;
they will be replaced by calibrated curves (lc-v1) fit on the laundering
ladder (docs 8.4). Version string is embedded in every verdict.
"""
from schemas import DegradationState

RELIABILITY_VERSION = "lc-v0"

_DEFAULT = 0.5


def _q_bin(q, bins):
    """bins: list of (q_threshold_inclusive, value) descending by q."""
    if q is None:
        return None
    for thr, val in bins:
        if q >= thr:
            return val
    return bins[-1][1]


def _metadata(d: DegradationState) -> float:
    # Metadata survives recompression but is stripped by screenshots/platforms.
    r = 0.85
    r *= 1.0 - 0.55 * d.screenshot_probability
    return r


def _compression(d: DegradationState) -> float:
    # ELA/ghost analysis dies with recompression generations and low quality.
    if d.jpeg_quality_est is not None:
        r = _q_bin(d.jpeg_quality_est, [(90, 0.85), (80, 0.72), (70, 0.58), (60, 0.42), (0, 0.28)])
    else:
        r = 0.50  # PNG container: ghost sweep can still reveal concealed JPEG history
    if d.recompression_generations >= 3:
        r *= 0.45
    elif d.recompression_generations == 2:
        r *= 0.70
    r *= 1.0 - 0.35 * d.screenshot_probability
    return r


def _spectral(d: DegradationState) -> float:
    # Resize laundering manufactures spectral peaks (docs 4.1 worked example):
    # reliability collapses when resampling is detected.
    r = 0.80
    if abs(d.resize_factor_est - 1.0) > 0.08:
        r = 0.18
    if d.jpeg_quality_est is not None:
        r *= {True: 1.0}.get(d.jpeg_quality_est >= 85, 1.0)
        if d.jpeg_quality_est < 85:
            r *= 0.85
        if d.jpeg_quality_est < 70:
            r *= 0.75
        if d.jpeg_quality_est < 55:
            r *= 0.65
    if d.recompression_generations >= 2:
        r *= 0.75
    r *= 1.0 - 0.30 * d.screenshot_probability
    return r


def _realness(d: DegradationState) -> float:
    # DINOv2 representations are laundering-robust (docs: ~92% under transforms).
    r = 0.85
    if d.effective_resolution < 150:
        r = 0.40
    elif d.effective_resolution < 300:
        r = 0.62
    if d.jpeg_quality_est is not None and d.jpeg_quality_est < 50:
        r *= 0.85
    return r


def _perturbation(d: DegradationState) -> float:
    # High-frequency probe: partially laundering-sensitive (docs module F).
    r = 0.80
    if d.jpeg_quality_est is not None:
        if d.jpeg_quality_est < 85:
            r *= 0.85
        if d.jpeg_quality_est < 70:
            r *= 0.70
        if d.jpeg_quality_est < 55:
            r *= 0.60
    if abs(d.resize_factor_est - 1.0) > 0.08:
        r *= 0.60
    r *= 1.0 - 0.30 * d.screenshot_probability
    if d.recompression_generations >= 2:
        r *= 0.80
    return r


_CURVES = {
    "metadata": _metadata,
    "compression_history": _compression,
    "spectral_probe": _spectral,
    "real_distribution_probe": _realness,
    "perturbation_probe": _perturbation,
    "stub": lambda d: 0.5,
}


def reliability(module_id: str, d: DegradationState) -> float:
    fn = _CURVES.get(module_id)
    r = fn(d) if fn else _DEFAULT
    return max(0.0, min(1.0, round(r, 3)))


def reliability_explanation(module_id: str, d: DegradationState) -> str:
    """Human-readable cause for a low reliability value (used by the gate)."""
    causes = []
    if module_id == "spectral_probe" and abs(d.resize_factor_est - 1.0) > 0.08:
        causes.append(f"resampling detected (factor ~{d.resize_factor_est:.2f}) manufactures spectral peaks")
    if d.recompression_generations >= 2:
        causes.append(f"{d.recompression_generations} recompression generations suppress forensic traces")
    if d.jpeg_quality_est is not None and d.jpeg_quality_est < 70:
        causes.append(f"low JPEG quality (~{d.jpeg_quality_est}) destroys high-frequency evidence")
    if d.screenshot_probability > 0.6:
        causes.append("probable screenshot: container metadata and pipeline traces replaced")
    if d.effective_resolution and d.effective_resolution < 300:
        causes.append(f"low effective resolution ({d.effective_resolution}px)")
    if not causes:
        causes.append("measurement conditions insufficient for this module")
    return "; ".join(causes)
