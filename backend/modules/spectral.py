"""Module C — Spectral probe (docs 8.2 Module C).

FFT radial power spectrum vs a calibrated real-image envelope. Training-free.
MUST discount its own evidence when d shows resize (docs resize-disambiguation
rule) — on top of the lc-v0 reliability collapse for resampled inputs.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from base import EvidenceModule, ImageContext
from schemas import Artifact, DegradationState, ModuleOutput
from viz import save_spectrum_plot

_DATA = Path(__file__).resolve().parent.parent / "data"
_ENV_PATH = _DATA / "spectral_envelope.npz"
_SIZE = 512
_BINS = 128

_env_cache = None


def radial_profile(gray: np.ndarray) -> np.ndarray:
    """log10 radial power spectrum of a 512x512 grayscale array, 128 bins.
    Shared verbatim with scripts/build_envelope.py via this import."""
    win = np.outer(np.hanning(_SIZE), np.hanning(_SIZE))
    f = np.fft.fftshift(np.fft.fft2(gray * win))
    power = np.abs(f) ** 2
    yy, xx = np.mgrid[0:_SIZE, 0:_SIZE]
    r = np.sqrt((yy - _SIZE / 2) ** 2 + (xx - _SIZE / 2) ** 2) / (_SIZE / 2)
    prof = np.zeros(_BINS)
    for i in range(_BINS):
        m = (r >= i / _BINS) & (r < (i + 1) / _BINS)
        prof[i] = np.log10(power[m].mean() + 1e-12)
    return prof


def bin_freqs() -> np.ndarray:
    return (np.arange(_BINS) + 0.5) / _BINS


def standardize(pil: Image.Image) -> np.ndarray:
    g = pil.convert("L")
    side = min(g.size)
    left, top = (g.width - side) // 2, (g.height - side) // 2
    g = g.crop((left, top, left + side, top + side)).resize((_SIZE, _SIZE), Image.LANCZOS)
    return np.asarray(g, dtype=np.float64)


class SpectralModule(EvidenceModule):
    module_id = "spectral_probe"
    version = "0.1.0"

    def assess(self, ctx: ImageContext, d: DegradationState, base_reliability: float) -> ModuleOutput:
        global _env_cache
        if not _ENV_PATH.exists():
            return self._unavailable("spectral_envelope_missing")
        if _env_cache is None:
            _env_cache = np.load(_ENV_PATH)
        env_mean, env_std, env_n = _env_cache["mean"], _env_cache["std"], int(_env_cache["n"])

        prof = radial_profile(standardize(ctx.pil))
        freqs = bin_freqs()
        z = (prof - env_mean) / (env_std + 1e-6)

        mid = freqs > 0.12
        peak_idx = [
            i for i in range(2, _BINS - 2)
            if mid[i] and z[i] > 3.0 and z[i] >= z[i - 1] and z[i] >= z[i + 1]
            and z[i] - max(z[max(0, i - 4)], z[min(_BINS - 1, i + 4)]) > 0.5
        ]
        peak_freqs = [round(float(freqs[i]), 3) for i in peak_idx]
        hf = freqs > 0.6
        hf_deficit = float(z[hf].mean())
        max_abs_z = float(np.abs(z[mid]).max())

        e = 0.0
        notes = []
        if peak_freqs:
            e -= min(0.85, 0.35 + 0.12 * len(peak_freqs))
            notes.append(f"periodic energy peaks at normalized frequencies {peak_freqs} exceed the real envelope by >3 sigma")
        if hf_deficit < -2.5:
            e -= 0.30
            notes.append(f"high-frequency band (f>0.6) sits {abs(hf_deficit):.1f} sigma BELOW the real envelope (over-smooth spectrum)")
        if not peak_freqs and max_abs_z < 2.5:
            e += 0.35
            notes.append(f"radial spectrum stays within the calibrated real envelope (max |z|={max_abs_z:.1f} < 2.5)")

        discounted = False
        if abs(d.resize_factor_est - 1.0) > 0.08:
            e *= 0.25  # docs resize-disambiguation rule: peaks may be manufactured by resampling
            discounted = True
            notes.append(f"evidence discounted 4x: triage detected resampling (factor ~{d.resize_factor_est:.2f}), "
                         "which manufactures spectral peaks")
        e = float(np.clip(e, -1.0, 1.0))

        plot_name = "spectrum_polar.png"
        save_spectrum_plot(freqs, prof, env_mean, env_std, peak_freqs, ctx.artifact_abs(plot_name))

        claim = (f"FFT radial power spectrum (512px standardization, {_BINS} bins) vs real envelope "
                 f"(n={env_n}): " + "; ".join(notes))
        artifacts = [Artifact(
            type="spectral_peak" if peak_freqs else "spectral_profile",
            description="Radial power spectrum against calibrated real-image envelope"
                        + (" (resize-discounted)" if discounted else ""),
            strength=min(1.0, abs(e) + 0.15),
            visual=ctx.artifact_rel(plot_name),
            checkable_claim=claim,
        )]

        direction = "synthetic" if e < -0.15 else ("authentic" if e > 0.15 else "neutral")
        confidence = min(0.85, 0.45 + 0.08 * len(peak_freqs) + (0.15 if hf_deficit < -2.5 else 0.0))
        if discounted:
            confidence = min(confidence, 0.45)

        return ModuleOutput(
            module_id=self.module_id, version=self.version,
            evidence_score=round(e, 4), reliability_score=base_reliability,
            confidence_score=round(confidence, 4), verdict_direction=direction,
            artifacts=artifacts,
        )


MODULE = SpectralModule
