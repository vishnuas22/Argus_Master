"""Module F — Perturbation-sensitivity probe (docs 8.2 Module F, RIGID-style).

Cosine-similarity drop of the frozen DINOv2 embedding under (a) structured
high-frequency Fourier noise and (b) Gaussian blur. Real camera images sit in
more perturbation-sensitive regions of representation space than generated
images; thresholds calibrated on the same real reference corpus as module E.
Shares the DINOv2 backbone with module E (one extra forward pass each).
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from base import EvidenceModule, ImageContext
from dino_service import DinoService, IMG_SIZE
from schemas import Artifact, DegradationState, ModuleOutput
from viz import save_bar_panel

_DATA = Path(__file__).resolve().parent.parent / "data"
_CAL = _DATA / "perturb_calib.npz"
_SEED = 1337
_NOISE_STRENGTH = 8.0
_BLUR_RADIUS = 2.0

# CAL_SIGN +1: higher-than-real-typical drop percentile reads authentic.
# Verified at the M3 gate; flip to -1 only with a logged decision.
_CAL_SIGN = +1.0

_noise_cache = None
_cal_cache = None


def structured_noise(size: int = IMG_SIZE) -> np.ndarray:
    """Deterministic high-frequency Fourier-domain noise (seeded)."""
    global _noise_cache
    if _noise_cache is None:
        rng = np.random.default_rng(_SEED)
        fy = np.fft.fftfreq(size)[:, None]
        fx = np.fft.fftfreq(size)[None, :]
        r = np.sqrt(fx ** 2 + fy ** 2)
        spec = (r > 0.25).astype(np.float64) * np.exp(1j * rng.uniform(0, 2 * np.pi, (size, size)))
        noise = np.real(np.fft.ifft2(spec))
        _noise_cache = noise / (noise.std() + 1e-9) * _NOISE_STRENGTH
    return _noise_cache


def perturb_drops(pil: Image.Image, service: DinoService, base_emb: np.ndarray):
    base224 = pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.asarray(base224, dtype=np.float64)
    noisy = Image.fromarray(np.clip(arr + structured_noise()[..., None], 0, 255).astype(np.uint8))
    blurred = base224.filter(ImageFilter.GaussianBlur(_BLUR_RADIUS))
    e_n = service.embed(noisy)
    e_b = service.embed(blurred)
    return float(1.0 - np.dot(base_emb, e_n)), float(1.0 - np.dot(base_emb, e_b))


class PerturbationModule(EvidenceModule):
    module_id = "perturbation_probe"
    version = "0.1.0"

    def assess(self, ctx: ImageContext, d: DegradationState, base_reliability: float) -> ModuleOutput:
        global _cal_cache
        service = DinoService.get()
        if not service.available():
            return self._unavailable("backbone_unavailable: torch/timm not installed")
        if not _CAL.exists():
            return self._unavailable("perturbation_calibration_missing")
        if _cal_cache is None:
            data = np.load(_CAL)
            _cal_cache = (np.sort(data["noise_drops"]), np.sort(data["blur_drops"]))
        cal_n, cal_b = _cal_cache

        base_emb = service.embed(ctx.pil, cache_key=ctx.sha256)  # shared with module E
        drop_n, drop_b = perturb_drops(ctx.pil, service, base_emb)
        q_n = float(np.searchsorted(cal_n, drop_n) / len(cal_n))
        q_b = float(np.searchsorted(cal_b, drop_b) / len(cal_b))

        e = float(np.clip(_CAL_SIGN * 0.8 * (0.6 * (2 * q_n - 1) + 0.4 * (2 * q_b - 1)), -1.0, 1.0))
        direction = "synthetic" if e < -0.15 else ("authentic" if e > 0.15 else "neutral")
        confidence = min(0.8, 0.35 + 0.5 * abs(0.6 * (2 * q_n - 1) + 0.4 * (2 * q_b - 1)))

        panel_name = "perturbation_drops.png"
        save_bar_panel(
            ["hf-noise drop", "blur drop"], [drop_n, drop_b],
            ctx.artifact_abs(panel_name),
            "Embedding similarity drop under structured perturbations",
            ref_lines=[("real median (noise)", float(np.median(cal_n)), "#3FB950"),
                       ("real median (blur)", float(np.median(cal_b)), "#58A6FF")],
            ylabel="1 − cosine similarity",
        )

        claim = (f"DINOv2 cosine-similarity drop under seeded high-frequency Fourier noise = {drop_n:.4f} "
                 f"({q_n*100:.1f}th real-corpus percentile) and under Gaussian blur r={_BLUR_RADIUS} = {drop_b:.4f} "
                 f"({q_b*100:.1f}th percentile); generated images sit in flatter representation regions "
                 f"(low percentiles read synthetic)")

        return ModuleOutput(
            module_id=self.module_id, version=self.version,
            evidence_score=round(e, 4), reliability_score=base_reliability,
            confidence_score=round(confidence, 4), verdict_direction=direction,
            artifacts=[Artifact(
                type="perturbation_flatness" if e < 0 else "perturbation_sensitivity",
                description="Representation sensitivity to structured high-frequency noise and blur (RIGID-style)",
                strength=min(1.0, abs(e) + 0.1),
                visual=ctx.artifact_rel(panel_name),
                checkable_claim=claim,
            )],
        )


MODULE = PerturbationModule
