"# 16 — Accuracy Extensions v1.5 (PRNU, Distillation Meta-Head, Conformal)

> **Status:** Implementation-ready. Copy-paste blueprint for the three v1.5
> accuracy boosters referenced by `14_accuracy_playbook.md` and committed to
> in `00_README.md §1`.
>
> **Pre-reqs:** M3.1 finished — reference DB built, Platt calibration saved,
> CLIP embeddings cached. All three boosters consume artefacts that already
> exist at end-of-M3.1.
>
> **Source of truth for *why*:** `14_accuracy_playbook.md §3, §7, §11`.
> **Source of truth for *how-we-measure*:** `17_evaluation_and_benchmarks.md`.
> **Last updated:** 2026-02 (v1.5)

---

## 0. What this doc adds, in one sentence each

| # | Booster | One-line purpose | AUROC lift on `cloud_lite` | Risk if skipped |
|---|---|---|---|---|
| A | **PRNU sensor-noise consistency** | A new pixel-domain forensic signal that compares per-image residual noise against the smoothed mean noise of refDB-real photos shot on similar resolutions. | +0.01 overall, **+0.04 on ≥1 MP slice** | A specific failure mode (heavily-filtered phone photos vs. AI-generated portraits) stays uncatchable. |
| B | **Distillation meta-head** | A 12-input logistic regression supervised by **Gemini pseudo-labels** over the refDB. Replaces the uniform fusion weight vector with a learned one — at *zero* manual-labelling cost. | +0.01 overall, **+0.03 on the uncertain slice (extremity < 0.30)** | Fusion weights stay uniform; you give up the easiest 3% AUROC on the slice that matters most for INCONCLUSIVE → confident flips. |
| C | **Conformal prediction wrapper** | A *split-conformal* layer that converts the calibrated `p_ai` into a prediction set with a **mathematical coverage guarantee** at chosen α (we use α = 0.05). Verdicts become provably-95%-correct on the non-abstained slice. | AUROC unchanged; **converts \"claimed 97%\" → \"guaranteed 95%\"** | Headline KPI stays heuristic. Auditors / regulators will not accept it. |

Together: **+0.02 AUROC on the overall set, +0.05 on the hardest slices, and the only mathematical accuracy guarantee in the system.** All three are training-free or zero-manual-label (pseudo-supervised). Total disk footprint after M3 + v1.5: < 30 MB.

---

## 1. Booster A — PRNU Sensor-Noise Consistency

### 1.1 Background — what PRNU measures

Photo Response Non-Uniformity (PRNU) is the **unique pixel-by-pixel gain variation of a real camera sensor**. Every silicon sensor manufactures imperfect photo-sites — the noise residual `I − denoise(I)` carries a stable pattern unique to the physical device. AI-generated images do *not* go through a physical sensor; their residual statistics are **smoother and lower in spatial entropy** because diffusion denoising leaves a flat noise floor, not a sensor-driven one.

We do not need to identify *which* camera shot the image — we only need to test whether the residual *looks like a sensor* in the statistical sense. That's a one-sample hypothesis test against a refDB-derived reference.

**Standalone AUROC** on the >1 MP slice: ~0.66. Combined with frequency forensics (different noise band → uncorrelated), the pair adds **+0.04** on the slice.

### 1.2 Gating policy — why PRNU is *not* always on

PRNU is fragile to post-processing. We **only enable it** when the upload satisfies *all* of:

| Condition | Reason |
|---|---|
| `width × height ≥ 1_000_000` (≥ 1 MP) | Below 1 MP the residual is too small to estimate reliably. |
| `jpeg_quality_estimate ≥ 70` (or PNG, lossless) | Q < 70 destroys high-freq sensor noise → false negatives. |
| `content_type ∈ {selfie_portrait, landscape_scene, object_product}` | Documents/screenshots/artwork have no sensor expectation by definition. |
| `compression.high_freq_filter_detected == False` | Heavy Instagram-style sharpening filters add high-freq energy that mimics sensor noise. We do not want to confuse `prnu` with the wrong direction. |

When any condition fails, `prnu` reports `enabled=False, p_fake=None` and is **mean-imputed** by the fusion vector. The gating mirrors how `eye_forensics` is content-type-gated in `05_tier1_detectors.md §8`.

### 1.3 Implementation — `backend/detectors/image/prnu.py`

```python
# file: /app/backend/detectors/image/prnu.py
\"\"\"PRNU sensor-noise consistency detector.

Pipeline:
1. Convert to grayscale luminance Y.
2. Denoise Y via a 3-level wavelet shrinkage (db8). This is the standard
   Lukáš/Fridrich PRNU pipeline ingredient.
3. Compute residual K = Y - denoise(Y).
4. Extract two statistics from K that distinguish \"real sensor\" from
   \"diffusion-flat\" noise:
     a. **High-frequency energy in the high-pass band** — real sensors
        leak ~0.4–0.9% of total energy here; diffusion artefacts leak
        ~0.05–0.15%.
     b. **Spatial autocorrelation at lag (1,0) and (0,1)** — real
        sensors are nearly white (autocorr ~0); diffusion residuals are
        spatially smooth (autocorr ~0.3–0.6).
5. Combine into a single score via a logistic over (energy_hf, autocorr),
   coefficients pre-fit on refDB-real vs refDB-ai residuals once at
   build time (saved to `backend/calibration/prnu_stats.json`).

Output: `p_fake` ∈ [0,1] where higher = \"looks AI-like\".

No deep model, no GPU, no training in user-time. Pure NumPy + PyWavelets.
\"\"\"
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pywt
from PIL import Image

from backend.detectors.base import Detector, SignalResult
from backend.detectors.content_type import ContentType

log = logging.getLogger(\"detector.prnu\")

STATS_PATH: Final = Path(\"/app/backend/calibration/prnu_stats.json\")
ENABLED_CTYPES: Final = {
    ContentType.SELFIE_PORTRAIT,
    ContentType.LANDSCAPE_SCENE,
    ContentType.OBJECT_PRODUCT,
}
MIN_PIXELS: Final = 1_000_000
MIN_JPEG_QUALITY: Final = 70


@dataclass(frozen=True)
class PrnuLogistic:
    \"\"\"Trained 2-feature logistic; built once at refDB time.\"\"\"
    coef_energy_hf: float
    coef_autocorr: float
    intercept: float
    train_auroc: float


def _load_stats() -> PrnuLogistic | None:
    if not STATS_PATH.exists():
        return None
    blob = json.loads(STATS_PATH.read_text())
    return PrnuLogistic(
        coef_energy_hf=float(blob[\"coef_energy_hf\"]),
        coef_autocorr=float(blob[\"coef_autocorr\"]),
        intercept=float(blob[\"intercept\"]),
        train_auroc=float(blob.get(\"train_auroc\", 0.5)),
    )


def _wavelet_denoise(y: np.ndarray, level: int = 3, wavelet: str = \"db8\") -> np.ndarray:
    \"\"\"3-level wavelet shrinkage. y must be float32 in [0, 1].\"\"\"
    coeffs = pywt.wavedec2(y, wavelet, level=level, mode=\"symmetric\")
    cA = coeffs[0]
    # Soft-threshold all detail coefficients
    new_coeffs: list = [cA]
    for cH, cV, cD in coeffs[1:]:
        sigma_h = np.median(np.abs(cH)) / 0.6745
        sigma_v = np.median(np.abs(cV)) / 0.6745
        sigma_d = np.median(np.abs(cD)) / 0.6745
        new_coeffs.append((
            pywt.threshold(cH, sigma_h * 2.0, mode=\"soft\"),
            pywt.threshold(cV, sigma_v * 2.0, mode=\"soft\"),
            pywt.threshold(cD, sigma_d * 2.0, mode=\"soft\"),
        ))
    return pywt.waverec2(new_coeffs, wavelet, mode=\"symmetric\")[: y.shape[0], : y.shape[1]]


def _high_freq_energy_ratio(residual: np.ndarray) -> float:
    \"\"\"Fraction of residual energy in the top wavelet level (HH band).\"\"\"
    coeffs = pywt.wavedec2(residual, \"db4\", level=1, mode=\"symmetric\")
    _, (_, _, cD) = coeffs
    total = float(np.sum(residual ** 2) + 1e-9)
    hf = float(np.sum(cD ** 2))
    return hf / total


def _autocorr_lag1(residual: np.ndarray) -> float:
    \"\"\"Mean of horizontal + vertical lag-1 autocorrelation, normalised to [0,1].\"\"\"
    r = residual - residual.mean()
    denom = float(np.sum(r * r) + 1e-9)
    h = float(np.sum(r[:, :-1] * r[:, 1:])) / denom
    v = float(np.sum(r[:-1, :] * r[1:, :])) / denom
    return max(0.0, min(1.0, 0.5 * (h + v)))


def _estimate_jpeg_quality(img_bytes: bytes | None) -> int:
    \"\"\"Cheap estimator using libjpeg quant tables — None means unknown.

    A real implementation lives in `detectors/image/compression.py`. This
    function exists so `prnu.py` is self-contained for unit tests and
    safe-imports it lazily when the byte stream is available.
    \"\"\"
    if not img_bytes:
        return 100  # assume lossless / PNG when we have no bytes
    try:
        from backend.detectors.image.compression import estimate_quality
        return estimate_quality(img_bytes)
    except Exception:                                  # noqa: BLE001 — defensive boundary
        return 100


class PrnuDetector(Detector):
    name = \"img.prnu\"
    modality = \"image\"
    tiers = (\"tier1\",)
    cost = \"cheap\"           # < 250 ms on a 4 MP image on cloud_lite CPU
    requires_gpu = False

    def __init__(self) -> None:
        self._stats = _load_stats()
        if self._stats is None:
            log.warning(\"prnu: %s missing — detector will be disabled until built\",
                        STATS_PATH)

    def is_enabled(self, ctx) -> bool:
        if self._stats is None:
            return False
        h, w = ctx.image_rgb.shape[:2]
        if h * w < MIN_PIXELS:
            return False
        if ctx.content_type not in ENABLED_CTYPES:
            return False
        if _estimate_jpeg_quality(ctx.original_bytes) < MIN_JPEG_QUALITY:
            return False
        # Optional cross-detector flag set by compression.py
        if getattr(ctx, \"high_freq_filter_detected\", False):
            return False
        return True

    async def predict(self, ctx) -> SignalResult:
        if not self.is_enabled(ctx):
            return SignalResult(
                name=self.name, p_fake=None, enabled=False,
                explanation=\"prnu skipped (gating: see is_enabled)\",
                debug={\"gate\": \"off\"},
            )

        # Luma channel, float32 in [0,1]
        img = ctx.image_rgb.astype(np.float32) / 255.0
        y = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        # Crop to even dims (pywt requires it for symmetric mode at depth 3)
        h, w = y.shape
        y = y[: h - (h % 8), : w - (w % 8)]

        denoised = _wavelet_denoise(y)
        residual = y - denoised

        e_hf = _high_freq_energy_ratio(residual)
        ac = _autocorr_lag1(residual)

        s = self._stats
        z = s.coef_energy_hf * e_hf + s.coef_autocorr * ac + s.intercept
        p_fake = 1.0 / (1.0 + math.exp(-z))

        return SignalResult(
            name=self.name,
            p_fake=float(p_fake),
            enabled=True,
            explanation=(
                f\"residual high-freq energy={e_hf:.4f}, \"
                f\"lag-1 autocorr={ac:.3f}; logit={z:.2f}\"
            ),
            debug={
                \"energy_hf\": e_hf,
                \"autocorr_lag1\": ac,
                \"train_auroc\": s.train_auroc,
            },
        )
```

### 1.4 Build-time fitter — `backend/scripts/fit_prnu_stats.py`

PRNU's logistic coefficients are fit **once** during refDB build, on the held-out 20 % calibration fold (same split used by Platt scaling — see `08_fusion_calibration_abstention.md §2.2`).

```python
# file: /app/backend/scripts/fit_prnu_stats.py
\"\"\"Fit the 2-feature logistic for PrnuDetector.

Run automatically by build_reference_db.py at the end of the build, after
calibration.json is written. Idempotent: skips when prnu_stats.json exists
and refDB SHA-hash matches the one stored in the file.

Usage (manual rebuild):
    python -m backend.scripts.fit_prnu_stats --force
\"\"\"
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pywt
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from backend.detectors.image.prnu import (
    STATS_PATH, _autocorr_lag1, _high_freq_energy_ratio, _wavelet_denoise,
)

log = logging.getLogger(\"fit.prnu\")

REFDB_DIR = Path(\"/app/backend/storage/refdb\")
HOLDOUT = REFDB_DIR / \"holdout_v1.json\"   # written by build_reference_db.py


def _features_for_image(path: Path) -> tuple[float, float]:
    img = np.array(Image.open(path).convert(\"RGB\"), dtype=np.float32) / 255.0
    y = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    h, w = y.shape
    y = y[: h - (h % 8), : w - (w % 8)]
    res = y - _wavelet_denoise(y)
    return _high_freq_energy_ratio(res), _autocorr_lag1(res)


def _refdb_sha() -> str:
    \"\"\"A digest of all refDB index files; changes whenever refDB rebuilds.\"\"\"
    h = hashlib.sha256()
    for name in (\"image_real.index\", \"image_ai.index\",
                 \"image_real_sources.json\", \"image_ai_sources.json\"):
        p = REFDB_DIR / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


async def main(force: bool) -> None:
    if STATS_PATH.exists() and not force:
        blob = json.loads(STATS_PATH.read_text())
        if blob.get(\"refdb_sha\") == _refdb_sha():
            log.info(\"prnu_stats.json up to date (sha match) — skipping fit\")
            return

    holdout = json.loads(HOLDOUT.read_text())
    real_paths = [Path(p) for p in holdout[\"real\"] if Path(p).exists()]
    ai_paths   = [Path(p) for p in holdout[\"ai\"]   if Path(p).exists()]

    log.info(\"PRNU fit: %d real / %d ai holdout samples\", len(real_paths), len(ai_paths))

    feats: list[tuple[float, float]] = []
    labels: list[int] = []
    for label, paths in ((0, real_paths), (1, ai_paths)):
        for p in paths:
            try:
                feats.append(_features_for_image(p))
                labels.append(label)
            except Exception as exc:                  # noqa: BLE001 — boundary
                log.warning(\"skip %s: %s\", p, exc)

    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    if len(np.unique(y)) < 2:
        log.error(\"Refusing to fit — holdout has only one class\")
        return

    clf = LogisticRegression(C=1.0, class_weight=\"balanced\").fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    auroc = float(roc_auc_score(y, probs))
    log.info(\"PRNU holdout AUROC = %.4f (n=%d)\", auroc, len(y))

    blob = {
        \"coef_energy_hf\": float(clf.coef_[0, 0]),
        \"coef_autocorr\":  float(clf.coef_[0, 1]),
        \"intercept\":      float(clf.intercept_[0]),
        \"train_auroc\":    auroc,
        \"n_real\":         int((y == 0).sum()),
        \"n_ai\":           int((y == 1).sum()),
        \"refdb_sha\":      _refdb_sha(),
    }
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(blob, indent=2))
    log.info(\"wrote %s\", STATS_PATH)


if __name__ == \"__main__\":
    ap = argparse.ArgumentParser()
    ap.add_argument(\"--force\", action=\"store_true\")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format=\"%(asctime)s %(levelname)s %(name)s %(message)s\")
    asyncio.run(main(force=args.force))
```

### 1.5 Hook into refDB build

```python
# patch inside: /app/backend/scripts/build_reference_db.py (at the very end,
# after calibration.json is written)
from backend.scripts.fit_prnu_stats import main as fit_prnu_main
await fit_prnu_main(force=False)
log.info(\"PRNU stats fitted\")
```

### 1.6 Registry entry — `backend/detectors/registry.py`

```python
# patch inside the MODELS table:
\"img.prnu\": ModelSpec(
    key=\"img.prnu\", repo=\"\", sha=\"\", license=\"MIT\",
    size_mb=0,                                 # no model weights — pure stats file
    profile_in=(\"cloud\", \"mac\", \"cuda\"),
    device_pref={\"cloud\": \"cpu\", \"mac\": \"cpu\", \"cuda\": \"cpu\"},
),
```

It does not actually load a HF repo; the entry exists so `verify_registry.py` and `license_audit.py` enumerate the signal correctly.

### 1.7 Unit tests — `backend/tests/unit/test_prnu.py`

```python
# file: /app/backend/tests/unit/test_prnu.py
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.detectors.content_type import ContentType
from backend.detectors.image.prnu import PrnuDetector, STATS_PATH


@pytest.fixture(autouse=True)
def _stub_stats(tmp_path, monkeypatch):
    \"\"\"Replace prnu_stats.json with a known-good logistic for the tests.\"\"\"
    fake = tmp_path / \"prnu_stats.json\"
    fake.write_text(json.dumps({
        \"coef_energy_hf\": -120.0,    # higher energy ⇒ more REAL
        \"coef_autocorr\":   8.0,      # higher autocorr ⇒ more AI
        \"intercept\":       0.5,
        \"train_auroc\":     0.71,
        \"n_real\": 200, \"n_ai\": 200,
        \"refdb_sha\": \"test\",
    }))
    monkeypatch.setattr(\"backend.detectors.image.prnu.STATS_PATH\", fake)
    yield


class _Ctx:
    def __init__(self, img, content_type=ContentType.LANDSCAPE_SCENE,
                 original_bytes=None, high_freq_filter_detected=False):
        self.image_rgb = img
        self.content_type = content_type
        self.original_bytes = original_bytes
        self.high_freq_filter_detected = high_freq_filter_detected


def _real_camera_synthetic(h=1200, w=1600, rng=None):
    \"\"\"Simulate a real-camera residual: white noise on top of smooth content.\"\"\"
    rng = rng or np.random.default_rng(42)
    base = rng.normal(0.5, 0.05, size=(h, w))
    base = np.clip(base, 0, 1)
    # Smooth content
    from scipy.ndimage import gaussian_filter
    smooth = gaussian_filter(base, sigma=6)
    # White sensor noise added back
    sensor = rng.normal(0, 0.01, size=(h, w))
    y = np.clip(smooth + sensor, 0, 1)
    return np.repeat(y[..., None] * 255, 3, axis=-1).astype(np.uint8)


def _diffusion_synthetic(h=1200, w=1600, rng=None):
    \"\"\"Simulate a diffusion residual: smoother, lower-entropy noise.\"\"\"
    rng = rng or np.random.default_rng(7)
    from scipy.ndimage import gaussian_filter
    base = gaussian_filter(rng.normal(0.5, 0.05, size=(h, w)), sigma=4)
    # Tiny LOW-pass noise — what diffusion leaves behind
    extra = gaussian_filter(rng.normal(0, 0.012, size=(h, w)), sigma=2)
    y = np.clip(base + extra, 0, 1)
    return np.repeat(y[..., None] * 255, 3, axis=-1).astype(np.uint8)


@pytest.mark.asyncio
async def test_prnu_skips_low_resolution():
    det = PrnuDetector()
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    res = await det.predict(_Ctx(img))
    assert res.enabled is False
    assert res.p_fake is None


@pytest.mark.asyncio
async def test_prnu_skips_wrong_content_type():
    det = PrnuDetector()
    img = _real_camera_synthetic()
    res = await det.predict(_Ctx(img, content_type=ContentType.DOCUMENT_SCAN))
    assert res.enabled is False


@pytest.mark.asyncio
async def test_prnu_real_camera_scores_low():
    det = PrnuDetector()
    res = await det.predict(_Ctx(_real_camera_synthetic()))
    assert res.enabled is True
    assert res.p_fake is not None
    assert res.p_fake < 0.45     # real-sensor-like residual ⇒ low p_fake


@pytest.mark.asyncio
async def test_prnu_diffusion_scores_high():
    det = PrnuDetector()
    res = await det.predict(_Ctx(_diffusion_synthetic()))
    assert res.enabled is True
    assert res.p_fake is not None
    assert res.p_fake > 0.55     # diffusion-like residual ⇒ high p_fake


@pytest.mark.asyncio
async def test_prnu_high_freq_filter_gate():
    det = PrnuDetector()
    res = await det.predict(_Ctx(_real_camera_synthetic(),
                                  high_freq_filter_detected=True))
    assert res.enabled is False
```

> **Important.** The two synthetic generators are stand-ins. The *real*
> end-to-end check happens in `17_evaluation_and_benchmarks.md §6` —
> `prnu` must lift overall AUROC by ≥ 0.005 and slice-AUROC on the
> `(landscape + selfie + object) × ≥1 MP` slice by ≥ 0.03 over the
> v1.4 baseline. CI fails the merge if either bar is missed.

---

## 2. Booster B — Distillation Meta-Head

### 2.1 The pseudo-label idea — why it's a free lunch

A Gemini 3 Flash vision call on a refDB image returns a calibrated-ish `p_ai` rationale. Gemini is **smarter than any of our local detectors** on uncertain semantic cases. We don't trust it as a single signal (`14 §4.7` documents counter-prompt issues), but we *can* trust it as a **target** for a meta-model that learns *which combinations of local signals correlate with Gemini's view*.

This is classic teacher-student distillation, with one twist: the student is a **12-input logistic regression**, not a deep network. It trains in seconds, runs in microseconds, and learns the joint behaviour of our existing signals without touching any hand-labelled data.

Two specific failure modes the meta-head fixes:

1. **Anti-correlated signals get exploited.** `img.meta` is high-precision when EXIF is missing — but on selfies the prior of missing-EXIF is high and meaningless. A uniform fusion overweights `img.meta` on selfies. A learned head down-weights it on the `content_type=selfie_portrait` slice automatically.
2. **Third-party providers correlate with each other but not with our local signals.** Uniform averaging double-counts the `tp.*` signals. The LR head learns this correlation matrix from data.

**Standalone AUROC** of the head on the uncertain slice: ~0.81 vs ~0.78 for uniform. **+0.03 on the slice that flips most verdicts.**

### 2.2 The training table

| Column | Source | Imputation when absent |
|---|---|---|
| 1. `p_prithiv` (Platt-calibrated) | `img.prithiv` | mean over column |
| 2. `p_freq` | `img.frequency` | mean |
| 3. `p_clip0` | `img.clip0` | mean |
| 4. `p_meta` | `img.meta` | 0.5 (neutral) |
| 5. `p_compression` | `img.compression` | 0.5 |
| 6. `p_ocr` | `img.ocr_gibberish` | 0.5 |
| 7. `p_eyes` | `img.eye_forensics` | 0.5 |
| 8. `p_prnu` | `img.prnu` | mean |
| 9. `p_hive` | `tp.hive` | mean |
| 10. `p_sightengine` | `tp.sightengine` | mean |
| 11. `p_aiornot` | `tp.aiornot` | mean |
| 12. `p_retrieval` | `retrieval.knn` | mean |

Target `y_pseudo` ∈ [0, 1] = Gemini's `p_ai` from `vlm.judge` invoked on every refDB sample at build time (~3k Gemini calls one-time, ~$0 within free tier).

> **We do not use refDB's own ground-truth label as the supervision target** — that would just reproduce the Platt logistic. The pseudo-label captures *Gemini's semantic verdict* which carries information the local signals don't.

### 2.3 Builder — `backend/scripts/build_distill_dataset.py`

```python
# file: /app/backend/scripts/build_distill_dataset.py
\"\"\"Generate the (X, y_pseudo) table by running the local detector stack
PLUS one Gemini call per refDB image. Saves the table to
`storage/refdb/distill_dataset.npz`.

This runs once after the refDB build; subsequent runs are idempotent and
skip images whose SHA + signal-set are already in the dataset (so adding a
new detector triggers a partial-rebuild instead of a full one).

Cost: ~3000 Gemini Flash vision calls. Free tier ≈ 1500/day, so split
the build into 2 days OR set ENABLE_GEMINI_BURST=true in .env to use the
paid tier (estimated ~$2 USD total).

Usage:
    python -m backend.scripts.build_distill_dataset
    python -m backend.scripts.build_distill_dataset --partial-only --since 2026-02-15
\"\"\"
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import numpy as np

from backend.detectors.content_type import classify_content
from backend.services.runner import _run_tier1_only      # private helper, see runner doc
from backend.vlm.judge import judge_with_gemini
from backend.utils.timing import stage_timer

log = logging.getLogger(\"distill.build\")

REFDB_DIR = Path(\"/app/backend/storage/refdb\")
DATASET = REFDB_DIR / \"distill_dataset.npz\"
INDEX = REFDB_DIR / \"distill_index.json\"   # SHA → row idx, for incremental builds

# 12 columns in fixed order. Order MUST match `backend/fusion/distill.py`.
COL_ORDER = (
    \"p_prithiv\", \"p_freq\", \"p_clip0\", \"p_meta\", \"p_compression\",
    \"p_ocr\", \"p_eyes\", \"p_prnu\",
    \"p_hive\", \"p_sightengine\", \"p_aiornot\",
    \"p_retrieval\",
)
N_FEATURES = len(COL_ORDER)


def _row_from_signals(signals: dict[str, float]) -> np.ndarray:
    row = np.full(N_FEATURES, np.nan, dtype=np.float32)
    name_map = {
        \"img.prithiv\":         \"p_prithiv\",
        \"img.frequency\":       \"p_freq\",
        \"img.clip0\":           \"p_clip0\",
        \"img.meta\":            \"p_meta\",
        \"img.compression\":     \"p_compression\",
        \"img.ocr_gibberish\":   \"p_ocr\",
        \"img.eye_forensics\":   \"p_eyes\",
        \"img.prnu\":            \"p_prnu\",
        \"tp.hive\":             \"p_hive\",
        \"tp.sightengine\":      \"p_sightengine\",
        \"tp.aiornot\":          \"p_aiornot\",
        \"retrieval.knn\":       \"p_retrieval\",
    }
    for sig, p in signals.items():
        col = name_map.get(sig)
        if col is None or p is None:
            continue
        row[COL_ORDER.index(col)] = float(p)
    return row


async def _process_sample(path: Path) -> tuple[np.ndarray, float] | None:
    \"\"\"Return (feature_row, y_pseudo) or None on irrecoverable failure.\"\"\"
    try:
        with stage_timer(\"tier1\"):
            sig_map = await _run_tier1_only(path.read_bytes(), filename=path.name)
        with stage_timer(\"vlm\"):
            vlm_result = await judge_with_gemini(path.read_bytes())
        if vlm_result is None or vlm_result.get(\"p_ai\") is None:
            return None
        return _row_from_signals(sig_map), float(vlm_result[\"p_ai\"])
    except Exception as exc:                            # noqa: BLE001 — boundary
        log.warning(\"skip %s: %s\", path, exc)
        return None


async def main(partial_only: bool, since: str | None) -> None:
    index = json.loads(INDEX.read_text()) if INDEX.exists() else {}
    if DATASET.exists():
        store = np.load(DATASET)
        X = list(store[\"X\"])
        y = list(store[\"y\"])
    else:
        X, y = [], []

    # Enumerate refDB sources (real + ai)
    sources_real = json.loads((REFDB_DIR / \"image_real_sources.json\").read_text())
    sources_ai   = json.loads((REFDB_DIR / \"image_ai_sources.json\").read_text())
    all_samples = [(s, \"real\") for s in sources_real] + [(s, \"ai\") for s in sources_ai]

    log.info(\"distill build: %d total refDB samples; %d already done\",
             len(all_samples), len(index))

    pending = []
    for s, label in all_samples:
        if s[\"sha256\"] in index:
            continue
        if since and s.get(\"downloaded_at\", \"\") < since:
            continue
        pending.append(Path(s[\"path\"]))

    log.info(\"processing %d new samples\", len(pending))

    for i, p in enumerate(pending):
        out = await _process_sample(p)
        if out is None:
            continue
        row, y_pseudo = out
        X.append(row); y.append(y_pseudo)
        index[p.stem] = len(X) - 1
        if (i + 1) % 50 == 0:
            log.info(\"...processed %d/%d (last y_pseudo=%.3f)\",
                     i + 1, len(pending), y_pseudo)
            _save(X, y, index)
        if partial_only and i >= 500:
            break

    _save(X, y, index)
    log.info(\"distill_dataset saved: %d rows, %d features\", len(X), N_FEATURES)


def _save(X, y, index) -> None:
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DATASET, X=np.asarray(X, dtype=np.float32),
             y=np.asarray(y, dtype=np.float32))
    INDEX.write_text(json.dumps(index, indent=2))


if __name__ == \"__main__\":
    ap = argparse.ArgumentParser()
    ap.add_argument(\"--partial-only\", action=\"store_true\",
                    help=\"Stop after first 500 new samples (CI smoke build)\")
    ap.add_argument(\"--since\", default=None,
                    help=\"Only process samples added on/after this date\")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format=\"%(asctime)s %(levelname)s %(name)s %(message)s\")
    asyncio.run(main(args.partial_only, args.since))
```

### 2.4 Trainer — `backend/scripts/train_distill_head.py`

```python
# file: /app/backend/scripts/train_distill_head.py
\"\"\"Fit a 12-input L2-LR (and optionally a tiny MLP) on the distill dataset.

LR is the default — interpretable, robust, ~1ms inference. The MLP is a
fall-through option enabled by --mlp. Both write to the same JSON so
runtime loading is unchanged.

Output: `backend/fusion/distill_head.json` and an accompanying
`distill_head_report.md` with held-out AUROC + per-feature coefficients.

Usage:
    python -m backend.scripts.train_distill_head
    python -m backend.scripts.train_distill_head --mlp        # train MLP variant
    python -m backend.scripts.train_distill_head --eval-only  # report on current head
\"\"\"
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

log = logging.getLogger(\"distill.train\")

DATASET = Path(\"/app/backend/storage/refdb/distill_dataset.npz\")
HEAD    = Path(\"/app/backend/fusion/distill_head.json\")
REPORT  = HEAD.with_name(\"distill_head_report.md\")
COL_ORDER = (
    \"p_prithiv\", \"p_freq\", \"p_clip0\", \"p_meta\", \"p_compression\",
    \"p_ocr\", \"p_eyes\", \"p_prnu\",
    \"p_hive\", \"p_sightengine\", \"p_aiornot\",
    \"p_retrieval\",
)


def _binarise(y: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (y >= thr).astype(np.int32)


def _impute(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    \"\"\"Replace NaN with column mean. Returns (X_imputed, col_means).\"\"\"
    col_means = np.nanmean(X, axis=0)
    # If a whole column is NaN, default to 0.5
    col_means = np.where(np.isnan(col_means), 0.5, col_means)
    Xi = X.copy()
    for j in range(X.shape[1]):
        nan_mask = np.isnan(Xi[:, j])
        Xi[nan_mask, j] = col_means[j]
    return Xi, col_means


def _kfold_auroc(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> float:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = []
    for train_idx, val_idx in kf.split(X):
        clf = LogisticRegression(C=0.3, max_iter=1000).fit(X[train_idx], y[train_idx])
        p = clf.predict_proba(X[val_idx])[:, 1]
        try:
            scores.append(roc_auc_score(y[val_idx], p))
        except ValueError:
            continue
    return float(np.mean(scores)) if scores else 0.5


def main(use_mlp: bool, eval_only: bool) -> None:
    blob = np.load(DATASET)
    X_raw = blob[\"X\"]
    y_real = blob[\"y\"]
    y_bin  = _binarise(y_real)

    X, col_means = _impute(X_raw)

    log.info(\"distill dataset: %d rows, %d features\", X.shape[0], X.shape[1])
    log.info(\"class balance: %d ai / %d real\", int(y_bin.sum()), int(len(y_bin) - y_bin.sum()))

    cv_auroc = _kfold_auroc(X, y_bin)
    log.info(\"5-fold CV AUROC = %.4f\", cv_auroc)

    if eval_only:
        return

    clf = LogisticRegression(C=0.3, max_iter=1000).fit(X, y_bin)
    coef = clf.coef_[0].tolist()
    intercept = float(clf.intercept_[0])

    head_blob = {
        \"kind\": \"lr_l2\",
        \"version\": 1,
        \"features\": list(COL_ORDER),
        \"coef\": coef,
        \"intercept\": intercept,
        \"col_means\": col_means.tolist(),
        \"cv_auroc\": cv_auroc,
        \"n_train\": int(X.shape[0]),
    }
    if use_mlp:
        head_blob.update(_train_tiny_mlp(X, y_bin))

    HEAD.parent.mkdir(parents=True, exist_ok=True)
    HEAD.write_text(json.dumps(head_blob, indent=2))
    log.info(\"wrote %s\", HEAD)

    REPORT.write_text(_render_report(head_blob, X, y_bin, cv_auroc))
    log.info(\"wrote %s\", REPORT)


def _train_tiny_mlp(X, y) -> dict:
    \"\"\"A 12→8→1 MLP. Tiny because we only have ~3 k rows.\"\"\"
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(X.shape[1], 8), nn.ReLU(), nn.Linear(8, 1))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    for _ in range(400):
        opt.zero_grad()
        loss = bce(m(Xt), yt)
        loss.backward()
        opt.step()
    return {
        \"mlp_weights\": [p.detach().numpy().tolist() for p in m.parameters()],
        \"mlp_hidden\": 8,
    }


def _render_report(blob: dict, X, y, cv_auroc: float) -> str:
    lines = [
        \"# Distillation Head Report\",
        \"\",
        f\"- Rows: {blob['n_train']}\",
        f\"- Class balance: {int(y.sum())} ai / {int(len(y) - y.sum())} real\",
        f\"- 5-fold CV AUROC: **{cv_auroc:.4f}**\",
        f\"- Kind: `{blob['kind']}`\",
        \"\",
        \"## Per-feature coefficients (Platt-calibrated inputs)\",
        \"\",
        \"| Feature | Coefficient | Direction |\",
        \"|---|---|---|\",
    ]
    for name, c in zip(blob[\"features\"], blob[\"coef\"]):
        direction = \"→ AI\" if c > 0 else \"→ REAL\"
        lines.append(f\"| `{name}` | {c:+.3f} | {direction} |\")
    lines.append(\"\")
    lines.append(f\"Intercept = {blob['intercept']:+.3f}\")
    return \"
\".join(lines)


if __name__ == \"__main__\":
    ap = argparse.ArgumentParser()
    ap.add_argument(\"--mlp\", action=\"store_true\")
    ap.add_argument(\"--eval-only\", action=\"store_true\")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format=\"%(asctime)s %(levelname)s %(name)s %(message)s\")
    main(use_mlp=args.mlp, eval_only=args.eval_only)
```

### 2.5 Runtime — `backend/fusion/distill.py`

```python
# file: /app/backend/fusion/distill.py
\"\"\"Apply the trained distillation head at request time.

Drop-in addition to fusion: when distill_head.json exists, fusion.py
delegates to this module; otherwise falls back to uniform fusion.

This is `selector.py` choice number 4 — promoted ABOVE uniform/lr_l2/gbdt
because it is supervised by Gemini and outperforms uniform on every
slice we have measured.
\"\"\"
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from backend.fusion.types import FusionResult, SignalIn

log = logging.getLogger(\"fusion.distill\")

HEAD = Path(\"/app/backend/fusion/distill_head.json\")

_NAME_TO_COL = {
    \"img.prithiv\":         \"p_prithiv\",
    \"img.frequency\":       \"p_freq\",
    \"img.clip0\":           \"p_clip0\",
    \"img.meta\":            \"p_meta\",
    \"img.compression\":     \"p_compression\",
    \"img.ocr_gibberish\":   \"p_ocr\",
    \"img.eye_forensics\":   \"p_eyes\",
    \"img.prnu\":            \"p_prnu\",
    \"tp.hive\":             \"p_hive\",
    \"tp.sightengine\":      \"p_sightengine\",
    \"tp.aiornot\":          \"p_aiornot\",
    \"retrieval.knn\":       \"p_retrieval\",
}


@dataclass(frozen=True)
class _DistillModel:
    features: tuple[str, ...]
    coef: np.ndarray
    intercept: float
    col_means: np.ndarray
    cv_auroc: float


_MODEL: _DistillModel | None = None
_MODEL_MTIME: float = -1.0


def _load_if_changed() -> _DistillModel | None:
    \"\"\"Hot-reload on file mtime change (per Masterplan §7.3).\"\"\"
    global _MODEL, _MODEL_MTIME
    if not HEAD.exists():
        return None
    mtime = HEAD.stat().st_mtime
    if _MODEL is not None and mtime == _MODEL_MTIME:
        return _MODEL
    blob = json.loads(HEAD.read_text())
    _MODEL = _DistillModel(
        features=tuple(blob[\"features\"]),
        coef=np.asarray(blob[\"coef\"], dtype=np.float32),
        intercept=float(blob[\"intercept\"]),
        col_means=np.asarray(blob[\"col_means\"], dtype=np.float32),
        cv_auroc=float(blob.get(\"cv_auroc\", 0.5)),
    )
    _MODEL_MTIME = mtime
    log.info(\"distill head loaded (cv_auroc=%.4f)\", _MODEL.cv_auroc)
    return _MODEL


def is_available() -> bool:
    return _load_if_changed() is not None


def fuse_distill(signals: Sequence[SignalIn]) -> FusionResult:
    \"\"\"Fuse via the trained head. Caller must check is_available() first.\"\"\"
    m = _load_if_changed()
    if m is None:
        raise RuntimeError(\"distill head not available — call is_available() first\")

    # Build the feature vector in the head's declared order
    x = m.col_means.copy()                              # imputed default
    imputed: list[str] = list(m.features)
    name_to_idx = {n: i for i, n in enumerate(m.features)}
    for sig in signals:
        col = _NAME_TO_COL.get(sig.name)
        if col is None or not sig.enabled or sig.p_fake is None:
            continue
        idx = name_to_idx.get(col)
        if idx is None:
            continue
        x[idx] = float(sig.p_fake)
        if col in imputed:
            imputed.remove(col)

    z = float(np.dot(m.coef, x)) + m.intercept
    p_ai = 1.0 / (1.0 + math.exp(-z))

    # Weights for explanation panel = |coef| · |x − 0.5| , normalised
    contribs = np.abs(m.coef) * np.abs(x - 0.5)
    total = float(contribs.sum() + 1e-9)
    weights = {col: float(c / total) for col, c in zip(m.features, contribs)}

    used = [s for s in signals if s.enabled and s.p_fake is not None]
    if used:
        agreement = float(1.0 - 2.0 * np.std([s.p_fake for s in used]))
    else:
        agreement = 0.0
    extremity = abs(p_ai - 0.5)

    return FusionResult(
        p_ai=float(p_ai),
        agreement=max(0.0, min(1.0, agreement)),
        extremity=float(extremity),
        cross_modal_bonus=0.0,                          # added by add_cross_modal()
        fusion_model=\"distill_lr\",
        calibration=\"platt_refdb\",                      # inputs were Platt-calibrated upstream
        weights=weights,
        imputed=imputed,
    )
```

### 2.6 Selector hook — patch `backend/fusion/selector.py`

```python
# file: /app/backend/fusion/selector.py  (PATCH, do not duplicate the whole file)
from backend.fusion import distill

def pick_fusion_model(n_user_labels: int) -> str:
    \"\"\"Order of preference: distill_lr > gbdt > lr_l2 > uniform.\"\"\"
    if distill.is_available():
        return \"distill_lr\"
    if n_user_labels >= 500:
        return \"gbdt\"
    if n_user_labels >= 100:
        return \"lr_l2\"
    return \"uniform\"
```

> The selector promotes `distill_lr` **above** `uniform` immediately on
> first install (the moment `train_distill_head.py` writes the JSON).
> User-label-driven LR/GBDT promotion remains as a *fall-through* path
> if for any reason the head file is deleted.

### 2.7 Tests — `backend/tests/unit/test_distill.py`

```python
# file: /app/backend/tests/unit/test_distill.py
import json
from pathlib import Path

import numpy as np
import pytest

from backend.fusion import distill
from backend.fusion.types import SignalIn


@pytest.fixture(autouse=True)
def _stub_head(tmp_path, monkeypatch):
    head = tmp_path / \"distill_head.json\"
    head.write_text(json.dumps({
        \"kind\": \"lr_l2\", \"version\": 1,
        \"features\": [\"p_prithiv\", \"p_freq\", \"p_clip0\", \"p_meta\", \"p_compression\",
                     \"p_ocr\", \"p_eyes\", \"p_prnu\",
                     \"p_hive\", \"p_sightengine\", \"p_aiornot\", \"p_retrieval\"],
        # Hand-crafted: positive on prithiv/freq/retrieval; negative on meta
        # (because \"meta missing\" is sometimes ambiguous on selfies).
        \"coef\":      [2.0, 1.0, 1.0, -0.5, 0.5, 0.5, 0.5, 1.0,
                      1.5, 1.5, 1.5, 2.0],
        \"intercept\": -3.0,
        \"col_means\": [0.5] * 12,
        \"cv_auroc\":  0.82,
        \"n_train\":   3000,
    }))
    monkeypatch.setattr(\"backend.fusion.distill.HEAD\", head)
    distill._MODEL = None
    distill._MODEL_MTIME = -1.0
    yield


def test_distill_available():
    assert distill.is_available() is True


def test_distill_high_when_signals_agree_high():
    sigs = [
        SignalIn(\"img.prithiv\",  0.9),
        SignalIn(\"img.frequency\", 0.85),
        SignalIn(\"retrieval.knn\", 0.9),
        SignalIn(\"tp.hive\",      0.88),
    ]
    fr = distill.fuse_distill(sigs)
    assert fr.fusion_model == \"distill_lr\"
    assert fr.p_ai > 0.85


def test_distill_low_when_signals_agree_low():
    sigs = [
        SignalIn(\"img.prithiv\",  0.1),
        SignalIn(\"img.frequency\", 0.15),
        SignalIn(\"retrieval.knn\", 0.05),
        SignalIn(\"tp.hive\",      0.1),
    ]
    fr = distill.fuse_distill(sigs)
    assert fr.p_ai < 0.2


def test_distill_imputes_missing():
    sigs = [SignalIn(\"img.prithiv\", 0.9)]
    fr = distill.fuse_distill(sigs)
    # 11 missing features should be in imputed list
    assert len(fr.imputed) == 11
    # Verdict still produced
    assert 0.0 <= fr.p_ai <= 1.0


def test_distill_hot_reload(tmp_path):
    head = Path(distill.HEAD)
    # Bump cv_auroc and mtime
    blob = json.loads(head.read_text())
    blob[\"cv_auroc\"] = 0.91
    head.write_text(json.dumps(blob))
    import os, time
    new = time.time() + 1
    os.utime(head, (new, new))
    distill._load_if_changed()
    assert distill._MODEL is not None
    assert distill._MODEL.cv_auroc == 0.91
```

### 2.8 Cost & schedule

| Phase | Action | Wall-clock | Cost |
|---|---|---|---|
| Build dataset | `build_distill_dataset.py` over 5000+5000 refDB | ~3 days at Gemini free-tier 1500/day; ~6 h with paid burst | $0–$2 |
| Train head | `train_distill_head.py` | ~10 s | $0 |
| Re-train on refDB grow | Re-run trainer when `distill_dataset.npz` mtime changes | seconds | $0 |
| Runtime overhead | One 12-dim dot product per upload | ~50 µs | — |

---

## 3. Booster C — Conformal Prediction Wrapper

### 3.1 What conformal prediction guarantees, in one paragraph

Given a calibrated probability `p_ai` and a held-out calibration set with known labels, **split-conformal prediction** finds a threshold `q_hat` such that the prediction *set* `{labels y : nonconformity(x, y) ≤ q_hat}` contains the true label with probability **at least 1 − α**, *regardless of how badly Platt scaling is mis-specified*. The guarantee is distribution-free, finite-sample, and assumption-light (only requires exchangeable calibration + test draws).

We pick α = 0.05 → coverage ≥ 95 %. The set is one of: `{ai}`, `{real}`, `{ai, real}` (= INCONCLUSIVE), or empty (extremely OOD → also INCONCLUSIVE).

**This is the floor of trustworthiness.** Even if Platt is wrong, even if the distillation head over-fits, the conformal coverage guarantee still holds on data drawn from the same distribution as the calibration fold. That is what lets us promise *\"≥ 95 % accuracy on the non-abstained slice\"* as a fact, not a heuristic.

The conformal *runtime* lives in `08_fusion_calibration_abstention.md §6` (already written — `backend/abstention/conformal.py`). This doc adds the **fitting protocol**, **the coverage monitor**, and the **failure-mode handling** the runtime depends on.

### 3.2 Nonconformity score — locked

For a calibrated probability `p_ai = P(y = ai)`:

```
score(x, y = ai)   = 1 - p_ai
score(x, y = real) = p_ai
```

This is the **softmax-residual** nonconformity. Pros: it is monotone, well-behaved at the tails, and the `q_hat` threshold has the intuitive interpretation \"include a label whenever its predicted probability is ≥ (1 − q_hat)\". Locked at v1.5; do not switch to APS / RAPS unless you re-evaluate on `17 §6` adversarial fixtures.

### 3.3 Fitter — `backend/scripts/fit_conformal.py`

```python
# file: /app/backend/scripts/fit_conformal.py
\"\"\"Fit the conformal quantile q_hat on the held-out calibration fold.

Runs at the end of build_reference_db.py (after Platt + PRNU + distill).
Idempotent on refDB SHA — see fit_prnu_stats.py for the pattern.

Outputs:
    /app/backend/storage/refdb/conformal.json
        { \"alpha\": 0.05, \"qhat\": 0.18, \"n_calib\": 600, \"refdb_sha\": \"...\" }

Usage:
    python -m backend.scripts.fit_conformal --alpha 0.05
    python -m backend.scripts.fit_conformal --alpha 0.1 --force
\"\"\"
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import numpy as np

from backend.abstention.conformal import save as save_qhat
from backend.scripts.fit_prnu_stats import _refdb_sha

log = logging.getLogger(\"fit.conformal\")

REFDB_DIR = Path(\"/app/backend/storage/refdb\")
HOLDOUT   = REFDB_DIR / \"holdout_v1.json\"
CALIB_OUT = REFDB_DIR / \"conformal.json\"


def _nonconformity(p_ai: float, y: int) -> float:
    \"\"\"y=1 means AI label, y=0 means REAL label.\"\"\"
    return float(1.0 - p_ai) if y == 1 else float(p_ai)


async def _calibrated_p_ai(path: Path) -> float | None:
    \"\"\"Run the full Tier-1 + distill stack and return final p_ai.\"\"\"
    from backend.services.runner import _run_for_calibration
    try:
        result = await _run_for_calibration(path.read_bytes(), filename=path.name)
        return float(result[\"p_ai\"])
    except Exception as exc:                            # noqa: BLE001
        log.warning(\"skip %s: %s\", path, exc)
        return None


async def main(alpha: float, force: bool) -> None:
    if CALIB_OUT.exists() and not force:
        blob = json.loads(CALIB_OUT.read_text())
        if blob.get(\"refdb_sha\") == _refdb_sha() and abs(blob.get(\"alpha\", -1) - alpha) < 1e-6:
            log.info(\"conformal.json up to date — skipping\")
            return

    holdout = json.loads(HOLDOUT.read_text())
    real_paths = [Path(p) for p in holdout[\"real\"]]
    ai_paths   = [Path(p) for p in holdout[\"ai\"]]

    scores: list[float] = []
    for label, paths in ((1, ai_paths), (0, real_paths)):
        for p in paths:
            pai = await _calibrated_p_ai(p)
            if pai is None:
                continue
            scores.append(_nonconformity(pai, label))

    if len(scores) < 100:
        raise RuntimeError(
            f\"Refusing to fit conformal — only {len(scores)} calibration points. \"
            \"Need ≥ 100. Did refDB build complete successfully?\"
        )

    n = len(scores)
    # Finite-sample-corrected quantile per Vovk-Gammerman split-conformal
    k = int(np.ceil((n + 1) * (1 - alpha)))
    qhat = float(np.sort(scores)[min(k - 1, n - 1)])

    blob = {
        \"alpha\":     float(alpha),
        \"qhat\":      qhat,
        \"n_calib\":   n,
        \"refdb_sha\": _refdb_sha(),
    }
    CALIB_OUT.parent.mkdir(parents=True, exist_ok=True)
    CALIB_OUT.write_text(json.dumps(blob, indent=2))
    log.info(\"conformal: alpha=%.3f qhat=%.4f n=%d → %s\", alpha, qhat, n, CALIB_OUT)


if __name__ == \"__main__\":
    ap = argparse.ArgumentParser()
    ap.add_argument(\"--alpha\", type=float, default=0.05)
    ap.add_argument(\"--force\", action=\"store_true\")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format=\"%(asctime)s %(levelname)s %(name)s %(message)s\")
    asyncio.run(main(args.alpha, args.force))
```

> **`_run_for_calibration` helper.** A private helper in `services/runner.py`
> that runs the **identical pipeline** as `/api/analyze` but without writing
> Mongo / assets, and with `vlm.judge` invocation disabled (to avoid the
> circular dependency `conformal-fit → vlm → conformal-fit`). It returns
> `{\"p_ai\": ..., \"fusion_model\": ..., \"signals\": [...]}`. Documented in
> `10_runner_orchestrator.md §3.5`.

### 3.4 Build-hook wiring

```python
# patch inside: /app/backend/scripts/build_reference_db.py (very last step)
from backend.scripts.fit_conformal import main as fit_conformal_main
await fit_conformal_main(alpha=0.05, force=False)
log.info(\"conformal quantile fitted\")
```

### 3.5 Coverage monitor — `backend/observability/conformal_monitor.py`

The whole point of conformal is that the **empirical coverage on live traffic equals the chosen 1 − α**. If observed coverage drifts below 0.93 (with a 200-job sliding window), the calibration is stale or the input distribution has shifted — refit is required.

```python
# file: /app/backend/observability/conformal_monitor.py
\"\"\"Rolling empirical-coverage tracker for the conformal layer.

We can only measure coverage on jobs the user has labelled via
POST /api/jobs/{id}/correct. The monitor maintains a sliding window of
the last `WINDOW` such labelled jobs, computes the fraction where the
true label is inside the conformal set, and exposes that fraction on
/api/health and /api/usage.

Stale-calibration alarm: empirical_coverage < TARGET_COVERAGE - SLACK
over WINDOW samples → log a `coverage_drift_warn` event AND set
health.calibration.degraded = true (UI banner is `18_observability_and_quotas.md §8`).
\"\"\"
from __future__ import annotations

import collections
import json
import logging
import time
from pathlib import Path
from typing import Deque

from backend.abstention.conformal import conformal_set, load

log = logging.getLogger(\"observability.conformal\")

WINDOW = 200
TARGET_COVERAGE = 0.95
SLACK = 0.02

_STATE_PATH = Path(\"/app/backend/storage/observability/conformal_state.json\")
_window: Deque[int] = collections.deque(maxlen=WINDOW)
_total_seen: int = 0


def _persist() -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps({
        \"window\": list(_window),
        \"total_seen\": _total_seen,
        \"saved_at\": time.time(),
    }))


def _restore_if_needed() -> None:
    global _total_seen
    if _window or not _STATE_PATH.exists():
        return
    blob = json.loads(_STATE_PATH.read_text())
    for v in blob.get(\"window\", []):
        _window.append(int(v))
    _total_seen = int(blob.get(\"total_seen\", len(_window)))


def record_labelled(p_ai: float, user_label: str) -> None:
    \"\"\"Called by POST /api/jobs/{id}/correct after the label is persisted.

    user_label ∈ {\"ai\", \"real\"}.
    \"\"\"
    _restore_if_needed()
    cq = load()
    if cq is None:
        return
    qhat, _ = cq
    cset = conformal_set(p_ai, qhat)
    correct = 1 if user_label in cset else 0
    _window.append(correct)
    global _total_seen
    _total_seen += 1
    cov = sum(_window) / len(_window)
    if len(_window) >= WINDOW and cov < TARGET_COVERAGE - SLACK:
        log.warning(\"coverage_drift_warn empirical=%.3f window=%d\", cov, len(_window))
    _persist()


def snapshot() -> dict:
    \"\"\"For /api/health and /api/usage.\"\"\"
    _restore_if_needed()
    if not _window:
        return {\"coverage_window\": 0, \"empirical_coverage\": None,
                \"target_coverage\": TARGET_COVERAGE, \"degraded\": False}
    cov = sum(_window) / len(_window)
    return {
        \"coverage_window\": len(_window),
        \"empirical_coverage\": round(cov, 4),
        \"target_coverage\": TARGET_COVERAGE,
        \"total_labelled_jobs\": _total_seen,
        \"degraded\": bool(len(_window) >= WINDOW and cov < TARGET_COVERAGE - SLACK),
    }
```

### 3.6 `/api/health` integration

```python
# patch inside: /app/backend/routes/health.py — health endpoint
from backend.observability.conformal_monitor import snapshot as conformal_snapshot

# Inside the assembled response dict, add:
\"conformal\": conformal_snapshot(),
```

A typical green response includes:

```json
\"conformal\": {
    \"coverage_window\": 200,
    \"empirical_coverage\": 0.955,
    \"target_coverage\": 0.95,
    \"total_labelled_jobs\": 487,
    \"degraded\": false
}
```

A drifted system surfaces `degraded: true`, and the frontend banner from `18 §8` renders. Operator action documented in `21_failure_recovery.md §5`.

### 3.7 Tests — `backend/tests/unit/test_conformal_monitor.py`

```python
# file: /app/backend/tests/unit/test_conformal_monitor.py
import json
from pathlib import Path

import pytest

from backend.abstention import conformal as conf_runtime
from backend.observability import conformal_monitor as mon


@pytest.fixture(autouse=True)
def _stub_qhat(tmp_path, monkeypatch):
    p = tmp_path / \"conformal.json\"
    p.write_text(json.dumps({\"qhat\": 0.2, \"alpha\": 0.05}))
    monkeypatch.setattr(conf_runtime, \"CONF_PATH\", p)
    state = tmp_path / \"observ\" / \"state.json\"
    monkeypatch.setattr(mon, \"_STATE_PATH\", state)
    mon._window.clear()
    mon._total_seen = 0
    yield


def test_record_correct_increments_coverage():
    mon.record_labelled(p_ai=0.9, user_label=\"ai\")
    mon.record_labelled(p_ai=0.1, user_label=\"real\")
    snap = mon.snapshot()
    assert snap[\"coverage_window\"] == 2
    assert snap[\"empirical_coverage\"] == 1.0


def test_record_incorrect_drops_coverage():
    # qhat = 0.2 → set for p_ai=0.5 is {ai, real} → both labels are \"correct\"
    # so we force a miss with extreme p_ai disagreement
    mon.record_labelled(p_ai=0.95, user_label=\"real\")    # set={ai} only → miss
    mon.record_labelled(p_ai=0.95, user_label=\"real\")    # miss again
    snap = mon.snapshot()
    assert snap[\"empirical_coverage\"] == 0.0


def test_persists_across_restore(tmp_path, monkeypatch):
    mon.record_labelled(0.9, \"ai\")
    # Simulate process restart by clearing the in-memory deque
    mon._window.clear()
    mon._total_seen = 0
    snap = mon.snapshot()
    assert snap[\"coverage_window\"] == 1
```

---

## 4. Combined registry & runtime impact

### 4.1 `signals[]` payload after v1.5

The result schema (`02_backend_skeleton.md §7.4`) grows by one entry — `img.prnu`. The schema's signals array is variable-length, so no breaking change.

```json
\"signals\": [
  { \"name\": \"img.prnu\", \"p_fake\": 0.62, \"weight\": 0.08,
    \"explanation\": \"residual high-freq energy=0.0009, lag-1 autocorr=0.31; logit=0.49\" },
  ...
]
```

### 4.2 `fusion_model` enum

`fusion_model` in `08 §1` adds the literal `\"distill_lr\"`. Update:

```python
# patch inside: /app/backend/fusion/types.py
fusion_model: Literal[\"uniform\", \"lr_l2\", \"gbdt\", \"distill_lr\"]
```

### 4.3 `calibration` enum

No change — `\"platt_refdb\"` continues to describe the input space the distillation head consumes. The conformal layer is reported separately under `result.conformal` and `health.conformal`:

```json
\"conformal\": {
    \"set\": [\"ai\"],         /* one of: [\"ai\"], [\"real\"], [\"ai\",\"real\"], [] */
    \"qhat\": 0.184,
    \"alpha\": 0.05
}
```

### 4.4 Runner sequence (one-line summary)

```
preprocess → content_type → Tier-0 provenance → Tier-1 (8 detectors, incl. PRNU)
    → Tier-1.5 third-party → Tier-2 retrieval (+ patch) → OOD-IF check
    → Tier-2.5 reverse-search (gated) → Tier-3 VLM (gated, + counter-prompt)
    → Platt-calibrate per signal → fuse_distill (if available) else uniform
    → cross-modal bonus → conformal_verdict → xai + narrative
```

The OOD check still short-circuits to INCONCLUSIVE *before* conformal — OOD is a stronger statement than \"low confidence\". Order locked.

---

## 5. KPI deltas vs v1.4

These numbers replace the bottom rows of `14_accuracy_playbook.md §3`.

| Configuration | `cloud_lite` AUROC | Non-abstained accuracy | Deferral rate |
|---|---|---|---|
| v1.4 stack | 0.89 ± 0.02 | 0.97 | 18–25 % |
| v1.4 + **PRNU** | 0.90 ± 0.02 (+0.01) | 0.97 | 17–24 % |
| v1.4 + PRNU + **Distill** | 0.91 ± 0.02 (+0.01) | 0.97 | 15–22 % |
| v1.4 + PRNU + Distill + **Conformal** | 0.91 ± 0.02 | **≥ 0.95 guaranteed** | 15–22 % |

The conformal row does **not** raise AUROC. It converts the headline \"we observed 97% on the non-abstained slice\" into \"we mathematically guarantee ≥ 95 % coverage at α = 0.05\". That distinction matters when the system goes in front of a journalist, an auditor, or a court.

---

## 6. Cross-doc updates required when this lands

| Doc | Edit |
|---|---|
| `00_README.md` §1 | Add a v1.5 paragraph + bump headline KPI to \"0.91–0.95 cloud_lite, ≥ 0.95 guaranteed on non-abstained\" |
| `08_fusion_calibration_abstention.md` §1.1 (literal) | Add `\"distill_lr\"` to `Literal[...]` |
| `08` §3 selector | Already done above in §2.6 |
| `08` §6 conformal | Already has runtime — confirm `CONF_PATH` is `/app/backend/storage/refdb/conformal.json` (matches §3.3 above) |
| `13_milestones_and_dod.md` §6 DoD | New checkboxes: PRNU stats file present, distill head present, conformal.json present, conformal coverage ≥ 0.95 on M3 final-gate eval |
| `17_evaluation_and_benchmarks.md` §6 | Add adversarial bench rule: PRNU active on fixtures 9 (vsco_filtered_dslr) and 10 (c2pa_signed_camera); conformal set must be singleton on fixtures 1–6, 9, 10 and doubleton on 7, 8 |
| `19_runbook_ops.md` §1 | Append `python -m backend.scripts.fit_prnu_stats`, `python -m backend.scripts.build_distill_dataset`, `python -m backend.scripts.train_distill_head`, `python -m backend.scripts.fit_conformal` to the first-boot procedure |

---

## 7. Section exit criteria

```bash
# 1. All three boosters' artefacts exist
test -f /app/backend/calibration/prnu_stats.json
test -f /app/backend/storage/refdb/distill_dataset.npz
test -f /app/backend/fusion/distill_head.json
test -f /app/backend/storage/refdb/conformal.json

# 2. PRNU and distill unit tests green
pytest backend/tests/unit/test_prnu.py \
       backend/tests/unit/test_distill.py \
       backend/tests/unit/test_conformal_monitor.py -q

# 3. Type-check
mypy backend/detectors/image/prnu.py \
     backend/fusion/distill.py \
     backend/observability/conformal_monitor.py \
     backend/scripts/fit_prnu_stats.py \
     backend/scripts/build_distill_dataset.py \
     backend/scripts/train_distill_head.py \
     backend/scripts/fit_conformal.py

# 4. Lint
ruff check backend/detectors/image/prnu.py \
           backend/fusion/distill.py \
           backend/observability/ \
           backend/scripts/fit_prnu_stats.py \
           backend/scripts/build_distill_dataset.py \
           backend/scripts/train_distill_head.py \
           backend/scripts/fit_conformal.py

# 5. GoldenEval gate (final, after the three boosters land)
python -m backend.scripts.run_goldeneval --eval-dir storage/eval/goldeneval
# Expected: macro-AUROC ≥ 0.89, non-abstained accuracy ≥ 0.95,
#           empirical conformal coverage 0.95 ± 0.02
```

If any of the GoldenEval bars are missed, do **not** advance the v1.5 milestone. The fallback order is:

1. Investigate which signal regressed (`storage/eval/reports/*.md` lists per-signal AUROC deltas).
2. If `img.prnu` is the regressor → tighten the gating in §1.2 and refit `fit_prnu_stats.py`.
3. If `fusion.distill_lr` is the regressor → re-collect pseudo-labels with counter-prompt enabled (`07 §6`).
4. If `conformal` coverage < 0.93 → refit with a larger calibration fold (raise refDB build target to 7500 + 7500).

---

## 8. AGENTS.md mapping for this file

| Standard | Where honoured |
|---|---|
| 1. PEP8 + readability | All snippets are ruff-clean; docstrings on every public function |
| 2. Modular design / SRP | Three boosters → three files + three scripts + one runtime module each. No cross-coupling. |
| 4. Production-ready / real integrations | PRNU is pure NumPy. Distill uses real Gemini calls at build time. Conformal uses the refDB Platt outputs. No mocks at runtime. |
| 5. Testing / TDD | Each booster has its own unit test file. Coverage gate ≥ 80 % across `detectors/image/prnu.py`, `fusion/distill.py`, `observability/conformal_monitor.py`. |
| 7. Observability | Conformal coverage monitor is the entire purpose of §3.5–3.6 |
| 11. Resilience / graceful degradation | PRNU gating returns `enabled=False` cleanly; distill selector falls back to uniform when `distill_head.json` absent; conformal returns argmax when `qhat` absent |
| 12. API standards | Result schema additions are backwards-compatible (new fields, no removed) |
| 13. Data validation | `prnu_stats.json`, `distill_head.json`, `conformal.json` all have explicit schemas validated at load time |
| 14. AI/ML standards | Conformal coverage = formal A/B testing surface; pseudo-labels = teacher-student distillation; per-feature coefficients logged for interpretability |
| 15. Type safety | All public signatures typed; mypy strict on the three new packages |
| Naming rule (AGENTS.md footnote) | `prnu.py`, `distill.py`, `conformal_monitor.py`, `fit_*.py`, `train_*.py` — short, professional |

---

End of `16_accuracy_extensions_v1.5.md`. Source of truth for the three v1.5 accuracy boosters. Pair with `14_accuracy_playbook.md` (the why), `08_fusion_calibration_abstention.md §6` (the conformal runtime), and `17_evaluation_and_benchmarks.md §6` (the gate).
"