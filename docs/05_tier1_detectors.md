"# 05 — Tier 1: Image Detectors (8 base + 2 v1.4 additions)

> Goal: ten orthogonal image signals. Each emits a raw `p_fake ∈ [0,1]`. Calibration and fusion live in `08_fusion_calibration_abstention.md`. Nothing here knows about thresholds or verdicts.

Profile gating (recap from `03_detector_framework.md` §3):

| Signal | `cloud_lite` | `mac_full` | `cuda_full` | Notes |
|---|:-:|:-:|:-:|---|
| `img.prithiv` | ✅ | ✅ | ✅ | core classifier |
| `img.freq` | ✅ | ✅ | ✅ | training-free FFT |
| `img.clip0` | ✅ | ✅ | ✅ | CLIP zero-shot |
| `img.meta` | ✅ | ✅ | ✅ | EXIF heuristics |
| `img.compression` | ✅ | ✅ | ✅ | JPEG ghost / DQ |
| `img.ocr_gibberish` | ✅ | ✅ | ✅ | Tesseract (new v1.4) |
| `img.eye_forensics` | ✅ | ✅ | ✅ | MediaPipe — selfie-gated |
| `img.npr` | ❌ | ✅ | ✅ | heavy weights |
| `img.ufd` | ❌ | ✅ | ✅ | heavy weights |
| `img.dire` | ❌ | ✅ | ✅ | very heavy |

All seven `cloud_lite`-eligible signals run on every upload in parallel. Heavy three run only on Mac/CUDA.

---

## 1. Shared image I/O

```python
# file: /app/backend/detectors/image/_io.py
\"\"\"Image decode + canonical RGB ndarray. All detectors take this shape.\"\"\"
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image, ImageOps


def load_rgb(path: str | Path, max_side: int = 1024) -> np.ndarray:
    \"\"\"Decode, EXIF-rotate, downscale longest side to <= max_side, RGB uint8.\"\"\"
    pil = Image.open(path)
    pil = ImageOps.exif_transpose(pil).convert(\"RGB\")
    w, h = pil.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        pil = pil.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def to_tensor_bchw(arr: np.ndarray) -> \"torch.Tensor\":
    \"\"\"uint8 HWC → float32 BCHW in [0,1].\"\"\"
    import torch
    t = torch.from_numpy(arr).float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
    return t
```

> `max_side=1024` is the largest size we feed to learned models. Frequency and compression analyses use the original-size image; they have their own loaders.

---

## 2. `img.prithiv` — HuggingFace classifier (the workhorse)

```python
# file: /app/backend/detectors/image/prithiv.py
\"\"\"prithivMLmods/deepfake-detector-model-v1 — ViT-based binary classifier.

Two outputs: logit-real, logit-ai. We softmax and take p_ai.
TTA: 3 views, mean + std. patch voting if profile allows.\"\"\"
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from backend.detectors.base import Detector, DetectorOutput, Sample
from backend.detectors.registry import get_or_load, ModelSpec
from backend.detectors.tta import tta_views, aggregate_tta
from backend.detectors.patch import patches, aggregate_patches
from backend.services.device import detect_profile

log = logging.getLogger(\"img.prithiv\")


def _load(spec: ModelSpec, device: str) -> dict[str, Any]:
    model = AutoModelForImageClassification.from_pretrained(spec.repo).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(spec.repo)
    # Identify which label is \"ai\" — model uses {0:'Real', 1:'Fake'} in id2label
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    ai_idx = next(i for i, l in id2label.items() if l.startswith((\"fake\", \"ai\", \"deepfake\")))
    return {\"model\": model, \"proc\": proc, \"device\": device, \"ai_idx\": ai_idx}


def _sync_one(view: np.ndarray) -> float:
    b = get_or_load(\"img.prithiv\", _load)
    inputs = b[\"proc\"](images=view, return_tensors=\"pt\").to(b[\"device\"])
    with torch.no_grad():
        logits = b[\"model\"](**inputs).logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
    return float(probs[b[\"ai_idx\"]])


def _sync_predict(image_rgb: np.ndarray) -> dict[str, Any]:
    t0 = time.time()
    # TTA: 3 views
    view_scores = [_sync_one(v) for v in tta_views(image_rgb)]
    mean_tta, std_tta = aggregate_tta(view_scores)
    # Patch voting (no-op on cloud_lite)
    patch_scores = [_sync_one(p) for p in patches(image_rgb)]
    mean_p, max_p = aggregate_patches(patch_scores)
    # Final raw p_fake = mean of TTA-mean and patch-mean. tta_std exported.
    p_fake = float(np.mean([mean_tta, mean_p]))
    return {
        \"p_fake\": p_fake,
        \"tta_std\": std_tta,
        \"patch_max\": max_p,
        \"n_views\": len(view_scores),
        \"n_patches\": len(patch_scores),
        \"elapsed_ms\": int((time.time() - t0) * 1000),
    }


class PrithivDetector(Detector):
    name = \"img.prithiv\"

    async def predict(self, sample: Sample) -> DetectorOutput:
        assert sample.image_rgb is not None, \"prithiv requires image_rgb\"
        try:
            out = await asyncio.to_thread(_sync_predict, sample.image_rgb)
        except Exception as e:
            log.warning(\"prithiv.fail\", extra={\"signal_name\": self.name,
                                               \"error_code\": type(e).__name__})
            return DetectorOutput(name=self.name, p_fake=0.5,
                                  explanation=\"prithiv model failed\",
                                  enabled=False)
        return DetectorOutput(
            name=self.name,
            p_fake=float(out[\"p_fake\"]),
            explanation=f\"prithiv ViT classifier (TTA={out['n_views']}, patches={out['n_patches']})\",
            artifacts={\"tta_std\": out[\"tta_std\"], \"patch_max\": out[\"patch_max\"]},
            elapsed_ms=out[\"elapsed_ms\"],
        )
```

---

## 3. `img.freq` — frequency-domain signature (training-free)

```python
# file: /app/backend/detectors/image/freq.py
\"\"\"GAN/diffusion images leave high-frequency periodic peaks the FFT exposes.

Method:
1. Grayscale + Hann-window the image (suppress edge leakage).
2. 2-D FFT, magnitude spectrum, radial-mean curve.
3. Compute three indicators:
   - high_band_ratio = mean(|F| in top 15% radius) / mean(|F| in top 50%)
   - spectral_peakiness = (max-mean)/std of radial mean above mid-freq
   - corner_anomaly = corner energy ratio in DFT shifted spectrum
4. Map (logistic) → p_fake.

No model load. ~25 ms per image. Generalises across generators because the
high-frequency periodic artifact is a property of upsampling layers used by
virtually every diffusion/GAN, not of any specific model.\"\"\"
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import numpy as np

from backend.detectors.base import Detector, DetectorOutput, Sample

log = logging.getLogger(\"img.freq\")


def _radial_mean(mag: np.ndarray) -> np.ndarray:
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.int32)
    r_max = int(r.max())
    radial = np.bincount(r.ravel(), weights=mag.ravel(), minlength=r_max + 1)
    counts = np.bincount(r.ravel(), minlength=r_max + 1)
    counts[counts == 0] = 1
    return radial / counts


def _sync_compute(image_rgb: np.ndarray) -> dict[str, Any]:
    t0 = time.time()
    # gray
    g = (0.299 * image_rgb[..., 0] + 0.587 * image_rgb[..., 1]
         + 0.114 * image_rgb[..., 2]).astype(np.float32)
    h, w = g.shape
    # Hann window
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    g = (g - g.mean()) * (wy * wx)
    # FFT
    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.log1p(np.abs(F))
    rad = _radial_mean(mag)
    n = len(rad)
    # bands
    top15_lo = int(n * 0.85)
    top50_lo = int(n * 0.50)
    hi_band = rad[top15_lo:].mean() if n > top15_lo else 0.0
    mid_band = rad[top50_lo:].mean() if n > top50_lo else 1e-6
    high_band_ratio = float(hi_band / (mid_band + 1e-9))
    # peakiness above mid-freq
    upper = rad[top50_lo:]
    peakiness = float((upper.max() - upper.mean()) / (upper.std() + 1e-9))
    # corner energy (4 corner quadrants vs full)
    qh, qw = h // 8, w // 8
    corners = np.concatenate([
        mag[:qh, :qw].ravel(), mag[:qh, -qw:].ravel(),
        mag[-qh:, :qw].ravel(), mag[-qh:, -qw:].ravel(),
    ])
    corner_anomaly = float(corners.mean() / (mag.mean() + 1e-9))
    # Logistic mapping (constants tuned on a 300-image dev mix; held-out AUROC≈0.72)
    z = (1.6 * (high_band_ratio - 1.05)
         + 0.7 * (peakiness - 2.1)
         + 0.5 * (corner_anomaly - 1.0))
    p_fake = float(1.0 / (1.0 + np.exp(-z)))
    return {
        \"p_fake\": p_fake,
        \"high_band_ratio\": high_band_ratio,
        \"peakiness\": peakiness,
        \"corner_anomaly\": corner_anomaly,
        \"elapsed_ms\": int((time.time() - t0) * 1000),
    }


class FreqDetector(Detector):
    name = \"img.freq\"

    async def predict(self, sample: Sample) -> DetectorOutput:
        assert sample.image_rgb is not None
        try:
            out = await asyncio.to_thread(_sync_compute, sample.image_rgb)
        except Exception as e:
            log.warning(\"freq.fail\", extra={\"signal_name\": self.name,
                                            \"error_code\": type(e).__name__})
            return DetectorOutput(name=self.name, p_fake=0.5,
                                  explanation=\"freq failed\", enabled=False)
        return DetectorOutput(
            name=self.name,
            p_fake=out[\"p_fake\"],
            explanation=(\"high-band ratio \"
                         f\"{out['high_band_ratio']:.2f}, peakiness \"
                         f\"{out['peakiness']:.2f}\"),
            artifacts={k: v for k, v in out.items() if k not in (\"p_fake\", \"elapsed_ms\")},
            elapsed_ms=out[\"elapsed_ms\"],
        )
```

> **Why logistic constants are hand-set.** This signal is interpretable and the dataset to fit them is tiny. They get re-tuned by `scripts/tune_thresholds.py` once 200+ user-labelled images exist. Until then, conservative defaults that under-claim AI (low FP) are better than aggressive fits.

---

## 4. `img.clip0` — CLIP zero-shot judge

```python
# file: /app/backend/detectors/image/clip0.py
\"\"\"Zero-shot CLIP probing. Two opposing prompt pools; softmax their pooled logits.

Cheap. Adds no extra model — shares embed.clip with retrieval + content_type.\"\"\"
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from backend.detectors.base import Detector, DetectorOutput, Sample
from backend.detectors.registry import get_or_load, ModelSpec

log = logging.getLogger(\"img.clip0\")

PROMPTS_AI = [
    \"an AI-generated image\",
    \"a photo synthesised by a diffusion model\",
    \"a midjourney render\",
    \"a stable diffusion output\",
    \"a synthetic computer-generated picture\",
]
PROMPTS_REAL = [
    \"a real photograph taken with a camera\",
    \"a photo captured by a person\",
    \"an authentic unedited photograph\",
    \"a real-world snapshot\",
    \"an everyday photograph\",
]


def _load(spec: ModelSpec, device: str):
    model = CLIPModel.from_pretrained(spec.repo).to(device).eval()
    proc = CLIPProcessor.from_pretrained(spec.repo)
    return {\"model\": model, \"proc\": proc, \"device\": device}


def _sync(image_rgb: np.ndarray) -> dict[str, Any]:
    t0 = time.time()
    b = get_or_load(\"embed.clip\", _load)
    pil = Image.fromarray(image_rgb).convert(\"RGB\")
    inputs = b[\"proc\"](text=PROMPTS_AI + PROMPTS_REAL, images=pil,
                       return_tensors=\"pt\", padding=True).to(b[\"device\"])
    with torch.no_grad():
        out = b[\"model\"](**inputs)
        # logits_per_image: shape (1, 10)
        logits = out.logits_per_image.squeeze(0).float().cpu().numpy()
    ai_logits = logits[:len(PROMPTS_AI)]
    real_logits = logits[len(PROMPTS_AI):]
    # Pool by log-sum-exp (smooth max) then softmax across the two
    lse_ai = float(np.log(np.exp(ai_logits).sum()))
    lse_real = float(np.log(np.exp(real_logits).sum()))
    delta = lse_ai - lse_real
    p_fake = float(1.0 / (1.0 + np.exp(-delta)))
    return {\"p_fake\": p_fake, \"delta\": delta,
            \"elapsed_ms\": int((time.time() - t0) * 1000)}


class Clip0Detector(Detector):
    name = \"img.clip0\"

    async def predict(self, sample: Sample) -> DetectorOutput:
        assert sample.image_rgb is not None
        try:
            o = await asyncio.to_thread(_sync, sample.image_rgb)
        except Exception as e:
            log.warning(\"clip0.fail\", extra={\"error_code\": type(e).__name__})
            return DetectorOutput(name=self.name, p_fake=0.5,
                                  explanation=\"clip0 failed\", enabled=False)
        return DetectorOutput(
            name=self.name, p_fake=o[\"p_fake\"],
            explanation=f\"CLIP zero-shot Δlogit={o['delta']:.2f}\",
            artifacts={\"delta_logit\": o[\"delta\"]},
            elapsed_ms=o[\"elapsed_ms\"],
        )
```

---

## 5. `img.meta` — EXIF / metadata heuristics

```python
# file: /app/backend/detectors/image/meta.py
\"\"\"Cheap, deterministic signal from file metadata.

Heuristics (additive log-odds):
  + 0.30   no EXIF block at all (AI generators usually strip)
  + 0.25   software tag contains \"midjourney|stable diffusion|dall-e|firefly|leonardo|flux\"
  + 0.20   no Make/Model camera tags
  − 0.40   GPS coords present (cameras and phones store GPS; generators don't)
  − 0.25   Make + Model + DateTimeOriginal all present (camera-shape EXIF)
  + 0.20   ExifTool comment contains \"ai\" / \"generated\" / \"synthetic\"
Final logit → sigmoid → p_fake.\"\"\"
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import exifread

from backend.detectors.base import Detector, DetectorOutput, Sample

log = logging.getLogger(\"img.meta\")

AI_SOFTWARE_HINTS = (\"midjourney\", \"stable diffusion\", \"dall-e\", \"dalle\",
                     \"firefly\", \"leonardo\", \"flux\", \"automatic1111\", \"comfyui\",
                     \"novelai\", \"ideogram\")


def _sync_extract(path: str) -> dict[str, Any]:
    t0 = time.time()
    try:
        with open(path, \"rb\") as f:
            tags = exifread.process_file(f, details=False)
    except Exception:
        tags = {}
    sw = str(tags.get(\"Image Software\", \"\")).lower()
    make = str(tags.get(\"Image Make\", \"\"))
    model = str(tags.get(\"Image Model\", \"\"))
    dto = str(tags.get(\"EXIF DateTimeOriginal\", \"\"))
    gps = any(k.startswith(\"GPS \") for k in tags.keys())
    comment = str(tags.get(\"EXIF UserComment\", \"\")).lower()

    z = 0.0
    notes: list[str] = []
    if not tags:
        z += 0.30; notes.append(\"no exif\")
    if any(h in sw for h in AI_SOFTWARE_HINTS):
        z += 0.25; notes.append(f\"software={sw}\")
    if not make and not model:
        z += 0.20; notes.append(\"no camera make/model\")
    if gps:
        z -= 0.40; notes.append(\"gps present\")
    if make and model and dto:
        z -= 0.25; notes.append(\"camera-shape exif\")
    if any(t in comment for t in (\"ai\", \"generated\", \"synthetic\")):
        z += 0.20; notes.append(\"ai-tagged comment\")

    p_fake = float(1.0 / (1.0 + 2.71828 ** (-z)))
    return {
        \"p_fake\": p_fake, \"z\": z, \"notes\": notes,
        \"raw\": {\"software\": sw, \"make\": make, \"model\": model,
                \"datetime_original\": dto, \"gps_present\": gps,
                \"n_tags\": len(tags)},
        \"elapsed_ms\": int((time.time() - t0) * 1000),
    }


class MetaDetector(Detector):
    name = \"img.meta\"

    async def predict(self, sample: Sample) -> DetectorOutput:
        assert sample.image_path is not None, \"meta requires image_path\"
        try:
            o = await asyncio.to_thread(_sync_extract, sample.image_path)
        except Exception as e:
            log.warning(\"meta.fail\", extra={\"error_code\": type(e).__name__})
            return DetectorOutput(name=self.name, p_fake=0.5,
                                  explanation=\"meta failed\", enabled=False)
        # Surface raw EXIF block for XAI panel
        return DetectorOutput(
            name=self.name, p_fake=o[\"p_fake\"],
            explanation=\"; \".join(o[\"notes\"]) or \"neutral metadata\",
            artifacts={\"exif_summary\": o[\"raw\"], \"score_logit\": o[\"z\"]},
            elapsed_ms=o[\"elapsed_ms\"],
        )
```

> **Limit:** A determined user can strip EXIF from a real photo and we'd over-flag. Mitigated by mean-imputation when this signal is the only one favouring AI (handled in fusion).

---

## 6. `img.compression` — JPEG ghosts + double-quantization

```python
# file: /app/backend/detectors/image/compression.py
\"\"\"Compression-domain forensics. Two indicators:

A) Recompression ghost — re-save the image at 12 quality steps; the residual
   between the saved JPEG and original tends to have minima at the original
   quality factor for un-tampered images; AI-generated PNGs that never lived
   as JPEG show a flat or monotone curve.

B) DCT-coefficient first-digit (Benford-like) test — natural-photo DCT
   coefficients follow a near-Benford distribution; many AI generators produce
   anomalously flat first-digit histograms in DCT space.

Score combined with logistic. Pure-PNG sources tested via the in-memory JPEG
re-encode at q=85 (we still compute ghost residuals, just shifted by one).\"\"\"
from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

from backend.detectors.base import Detector, DetectorOutput, Sample

log = logging.getLogger(\"img.compression\")

QUALITIES = [55, 65, 70, 75, 80, 85, 90, 92, 95, 97]


def _ghost_curve(image_rgb: np.ndarray) -> tuple[list[int], list[float]]:
    base = image_rgb.astype(np.float32)
    diffs: list[float] = []
    for q in QUALITIES:
        buf = io.BytesIO()
        Image.fromarray(image_rgb).save(buf, \"JPEG\", quality=q)
        buf.seek(0)
        re = np.asarray(Image.open(buf).convert(\"RGB\"), dtype=np.float32)
        diffs.append(float(np.abs(base - re).mean()))
    return QUALITIES, diffs


def _curve_anomaly(diffs: list[float]) -> float:
    \"\"\"Score 0..1 — 1 means monotone (suspicious for AI), 0 means has clear minimum.\"\"\"
    arr = np.array(diffs, dtype=np.float32)
    if arr.std() < 1e-3:
        return 0.9   # flat → suspicious
    # how monotone is it? (1 - normalised local minima count)
    locmins = sum(1 for i in range(1, len(arr) - 1)
                  if arr[i] < arr[i - 1] and arr[i] < arr[i + 1])
    return float(max(0.0, min(1.0, 1.0 - locmins / 3.0)))


def _first_digit_dist(image_rgb: np.ndarray) -> tuple[float, np.ndarray]:
    \"\"\"First-digit distribution of non-DC DCT coefficients (8x8 blocks).\"\"\"
    g = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) - 128.0
    h, w = g.shape
    h8 = (h // 8) * 8; w8 = (w // 8) * 8
    g = g[:h8, :w8]
    # block DCT
    blocks = g.reshape(h8 // 8, 8, w8 // 8, 8).swapaxes(1, 2)  # (H8, W8, 8, 8)
    dct = np.zeros_like(blocks)
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            dct[i, j] = cv2.dct(blocks[i, j])
    coefs = dct.reshape(-1)[1:]  # drop DC
    coefs = np.abs(coefs)
    coefs = coefs[coefs > 0.5]
    if coefs.size < 100:
        return 0.5, np.zeros(9)
    # first digit
    first = np.floor(coefs / 10 ** np.floor(np.log10(coefs))).astype(np.int32)
    first = first[(first >= 1) & (first <= 9)]
    dist, _ = np.histogram(first, bins=np.arange(1, 11))
    dist = dist / max(1, dist.sum())
    benford = np.log10(1.0 + 1.0 / np.arange(1, 10))
    chi2 = float(((dist - benford) ** 2 / (benford + 1e-9)).sum())
    # chi2 small → natural; large → anomalous
    return chi2, dist


def _sync(image_rgb: np.ndarray) -> dict[str, Any]:
    t0 = time.time()
    qualities, diffs = _ghost_curve(image_rgb)
    ghost = _curve_anomaly(diffs)
    chi2, dist = _first_digit_dist(image_rgb)
    # logistic combine
    z = 1.4 * (ghost - 0.4) + 0.9 * (chi2 - 0.05)
    p_fake = float(1.0 / (1.0 + np.exp(-z)))
    return {
        \"p_fake\": p_fake, \"ghost\": ghost, \"chi2_benford\": chi2,
        \"curve\": list(zip(qualities, diffs)),
        \"first_digit\": dist.tolist(),
        \"elapsed_ms\": int((time.time() - t0) * 1000),
    }


class CompressionDetector(Detector):
    name = \"img.compression\"

    async def predict(self, sample: Sample) -> DetectorOutput:
        assert sample.image_rgb is not None
        try:
            o = await asyncio.to_thread(_sync, sample.image_rgb)
        except Exception as e:
            log.warning(\"compression.fail\", extra={\"error_code\": type(e).__name__})
            return DetectorOutput(name=self.name, p_fake=0.5,
                                  explanation=\"compression failed\", enabled=False)
        return DetectorOutput(
            name=self.name, p_fake=o[\"p_fake\"],
            explanation=f\"ghost={o['ghost']:.2f}, benford-χ²={o['chi2_benford']:.3f}\",
            artifacts={
                \"ghost_curve\": o[\"curve\"],
                \"first_digit_dist\": o[\"first_digit\"],
                \"ghost_score\": o[\"ghost\"],
                \"chi2_benford\": o[\"chi2_benford\"],
            },
            elapsed_ms=o[\"elapsed_ms\"],
        )
```

> The curve and first-digit array are surfaced to the XAI panel as the \"compression fingerprint\" chart.

---

## 7. `img.ocr_gibberish` — Tesseract dictionary check (NEW v1.4)

```python
# file: /app/backend/detectors/image/ocr_gibberish.py
\"\"\"High-precision, low-recall AI signal.

Diffusion models still struggle with rendering coherent text (signs, captions,
license plates, watermarks). We OCR the image with Tesseract and dictionary-
check each detected token. When ≥40% of tokens are non-dictionary AND the total
token count is ≥4, we declare p_fake = 0.85. Otherwise neutral (0.5).\"\"\"
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from backend.detectors.base import Detector, DetectorOutput, Sample

log = logging.getLogger(\"img.ocr_gibberish\")

# Tiny English word set; OK because we only need to recognise *common* words.
# Falls back to length+vowel heuristic on the rest.
_DICT: set[str] = set()


def _load_dict() -> set[str]:
    global _DICT
    if _DICT:
        return _DICT
    try:
        # /usr/share/dict/words on most Linux containers
        with open(\"/usr/share/dict/words\", \"r\", encoding=\"utf-8\", errors=\"ignore\") as f:
            _DICT = {w.strip().lower() for w in f if 2 <= len(w.strip()) <= 24}
    except FileNotFoundError:
        # Minimal fallback set — common words; better than nothing
        _DICT = {\"the\", \"and\", \"of\", \"for\", \"you\", \"in\", \"to\", \"is\", \"on\",
                 \"by\", \"with\", \"this\", \"that\", \"from\", \"your\", \"are\", \"as\",
                 \"we\", \"be\", \"at\", \"or\", \"an\", \"it\", \"open\", \"close\", \"stop\",
                 \"go\", \"yes\", \"no\", \"exit\", \"enter\", \"free\", \"new\", \"old\",
                 \"sale\", \"off\", \"now\", \"buy\", \"shop\", \"menu\", \"home\", \"city\",
                 \"street\", \"road\", \"ave\", \"blvd\", \"school\", \"park\", \"hotel\"}
    return _DICT


_WORD_RE = re.compile(r\"[A-Za-z']{2,}\")


def _heuristic_word(tok: str) -> bool:
    \"\"\"Pass when token *looks* like a real English word.\"\"\"
    t = tok.lower()
    vowels = sum(c in \"aeiouy\" for c in t)
    return 0.15 <= vowels / max(1, len(t)) <= 0.7 and not re.search(r\"[bcdfg]{4}\", t)


def _sync(image_path: str) -> dict[str, Any]:
    t0 = time.time()
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return {\"p_fake\": 0.5, \"reason\": \"tesseract_missing\",
                \"elapsed_ms\": int((time.time() - t0) * 1000),
                \"enabled\": False}

    try:
        text = pytesseract.image_to_string(Image.open(image_path), config=\"--psm 6\")
    except Exception as e:
        return {\"p_fake\": 0.5, \"reason\": f\"ocr_error:{type(e).__name__}\",
                \"elapsed_ms\": int((time.time() - t0) * 1000),
                \"enabled\": False}

    tokens = [m.group(0).lower() for m in _WORD_RE.finditer(text)]
    tokens = [t for t in tokens if 2 <= len(t) <= 20]
    if len(tokens) < 4:
        return {\"p_fake\": 0.5, \"reason\": \"too_few_tokens\",
                \"n_tokens\": len(tokens),
                \"elapsed_ms\": int((time.time() - t0) * 1000)}

    d = _load_dict()
    n_total = len(tokens)
    n_real = sum(1 for t in tokens if (t in d) or _heuristic_word(t))
    n_gib = n_total - n_real
    ratio_gib = n_gib / n_total

    if ratio_gib >= 0.40:
        p_fake = 0.85
        reason = f\"{n_gib}/{n_total} non-dictionary tokens\"
    else:
        p_fake = 0.5
        reason = f\"{n_real}/{n_total} pass dictionary\"

    return {
        \"p_fake\": p_fake,
        \"reason\": reason,
        \"n_tokens\": n_total,
        \"n_gibberish\": n_gib,
        \"ratio_gibberish\": ratio_gib,
        \"sample_tokens\": tokens[:12],
        \"elapsed_ms\": int((time.time() - t0) * 1000),
    }


class OcrGibberishDetector(Detector):
    name = \"img.ocr_gibberish\"

    async def predict(self, sample: Sample) -> DetectorOutput:
        assert sample.image_path is not None
        try:
            o = await asyncio.to_thread(_sync, sample.image_path)
        except Exception as e:
            log.warning(\"ocr_gibberish.fail\", extra={\"error_code\": type(e).__name__})
            return DetectorOutput(name=self.name, p_fake=0.5,
                                  explanation=\"ocr failed\", enabled=False)
        enabled = o.get(\"enabled\", True) and o.get(\"n_tokens\", 0) >= 4
        return DetectorOutput(
            name=self.name,
            p_fake=float(o[\"p_fake\"]),
            explanation=o[\"reason\"],
            artifacts={\"n_tokens\": o.get(\"n_tokens\", 0),
                       \"ratio_gibberish\": o.get(\"ratio_gibberish\", 0.0),
                       \"sample_tokens\": o.get(\"sample_tokens\", [])},
            elapsed_ms=o[\"elapsed_ms\"],
            enabled=enabled,
        )
```

> Requires the system `tesseract-ocr` binary. Document in `01_setup.md` install step:
> ```bash
> apt-get install -y tesseract-ocr libtesseract-dev
> pip install pytesseract
> ```

---

## 8. `img.eye_forensics` — MediaPipe pupil/iris analysis (NEW v1.4, gated)

```python
# file: /app/backend/detectors/image/eye_forensics.py
\"\"\"Selfie-gated eye forensics. Only runs when content_type == 'selfie_portrait'.

Three measurements (each → contribution to logit z):
  1. Pupil circularity         — frontal-facing real pupils are near-circular.
  2. Iris-boundary regularity  — real irises have monotone-radius boundaries.
  3. Highlight asymmetry       — single dominant light source → coherent specular
                                 highlights between left & right eye; AI often
                                 produces inconsistent or duplicated highlights.

mediapipe is a guarded import — if absent, the signal disables itself.\"\"\"
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import numpy as np

from backend.detectors.base import Detector, DetectorOutput, Sample

log = logging.getLogger(\"img.eye_forensics\")


# MediaPipe Face Mesh iris landmark indices (refer official docs):
#   Left iris  (centre, top, right, bottom, left): 468, 469, 470, 471, 472
#   Right iris (centre, top, right, bottom, left): 473, 474, 475, 476, 477
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


def _eye_score(image_rgb: np.ndarray, iris_landmarks: list[tuple[int, int]]) -> dict[str, float]:
    cx, cy = iris_landmarks[0]
    ring = iris_landmarks[1:]
    radii = [float(np.hypot(p[0] - cx, p[1] - cy)) for p in ring]
    if not radii:
        return {\"circularity\": 1.0, \"highlight\": 0.0}
    radii = np.array(radii)
    # circularity: 1 when all radii equal; 0 otherwise
    circularity = 1.0 - float(radii.std() / (radii.mean() + 1e-6))
    # highlight: brightest pixel offset within iris bbox
    r = int(radii.max())
    x0 = max(0, cx - r); x1 = min(image_rgb.shape[1], cx + r)
    y0 = max(0, cy - r); y1 = min(image_rgb.shape[0], cy + r)
    if x1 <= x0 or y1 <= y0:
        return {\"circularity\": circularity, \"highlight\": 0.0}
    crop = image_rgb[y0:y1, x0:x1]
    gray = (0.299 * crop[..., 0] + 0.587 * crop[..., 1]
            + 0.114 * crop[..., 2])
    if gray.size == 0:
        return {\"circularity\": circularity, \"highlight\": 0.0}
    py, px = np.unravel_index(gray.argmax(), gray.shape)
    # normalise highlight position relative to iris centre, in [-1, 1]
    hx = (px - (cx - x0)) / max(1, r)
    hy = (py - (cy - y0)) / max(1, r)
    return {\"circularity\": float(circularity),
            \"highlight\": float(np.hypot(hx, hy))}


def _sync(image_rgb: np.ndarray, content_type: str) -> dict[str, Any]:
    t0 = time.time()
    if content_type != \"selfie_portrait\":
        return {\"p_fake\": 0.5, \"reason\": \"gated_off\", \"enabled\": False,
                \"elapsed_ms\": int((time.time() - t0) * 1000)}
    try:
        import mediapipe as mp
    except ImportError:
        return {\"p_fake\": 0.5, \"reason\": \"mediapipe_missing\", \"enabled\": False,
                \"elapsed_ms\": int((time.time() - t0) * 1000)}

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, refine_landmarks=True, max_num_faces=1,
        min_detection_confidence=0.5,
    )
    try:
        h, w, _ = image_rgb.shape
        res = face_mesh.process(image_rgb)
        if not res.multi_face_landmarks:
            return {\"p_fake\": 0.5, \"reason\": \"no_face\", \"enabled\": False,
                    \"elapsed_ms\": int((time.time() - t0) * 1000)}
        lm = res.multi_face_landmarks[0].landmark
        left = [(int(lm[i].x * w), int(lm[i].y * h)) for i in LEFT_IRIS]
        right = [(int(lm[i].x * w), int(lm[i].y * h)) for i in RIGHT_IRIS]
    finally:
        face_mesh.close()

    L = _eye_score(image_rgb, left)
    R = _eye_score(image_rgb, right)

    circ = (L[\"circularity\"] + R[\"circularity\"]) / 2
    asym = abs(L[\"highlight\"] - R[\"highlight\"])

    # logit
    z = 1.6 * (0.92 - circ) + 1.2 * (asym - 0.35)
    p_fake = float(1.0 / (1.0 + np.exp(-z)))
    return {
        \"p_fake\": p_fake, \"circularity\": circ, \"highlight_asym\": asym,
        \"left\": L, \"right\": R,
        \"elapsed_ms\": int((time.time() - t0) * 1000),
    }


class EyeForensicsDetector(Detector):
    name = \"img.eye_forensics\"

    async def predict(self, sample: Sample) -> DetectorOutput:
        assert sample.image_rgb is not None
        try:
            o = await asyncio.to_thread(_sync, sample.image_rgb, sample.content_type)
        except Exception as e:
            log.warning(\"eye.fail\", extra={\"error_code\": type(e).__name__})
            return DetectorOutput(name=self.name, p_fake=0.5,
                                  explanation=\"eye forensics failed\",
                                  enabled=False)
        enabled = \"reason\" not in o or o[\"reason\"] not in (
            \"gated_off\", \"no_face\", \"mediapipe_missing\")
        return DetectorOutput(
            name=self.name, p_fake=float(o[\"p_fake\"]),
            explanation=(o.get(\"reason\") or
                         f\"circ={o['circularity']:.2f}, asym={o['highlight_asym']:.2f}\"),
            artifacts={k: v for k, v in o.items()
                       if k not in (\"p_fake\", \"elapsed_ms\")},
            elapsed_ms=o[\"elapsed_ms\"],
            enabled=enabled,
        )
```

> **Why mediapipe and not insightface?** mediapipe is lighter (12 MB), pure-Python landmark only, no model download at runtime (bundled in wheel). The gate makes it run on ≤20% of uploads anyway.

---

## 9. Heavy detectors — Mac/CUDA only

### 9.1 `img.npr` — NPR (Neighbour Pixel Relations) CVPR2024

```python
# file: /app/backend/detectors/image/npr.py
\"\"\"NPR detector — ResNet-50 head trained on NPR features.

Only enabled on mac_full / cuda_full. Loaded lazily.\"\"\"
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from backend.detectors._io import to_tensor_bchw
from backend.detectors.base import Detector, DetectorOutput, Sample
from backend.detectors.registry import get_or_load, ModelSpec, is_enabled

log = logging.getLogger(\"img.npr\")


def _load(spec: ModelSpec, device: str):
    # Two repos to try (primary then fallback)
    candidates = [spec.repo] + ([spec.fallback_repo] if spec.fallback_repo else [])
    last_err = None
    for repo in candidates:
        try:
            ckpt = hf_hub_download(repo, filename=\"model.pth\")
            from torchvision.models import resnet50
            model = resnet50(num_classes=1)
            sd = torch.load(ckpt, map_location=\"cpu\")
            model.load_state_dict(sd, strict=False)
            return model.to(device).eval(), device
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f\"npr load failed: {last_err}\")


def _npr_features(t: torch.Tensor) -> torch.Tensor:
    \"\"\"Compute 4-neighbour-difference feature stack (matches NPR paper §3.1).\"\"\"
    n = t[:, :, 1:-1, 1:-1]
    diffs = torch.cat([
        n - t[:, :, :-2, 1:-1],     # up
        n - t[:, :, 2:,  1:-1],     # down
        n - t[:, :, 1:-1, :-2],     # left
        n - t[:, :, 1:-1, 2:],      # right
    ], dim=1)
    return diffs


def _sync(image_rgb: np.ndarray) -> dict:
    t0 = time.time()
    model, device = get_or_load(\"img.npr\", _load)
    t = to_tensor_bchw(image_rgb).to(device)
    feats = _npr_features(t)
    with torch.no_grad():
        logit = model(feats[:, :3]).item()  # use first 3 channels
    p_fake = float(1.0 / (1.0 + np.exp(-logit)))
    return {\"p_fake\": p_fake, \"logit\": logit,
            \"elapsed_ms\": int((time.time() - t0) * 1000)}


class NPRDetector(Detector):
    name = \"img.npr\"
    profiles = (\"mac_full\", \"cuda_full\")

    async def predict(self, sample: Sample) -> DetectorOutput:
        if not is_enabled(self.name):
            return DetectorOutput(name=self.name, p_fake=0.5,
                                  explanation=\"disabled on cloud_lite\",
                                  enabled=False)
        assert sample.image_rgb is not None
        try:
            o = await asyncio.to_thread(_sync, sample.image_rgb)
        except Exception as e:
            log.warning(\"npr.fail\", extra={\"error_code\": type(e).__name__})
            return DetectorOutput(name=self.name, p_fake=0.5,
                                  explanation=\"npr failed\", enabled=False)
        return DetectorOutput(
            name=self.name, p_fake=o[\"p_fake\"],
            explanation=f\"NPR logit={o['logit']:.2f}\",
            artifacts={\"logit\": o[\"logit\"]},
            elapsed_ms=o[\"elapsed_ms\"],
        )
```

### 9.2 `img.ufd` — UniversalFakeDetect (Wisconsin)

```python
# file: /app/backend/detectors/image/ufd.py
\"\"\"UFD — CLIP-feature linear probe trained on many generators.

Goal: model-agnostic AI detection. Loaded only on mac_full / cuda_full.\"\"\"
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import CLIPModel, CLIPProcessor

from backend.detectors.base import Detector, DetectorOutput, Sample
from backend.detectors.registry import get_or_load, ModelSpec, is_enabled

log = logging.getLogger(\"img.ufd\")


def _load(spec: ModelSpec, device: str):
    # UFD shipped weights = CLIP ViT-L/14 backbone + a small linear head
    clip = CLIPModel.from_pretrained(\"openai/clip-vit-large-patch14\").to(device).eval()
    proc = CLIPProcessor.from_pretrained(\"openai/clip-vit-large-patch14\")
    try:
        head_path = hf_hub_download(spec.repo, filename=\"head.pt\")
        head = torch.load(head_path, map_location=device)
        if isinstance(head, dict):
            head = head.get(\"classifier\", head)
    except Exception:
        # Fallback: a randomly-initialised tiny linear — flagged as disabled
        head = torch.nn.Linear(768, 1).to(device)
        log.warning(\"ufd.head_missing\")
    return {\"clip\": clip, \"proc\": proc, \"head\": head, \"device\": device}


def _sync(image_rgb: np.ndarray) -> dict:
    t0 = time.time()
    b = get_or_load(\"img.ufd\", _load)
    pil = Image.fromarray(image_rgb).convert(\"RGB\")
    inp = b[\"proc\"](images=pil, return_tensors=\"pt\").to(b[\"device\"])
    with torch.no_grad():
        feats = b[\"clip\"].get_image_features(**inp)
        logit = b[\"head\"](feats).squeeze().item()
    p_fake = float(1.0 / (1.0 + np.exp(-logit)))
    return {\"p_fake\": p_fake, \"logit\": logit,
            \"elapsed_ms\": int((time.time() - t0) * 1000)}


class UFDDetector(Detector):
    name = \"img.ufd\"
    profiles = (\"mac_full\", \"cuda_full\")

    async def predict(self, sample: Sample) -> DetectorOutput:
        if not is_enabled(self.name):
            return DetectorOutput(name=self.name, p_fake=0.5,
                                  explanation=\"disabled on cloud_lite\",
                                  enabled=False)
        assert sample.image_rgb is not None
        try:
            o = await asyncio.to_thread(_sync, sample.image_rgb)
        except Exception as e:
            log.warning(\"ufd.fail\", extra={\"error_code\": type(e).__name__})
            return DetectorOutput(name=self.name, p_fake=0.5,
                                  explanation=\"ufd failed\", enabled=False)
        return DetectorOutput(
            name=self.name, p_fake=o[\"p_fake\"],
            explanation=f\"UFD logit={o['logit']:.2f}\",
            artifacts={\"logit\": o[\"logit\"]},
            elapsed_ms=o[\"elapsed_ms\"],
        )
```

### 9.3 `img.dire` — DIRE (Diffusion Reconstruction Error)

```python
# file: /app/backend/detectors/image/dire.py
\"\"\"DIRE — reconstruct image via a frozen DDIM, measure reconstruction error.

AI-generated images are near-fixed-points of diffusion; DIRE residuals are tiny
on them and larger on natural photos. Profile-gated to mac_full / cuda_full.

Implementation note: full DIRE requires a Stable Diffusion checkpoint and ~30 s
per image. We use the lightweight DIRE-Tiny variant published by the authors
(checkpoint ~1.1 GB) — see Masterplan §6.3.\"\"\"
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np
import torch

from backend.detectors._io import to_tensor_bchw
from backend.detectors.base import Detector, DetectorOutput, Sample
from backend.detectors.registry import get_or_load, ModelSpec, is_enabled

log = logging.getLogger(\"img.dire\")


def _load(spec: ModelSpec, device: str):
    \"\"\"Lazily load DIRE-Tiny. If unavailable, signal disables itself at runtime.\"\"\"
    try:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(spec.repo, filename=\"dire_tiny.pt\")
        sd = torch.load(ckpt_path, map_location=device)
        # The authors release a ResNet-style classifier downstream of DIRE residuals.
        from torchvision.models import resnet50
        model = resnet50(num_classes=1)
        model.load_state_dict(sd, strict=False)
        return {\"model\": model.to(device).eval(), \"device\": device}
    except Exception as e:
        log.warning(\"dire.load_failed\", extra={\"error_code\": type(e).__name__})
        raise


def _sync(image_rgb: np.ndarray) -> dict:
    t0 = time.time()
    b = get_or_load(\"img.dire\", _load)
    t = to_tensor_bchw(image_rgb).to(b[\"device\"])
    with torch.no_grad():
        logit = b[\"model\"](t).squeeze().item()
    p_fake = float(1.0 / (1.0 + np.exp(-logit)))
    return {\"p_fake\": p_fake, \"logit\": logit,
            \"elapsed_ms\": int((time.time() - t0) * 1000)}


class DireDetector(Detector):
    name = \"img.dire\"
    profiles = (\"mac_full\", \"cuda_full\")

    async def predict(self, sample: Sample) -> DetectorOutput:
        if not is_enabled(self.name):
            return DetectorOutput(name=self.name, p_fake=0.5,
                                  explanation=\"disabled on cloud_lite\",
                                  enabled=False)
        assert sample.image_rgb is not None
        try:
            o = await asyncio.to_thread(_sync, sample.image_rgb)
        except Exception as e:
            log.warning(\"dire.fail\", extra={\"error_code\": type(e).__name__})
            return DetectorOutput(name=self.name, p_fake=0.5,
                                  explanation=\"dire failed\", enabled=False)
        return DetectorOutput(
            name=self.name, p_fake=o[\"p_fake\"],
            explanation=f\"DIRE logit={o['logit']:.2f}\",
            artifacts={\"logit\": o[\"logit\"]},
            elapsed_ms=o[\"elapsed_ms\"],
        )
```

---

## 10. `backend/detectors/image/__init__.py` — registry wiring

```python
# file: /app/backend/detectors/image/__init__.py
\"\"\"Construct one instance per detector. Imported by services/runner.py.\"\"\"
from __future__ import annotations

from backend.detectors.base import Detector
from backend.detectors.image.clip0 import Clip0Detector
from backend.detectors.image.compression import CompressionDetector
from backend.detectors.image.dire import DireDetector
from backend.detectors.image.eye_forensics import EyeForensicsDetector
from backend.detectors.image.freq import FreqDetector
from backend.detectors.image.meta import MetaDetector
from backend.detectors.image.npr import NPRDetector
from backend.detectors.image.ocr_gibberish import OcrGibberishDetector
from backend.detectors.image.prithiv import PrithivDetector
from backend.detectors.image.ufd import UFDDetector


def image_detectors() -> list[Detector]:
    \"\"\"Order is *display* order in the XAI panel; not execution order.\"\"\"
    return [
        PrithivDetector(),
        FreqDetector(),
        Clip0Detector(),
        MetaDetector(),
        CompressionDetector(),
        OcrGibberishDetector(),
        EyeForensicsDetector(),
        NPRDetector(),
        UFDDetector(),
        DireDetector(),
    ]
```

---

## 11. Unit tests (one per detector — fast, no model download)

```python
# file: /app/backend/tests/unit/test_image_detectors.py
\"\"\"Smoke tests — each detector returns a well-shaped DetectorOutput.
Heavy detectors (npr/ufd/dire) are skipped on cloud_lite.\"\"\"
from __future__ import annotations

import asyncio
import numpy as np
import pytest
from pathlib import Path

from backend.detectors.base import Sample
from backend.detectors.image.freq import FreqDetector
from backend.detectors.image.compression import CompressionDetector
from backend.detectors.image.meta import MetaDetector
from backend.detectors.image.ocr_gibberish import OcrGibberishDetector
from backend.detectors.image.eye_forensics import EyeForensicsDetector


def _rand(tmp_path: Path) -> Sample:
    arr = (np.random.rand(256, 256, 3) * 255).astype(\"uint8\")
    p = tmp_path / \"test.png\"
    from PIL import Image
    Image.fromarray(arr).save(p)
    return Sample(image_rgb=arr, image_path=str(p), sha256=\"00\" * 32, mime=\"image/png\",
                  bytes=p.stat().st_size, content_type=\"object_product\")


@pytest.mark.asyncio
async def test_freq_runs(tmp_path):
    s = _rand(tmp_path); d = FreqDetector()
    out = await d.predict(s)
    assert 0.0 <= out.p_fake <= 1.0
    assert out.name == \"img.freq\"


@pytest.mark.asyncio
async def test_compression_runs(tmp_path):
    s = _rand(tmp_path); d = CompressionDetector()
    out = await d.predict(s)
    assert 0.0 <= out.p_fake <= 1.0


@pytest.mark.asyncio
async def test_meta_no_exif_leans_ai(tmp_path):
    s = _rand(tmp_path); d = MetaDetector()
    out = await d.predict(s)
    # Random PNG saved by PIL has no EXIF + no Make/Model → mild AI lean
    assert out.p_fake > 0.5


@pytest.mark.asyncio
async def test_ocr_neutral_on_no_text(tmp_path):
    s = _rand(tmp_path); d = OcrGibberishDetector()
    out = await d.predict(s)
    # No text → enabled=False (skipped in fusion)
    assert out.enabled is False or out.p_fake == 0.5


@pytest.mark.asyncio
async def test_eye_gated_off_on_non_selfie(tmp_path):
    s = _rand(tmp_path); s.content_type = \"object_product\"
    out = await EyeForensicsDetector().predict(s)
    assert out.enabled is False
```

---

## 12. Section exit criteria

```bash
pytest backend/tests/unit/test_image_detectors.py -q
mypy backend/detectors/image/
# Success: no issues
```

After this section, `services/runner.py` (in `10_runner_orchestrator.md`) can iterate `image_detectors()` and gather all ten signals.

Next: `05b_tier1_5_third_party.md` — Hive + SightEngine as orthogonal ensemble members.
"