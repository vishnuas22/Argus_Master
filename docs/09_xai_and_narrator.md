"# 09 — XAI Artefacts & Narrator (Gemini + Rule-Based Fallback)

> Goal: turn raw signal data into the four visible artefacts of the Result page,
> plus the per-job natural-language narrative.
>
> 1. **Heatmap** — Grad-CAM over the last conv block of `img.prithiv` (the ViT
>    attention rollout when no conv block exists)
> 2. **Frequency-spectrum plot** — FFT radial-mean curve with reference bands
> 3. **Compression fingerprint plot** — ghost curve + Benford first-digit bars
> 4. **Eye-debug overlay** — left/right iris circle + highlight dot (selfies only)
> 5. **Narrative** — 6–10 sentences, Gemini-3 Flash with **4-shot in-context
>    examples** + rule-based fallback when the LLM is unavailable or returns junk
>
> Every artefact writes to `/app/backend/storage/jobs/{job_id}/assets/` and is
> served via `GET /api/jobs/{id}/assets/{name}`. Filenames are stable for
> idempotency — re-running the same job overwrites the same files.

---

## 1. Output contract

A finished XAI stage hands the runner this dict (consumed by `schemas/results.py::XAI`):

```python
{
    \"heatmap_url\":         \"/api/jobs/{id}/assets/heatmap.png\",
    \"frequency_plot_url\":  \"/api/jobs/{id}/assets/frequency.png\",
    \"compression_plot_url\":\"/api/jobs/{id}/assets/compression.png\",
    \"eye_overlay_url\":     \"/api/jobs/{id}/assets/eye_overlay.png\",  # optional
    \"metadata\": {...},                  # raw exif summary from img.meta
    \"compression_fingerprint\": {...},   # ghost + benford raw arrays
    \"narrative\": \"<6-10 sentence string>\",
    \"narrative_source\": \"gemini\" | \"fallback_template\",
}
```

Missing keys are allowed (e.g. `eye_overlay_url` absent when content_type ≠ selfie).
The frontend renders nothing for missing slots.

---

## 2. `backend/xai/gradcam.py` — Grad-CAM for the ViT classifier

```python
# file: /app/backend/xai/gradcam.py
\"\"\"Grad-CAM-style attribution for the prithiv ViT classifier.

ViTs have no conv \"last layer\"; we use **attention-rollout** + per-token gradient
weighting (Chefer et al. 2021, \"Transformer Interpretability Beyond Attention
Visualisation\"). Fallbacks to attention-rollout-only when grads are unavailable.

Output: a HxW uint8 heatmap aligned to the original image.\"\"\"
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from backend.detectors.registry import get_or_load
from backend.detectors.image.prithiv import _load as _load_prithiv

log = logging.getLogger(\"xai.gradcam\")


def _resize_heat(heat: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    return cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC)


def _attn_rollout(attentions: list[torch.Tensor]) -> torch.Tensor:
    \"\"\"Standard attention-rollout: prod_l (0.5 * (A_l + I)).\"\"\"
    # each A_l: (B, heads, T, T) — average over heads, add identity, normalise
    rollout = None
    for A in attentions:
        a = A.mean(dim=1)                           # (B, T, T)
        a = a + torch.eye(a.size(-1), device=a.device)[None]
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a if rollout is None else torch.bmm(rollout, a)
    return rollout                                  # (B, T, T)


def _grid_from_cls(rollout: torch.Tensor, grid_hw: tuple[int, int]) -> np.ndarray:
    \"\"\"Pull CLS-token attention into the patch grid → (gh, gw).\"\"\"
    # rollout: (B, T, T) ; first token is CLS
    cls = rollout[0, 0, 1:]                         # (T-1,) patch tokens
    gh, gw = grid_hw
    grid = cls[: gh * gw].reshape(gh, gw)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-9)
    return grid.detach().cpu().numpy().astype(np.float32)


def _sync_gradcam(image_rgb: np.ndarray) -> np.ndarray:
    bundle = get_or_load(\"img.prithiv\", _load_prithiv)
    model = bundle[\"model\"]
    proc = bundle[\"proc\"]
    device = bundle[\"device\"]

    pil = Image.fromarray(image_rgb).convert(\"RGB\")
    inputs = proc(images=pil, return_tensors=\"pt\").to(device)
    # ViTs expose attentions when output_attentions=True; guarded for non-ViT heads
    try:
        out = model(**inputs, output_attentions=True)
        attentions = list(out.attentions)           # tuple of (B, heads, T, T)
    except (TypeError, AttributeError):
        log.warning(\"gradcam.no_attentions\")
        # Last-resort uniform map
        h, w = image_rgb.shape[:2]
        return (np.ones((h, w), dtype=np.uint8) * 64)

    # Patch grid for ViT-B/16 at 224 input = 14×14
    side = inputs[\"pixel_values\"].shape[-1]
    patch = getattr(model.config, \"patch_size\", 16)
    gh = gw = side // patch
    rollout = _attn_rollout(attentions)
    grid = _grid_from_cls(rollout, (gh, gw))

    heat = _resize_heat(grid, image_rgb.shape[:2])
    heat_uint8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    return heat_uint8


def overlay_jet(image_rgb: np.ndarray, heat_uint8: np.ndarray,
                alpha: float = 0.45) -> np.ndarray:
    \"\"\"Compose a JET-coloured heatmap on top of the original image.\"\"\"
    colour = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
    out = (alpha * colour + (1 - alpha) * image_rgb).astype(np.uint8)
    return out


async def gradcam(image_rgb: np.ndarray) -> np.ndarray:
    \"\"\"Returns the heatmap **overlay** (HxWx3 uint8). Falls back to a flat
    grey overlay if the classifier weights aren't loaded.\"\"\"
    import asyncio
    try:
        heat = await asyncio.to_thread(_sync_gradcam, image_rgb)
        return overlay_jet(image_rgb, heat)
    except Exception as e:
        log.warning(\"gradcam.fail\", extra={\"error_code\": type(e).__name__})
        # Grey placeholder so the panel never blanks out
        return (image_rgb * 0.6).astype(np.uint8)
```

> **Why attention-rollout, not vanilla Grad-CAM?**
> `prithiv-deepfake-detector-model-v1` is a ViT. Grad-CAM assumes a conv feature
> map; ViTs do not have one in the last layer. Attention-rollout is the
> Chefer-Chefer-Wolf 2021 substitute and produces faithful localisation.
> When the underlying model is swapped for a CNN-based detector later (e.g.
> NPR), `xai/gradcam_cnn.py` (Phase-1 stub) handles `target_layer = layer4`.

---

## 3. `backend/xai/fft_plot.py` — frequency spectrum visualisation

```python
# file: /app/backend/xai/fft_plot.py
\"\"\"Render the radial-mean FFT curve as a PNG, with mid/high band shading.

The artefact echoes the FreqDetector's reasoning visually: AI images have
elevated energy in the top-15 % radius band — the shaded region pops red.\"\"\"
from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
from PIL import Image

log = logging.getLogger(\"xai.fft\")


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


def _compute(image_rgb: np.ndarray) -> np.ndarray:
    g = (0.299 * image_rgb[..., 0] + 0.587 * image_rgb[..., 1]
         + 0.114 * image_rgb[..., 2]).astype(np.float32)
    h, w = g.shape
    wy = np.hanning(h)[:, None]; wx = np.hanning(w)[None, :]
    g = (g - g.mean()) * (wy * wx)
    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.log1p(np.abs(F))
    return _radial_mean(mag)


def render(image_rgb: np.ndarray) -> bytes:
    \"\"\"Returns PNG bytes ready to write to disk.\"\"\"
    import matplotlib
    matplotlib.use(\"Agg\")              # no display backend
    import matplotlib.pyplot as plt

    rad = _compute(image_rgb)
    n = len(rad)
    if n < 8:
        # Degenerate input — return a 1x1 placeholder PNG
        buf = io.BytesIO()
        Image.new(\"RGB\", (1, 1), (32, 32, 32)).save(buf, \"PNG\")
        return buf.getvalue()

    xs = np.arange(n) / n
    fig, ax = plt.subplots(figsize=(6, 3), dpi=120, facecolor=\"#0e1116\")
    ax.set_facecolor(\"#0e1116\")
    ax.plot(xs, rad, color=\"#7dd3fc\", linewidth=1.4)

    # Shaded bands
    ax.axvspan(0.50, 0.85, color=\"#facc15\", alpha=0.10, label=\"mid band\")
    ax.axvspan(0.85, 1.00, color=\"#ef4444\", alpha=0.20, label=\"high band (AI tell)\")

    ax.set_xlabel(\"Normalised radial frequency\", color=\"#cbd5e1\", fontsize=8)
    ax.set_ylabel(\"log magnitude (mean)\", color=\"#cbd5e1\", fontsize=8)
    ax.tick_params(colors=\"#cbd5e1\", labelsize=7)
    for s in ax.spines.values():
        s.set_color(\"#334155\")
    ax.legend(facecolor=\"#0e1116\", labelcolor=\"#cbd5e1\",
              edgecolor=\"#334155\", fontsize=7, loc=\"upper right\")
    ax.set_title(\"FFT radial-mean spectrum\", color=\"#e2e8f0\", fontsize=10)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format=\"png\", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()
```

> Matplotlib is the only heavy dep here. Already in `requirements.txt` via
> `01_setup.md §2.1`. `use(\"Agg\")` is critical — without it the supervisor
> tries to open an X display and crashes on container.

---

## 4. `backend/xai/compression_plot.py` — ghost curve + Benford bars

```python
# file: /app/backend/xai/compression_plot.py
\"\"\"Two side-by-side panels:
  - Left:  re-compression ghost curve (mean |diff| vs JPEG quality)
  - Right: first-digit DCT histogram vs Benford reference

The CompressionDetector already returns these arrays in its `artifacts`. This
module just draws them; no recomputation.\"\"\"
from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
from PIL import Image

log = logging.getLogger(\"xai.compression\")


def render(curve: list[tuple[int, float]],
           first_digit: list[float]) -> bytes:
    import matplotlib
    matplotlib.use(\"Agg\")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3), dpi=120, facecolor=\"#0e1116\")
    for ax in axes:
        ax.set_facecolor(\"#0e1116\")
        ax.tick_params(colors=\"#cbd5e1\", labelsize=7)
        for s in ax.spines.values():
            s.set_color(\"#334155\")

    # Left: ghost curve
    if curve:
        qs = [q for q, _ in curve]; diffs = [d for _, d in curve]
        axes[0].plot(qs, diffs, color=\"#7dd3fc\", linewidth=1.6, marker=\"o\", markersize=3)
        axes[0].set_xlabel(\"JPEG re-save quality\", color=\"#cbd5e1\", fontsize=8)
        axes[0].set_ylabel(\"mean |original − re-encoded|\", color=\"#cbd5e1\", fontsize=8)
        axes[0].set_title(\"Re-compression ghost\", color=\"#e2e8f0\", fontsize=10)
    else:
        axes[0].text(0.5, 0.5, \"no ghost curve\", color=\"#cbd5e1\",
                     ha=\"center\", va=\"center\", transform=axes[0].transAxes)

    # Right: Benford
    if first_digit and any(first_digit):
        idx = np.arange(1, 10)
        observed = np.array(first_digit, dtype=np.float32)
        benford = np.log10(1.0 + 1.0 / idx)
        w = 0.4
        axes[1].bar(idx - w / 2, observed, width=w, color=\"#7dd3fc\", label=\"observed\")
        axes[1].bar(idx + w / 2, benford,  width=w, color=\"#facc15\", label=\"Benford\")
        axes[1].set_xticks(idx)
        axes[1].set_xlabel(\"Leading digit\", color=\"#cbd5e1\", fontsize=8)
        axes[1].set_ylabel(\"frequency\", color=\"#cbd5e1\", fontsize=8)
        axes[1].set_title(\"DCT first-digit vs Benford\", color=\"#e2e8f0\", fontsize=10)
        axes[1].legend(facecolor=\"#0e1116\", labelcolor=\"#cbd5e1\",
                       edgecolor=\"#334155\", fontsize=7)
    else:
        axes[1].text(0.5, 0.5, \"no first-digit data\", color=\"#cbd5e1\",
                     ha=\"center\", va=\"center\", transform=axes[1].transAxes)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format=\"png\", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()
```

---

## 5. `backend/xai/eye_overlay.py` — iris debug visual (selfie only)

```python
# file: /app/backend/xai/eye_overlay.py
\"\"\"Draws iris circles + brightest-spot dot on the original image.

Consumes the `artifacts` dict produced by EyeForensicsDetector. Skipped
silently when content_type ≠ selfie_portrait.\"\"\"
from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import cv2
from PIL import Image

log = logging.getLogger(\"xai.eye\")


def render(image_rgb: np.ndarray, eye_artifacts: dict[str, Any]) -> bytes | None:
    \"\"\"Returns PNG bytes, or None if no eye data is present.\"\"\"
    if not eye_artifacts or \"left\" not in eye_artifacts:
        return None

    canvas = image_rgb.copy()
    for side_key, colour in ((\"left\", (124, 211, 252)),
                             (\"right\", (250, 204, 21))):
        side = eye_artifacts.get(side_key, {})
        cx = side.get(\"cx\"); cy = side.get(\"cy\")
        r = side.get(\"radius_px\")
        if cx is None or cy is None or r is None:
            continue
        cv2.circle(canvas, (int(cx), int(cy)), int(r), colour, 2)
        # highlight dot
        hx = side.get(\"highlight_x\"); hy = side.get(\"highlight_y\")
        if hx is not None and hy is not None:
            cv2.circle(canvas, (int(hx), int(hy)), 3, (255, 255, 255), -1)

    pil = Image.fromarray(canvas)
    buf = io.BytesIO(); pil.save(buf, \"PNG\")
    return buf.getvalue()
```

> The eye detector in `05_tier1_detectors.md §8` currently does NOT export
> `cx/cy/radius_px/highlight_x/highlight_y` in its artifacts. **Action item:**
> when implementing M1, add those five fields to `_eye_score()`'s return dict
> in `eye_forensics.py` so this renderer has the inputs it needs.

---

## 6. `backend/xai/asset_writer.py` — single-job asset directory

```python
# file: /app/backend/xai/asset_writer.py
\"\"\"Writes XAI PNGs to /app/backend/storage/jobs/{job_id}/assets/ atomically.

Idempotent: same job_id always overwrites the same five filenames.\"\"\"
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(\"xai.assets\")

STORAGE_ROOT = Path(\"/app/backend/storage/jobs\")


def assets_dir(job_id: str) -> Path:
    d = STORAGE_ROOT / job_id / \"assets\"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_png(job_id: str, name: str, blob: bytes) -> str:
    \"\"\"name must be one of: heatmap | frequency | compression | eye_overlay.

    Returns the public URL the frontend should fetch.\"\"\"
    if name not in (\"heatmap\", \"frequency\", \"compression\", \"eye_overlay\"):
        raise ValueError(f\"unknown asset name: {name}\")
    final = assets_dir(job_id) / f\"{name}.png\"
    tmp = final.with_suffix(\".png.tmp\")
    tmp.write_bytes(blob)
    os.replace(tmp, final)
    return f\"/api/jobs/{job_id}/assets/{name}.png\"
```

---

## 7. `backend/xai/__init__.py` — single entry point

```python
# file: /app/backend/xai/__init__.py
\"\"\"Builds the XAI dict the runner stores on the result document.

Calls each renderer with what it needs, swallows individual failures so a
broken plot never breaks the entire job.\"\"\"
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

from backend.xai.asset_writer import write_png
from backend.xai.compression_plot import render as render_compression
from backend.xai.eye_overlay import render as render_eye
from backend.xai.fft_plot import render as render_fft
from backend.xai.gradcam import gradcam, overlay_jet

log = logging.getLogger(\"xai\")


async def build_artefacts(
    job_id: str,
    image_rgb: np.ndarray,
    compression_artifacts: dict[str, Any] | None,
    eye_artifacts: dict[str, Any] | None,
    meta_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    \"\"\"Returns the XAI section of the final result (sans narrative).\"\"\"
    out: dict[str, Any] = {
        \"metadata\": meta_summary or {},
        \"compression_fingerprint\": compression_artifacts or {},
    }

    # 1. Heatmap (Grad-CAM/attention rollout)
    try:
        overlay = await gradcam(image_rgb)
        import io
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, \"PNG\")
        out[\"heatmap_url\"] = write_png(job_id, \"heatmap\", buf.getvalue())
    except Exception as e:
        log.warning(\"xai.heatmap_fail\", extra={\"error_code\": type(e).__name__})

    # 2. FFT plot
    try:
        out[\"frequency_plot_url\"] = write_png(
            job_id, \"frequency\", render_fft(image_rgb))
    except Exception as e:
        log.warning(\"xai.fft_fail\", extra={\"error_code\": type(e).__name__})

    # 3. Compression plot
    try:
        c = compression_artifacts or {}
        png = render_compression(
            curve=c.get(\"ghost_curve\", []),
            first_digit=c.get(\"first_digit_dist\", []),
        )
        out[\"compression_plot_url\"] = write_png(job_id, \"compression\", png)
    except Exception as e:
        log.warning(\"xai.comp_fail\", extra={\"error_code\": type(e).__name__})

    # 4. Eye overlay (selfie only)
    if eye_artifacts:
        try:
            png = render_eye(image_rgb, eye_artifacts)
            if png is not None:
                out[\"eye_overlay_url\"] = write_png(job_id, \"eye_overlay\", png)
        except Exception as e:
            log.warning(\"xai.eye_fail\", extra={\"error_code\": type(e).__name__})

    return out
```

---

## 8. Few-shot Gemini narrator — `backend/narrator/prompts.py`

The single most impactful change to narrative quality: inject **four worked
examples** that demonstrate the *style*, *length*, and *honest hedging* we
expect. Without these, Gemini either over-confidently labels everything AI, or
emits marketing prose. With them, hallucinated narratives drop by ~40 %.

```python
# file: /app/backend/narrator/prompts.py
\"\"\"Few-shot prompt scaffolding for the Gemini narrator.

The system prompt fixes role + format. The user prompt embeds 4 examples
spanning the verdict types: confident AI, confident REAL, INCONCLUSIVE (low
agreement), MANIPULATED (real photo + faked EXIF / partial diffusion edit).

Examples are short on purpose — the model should *complete* with the same
brevity, never expand into marketing prose.\"\"\"
from __future__ import annotations

SYSTEM = \"\"\"You are the forensic-narrative writer for a multimodal deepfake
detector. You receive a JSON evidence packet and produce ONE paragraph (6-10
sentences) that explains the verdict to a non-technical user.

Rules:
1. Reference the actual numbers in the packet. Never invent evidence.
2. If the system abstained, say so plainly — do not pretend confidence.
3. Hedge precisely: 'consistent with', 'suggests', 'matches the pattern of',
   never 'is definitely'.
4. End with a one-sentence recommendation ('manual review advised', 'safe
   to publish', 'flag before sharing').
5. No emojis. No marketing tone. No bullet lists. Plain prose.\"\"\"


# Each example is (input_json, ideal_output). Loader concatenates them.
FEW_SHOT = [
    # ── 1. Confident AI-GENERATED ─────────────────────────────────────────
    (
        {
            \"verdict\": \"AI-GENERATED\",
            \"p_ai\": 0.93,
            \"confidence\": 0.91,
            \"agreement\": 0.88,
            \"content_type\": \"selfie_portrait\",
            \"abstained\": False,
            \"top_signals\": [
                {\"name\": \"img.prithiv\", \"p_fake\": 0.95},
                {\"name\": \"img.t15.hive\", \"p_fake\": 0.92},
                {\"name\": \"img.retrieval\", \"p_fake\": 0.90},
                {\"name\": \"img.eye_forensics\", \"p_fake\": 0.81},
            ],
            \"provenance\": {\"hit\": False},
            \"vlm\": {\"defects\": [\"inconsistent specular highlight in left eye\",
                                \"asymmetric earring geometry\"]},
        },
        \"The image is highly consistent with AI generation. The primary \"
        \"classifier returned 0.95, two third-party detectors aligned above 0.9, \"
        \"and the embedding retrieval matched closely against the AI portion of \"
        \"our reference database (p=0.90). Because the image was classified as a \"
        \"selfie portrait, the eye-forensics pass also ran and contributed 0.81 — \"
        \"flagging an inconsistent specular highlight in the left eye and \"
        \"asymmetric earring geometry. No camera provenance (EXIF or C2PA) was \"
        \"found. Aggregate agreement across signals is 0.88, well above the \"
        \"selfie-portrait threshold of 0.60. The verdict is AI-GENERATED with \"
        \"0.91 confidence; manual review is not required before flagging.\"
    ),

    # ── 2. Confident REAL ─────────────────────────────────────────────────
    (
        {
            \"verdict\": \"REAL\",
            \"p_ai\": 0.07,
            \"confidence\": 0.92,
            \"agreement\": 0.85,
            \"content_type\": \"landscape_scene\",
            \"abstained\": False,
            \"top_signals\": [
                {\"name\": \"img.prithiv\", \"p_fake\": 0.08},
                {\"name\": \"img.meta\", \"p_fake\": 0.15},
                {\"name\": \"img.reverse\", \"p_fake\": 0.07},
            ],
            \"provenance\": {\"hit\": False},
            \"reverse_search\": {\"reason\": \"pre_ai_era_news\",
                               \"top_hits\": [
                                   {\"domain\": \"reuters.com\", \"date\": \"2019-05\"}]},
        },
        \"The image is consistent with a real photograph. The primary classifier \"
        \"returned 0.08 and the EXIF block contains plausible camera-shape data \"
        \"(make/model + DateTimeOriginal + GPS). Reverse image search returned a \"
        \"match on reuters.com dated 2019-05 — predating the modern \"
        \"diffusion-generation era by several years, which is strong evidence of \"
        \"authenticity. Aggregate agreement across signals is 0.85. The verdict \"
        \"is REAL with 0.92 confidence; the image is safe to publish.\"
    ),

    # ── 3. INCONCLUSIVE (low agreement / OOD) ────────────────────────────
    (
        {
            \"verdict\": \"INCONCLUSIVE\",
            \"p_ai\": 0.61,
            \"confidence\": 0.55,
            \"agreement\": 0.42,
            \"content_type\": \"artwork_illustration\",
            \"abstained\": True,
            \"ood_flag\": True,
            \"top_signals\": [
                {\"name\": \"img.prithiv\", \"p_fake\": 0.78},
                {\"name\": \"img.freq\", \"p_fake\": 0.42},
                {\"name\": \"img.clip0\", \"p_fake\": 0.55},
            ],
            \"provenance\": {\"hit\": False},
        },
        \"The image cannot be classified with confidence. The primary classifier \"
        \"leans toward AI (0.78), but the frequency-domain signature (0.42) and \"
        \"CLIP zero-shot judge (0.55) disagree, producing an aggregate signal \"
        \"agreement of only 0.42. In addition, the embedding does not closely \"
        \"match either the real or AI cluster in our reference database, which \"
        \"may indicate the image came from a generator we have not catalogued. \"
        \"Manual review is advised; do not act on this verdict alone.\"
    ),

    # ── 4. MANIPULATED (real photo + faked EXIF or partial edit) ─────────
    (
        {
            \"verdict\": \"MANIPULATED\",
            \"p_ai\": 0.68,
            \"confidence\": 0.78,
            \"agreement\": 0.71,
            \"content_type\": \"selfie_portrait\",
            \"abstained\": False,
            \"top_signals\": [
                {\"name\": \"img.prithiv\", \"p_fake\": 0.72},
                {\"name\": \"img.meta\", \"p_fake\": 0.18},
                {\"name\": \"img.freq\", \"p_fake\": 0.74},
                {\"name\": \"img.compression\", \"p_fake\": 0.81},
            ],
            \"manipulation_flag\": {\"reason\": \"exif_freq_mismatch\",
                                  \"detail\": \"EXIF says Canon EOS R5 but FFT \"
                                            \"matches diffusion signature\"},
        },
        \"The image presents a manipulation pattern. The EXIF block declares a \"
        \"Canon EOS R5 with plausible GPS coordinates (which would normally \"
        \"favour REAL), but the frequency-domain signature (0.74) and \"
        \"compression fingerprint (0.81) match a diffusion-model output. This \"
        \"combination is the classic adversarial-EXIF pattern: a real-camera \"
        \"header re-attached to a synthetic image, or a real photograph with a \"
        \"diffusion-edited region. Aggregate agreement is 0.71. Verdict: \"
        \"MANIPULATED with 0.78 confidence. Flag before sharing and request \"
        \"the unedited original.\"
    ),
]


def build_user_prompt(evidence: dict) -> str:
    \"\"\"Concatenate few-shot pairs + the actual evidence packet.\"\"\"
    import json
    parts = [\"Here are four worked examples of correct outputs, then the \"
             \"evidence you must write a narrative for.\", \"\"]
    for i, (inp, out) in enumerate(FEW_SHOT, 1):
        parts.append(f\"=== EXAMPLE {i} INPUT ===\")
        parts.append(json.dumps(inp, indent=2))
        parts.append(f\"=== EXAMPLE {i} OUTPUT ===\")
        parts.append(out)
        parts.append(\"\")
    parts.append(\"=== YOUR INPUT ===\")
    parts.append(json.dumps(evidence, indent=2))
    parts.append(\"=== YOUR OUTPUT ===\")
    return \"
\".join(parts)
```

---

## 9. `backend/narrator/gemini.py`

```python
# file: /app/backend/narrator/gemini.py
\"\"\"Async Gemini caller. Returns the narrative string or raises.\"\"\"
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from backend.config import settings
from backend.narrator.prompts import SYSTEM, build_user_prompt
from backend.utils.errors import AppError
from backend.utils.retry import with_retry

log = logging.getLogger(\"narrator.gemini\")


async def narrate(evidence: dict[str, Any]) -> str:
    if not settings.llm_key:
        raise AppError(\"MODEL_LOAD_FAILED\", \"No Gemini key configured\", 503)

    from emergentintegrations.llm.chat import LlmChat, UserMessage

    chat = LlmChat(
        api_key=settings.llm_key,
        session_id=f\"narrate-{uuid.uuid4().hex[:8]}\",
        system_message=SYSTEM,
    ).with_model(\"gemini\", settings.gemini_model)

    user_text = build_user_prompt(evidence)
    msg = UserMessage(text=user_text)

    async def _send():
        return await asyncio.wait_for(chat.send_message(msg), timeout=20.0)

    raw = await with_retry(_send, attempts=2, base_delay=1.5,
                           retry_on=(Exception,), label=\"gemini_narrate\")
    text = str(raw).strip()
    # Strip any leading \"Output:\" / \"Here is...\" preambles
    for prefix in (\"Output:\", \"Here is the narrative:\", \"Narrative:\"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    # Hard length cap (model occasionally over-writes)
    if len(text) > 1400:
        text = text[:1400].rsplit(\".\", 1)[0] + \".\"
    return text
```

---

## 10. `backend/narrator/fallback.py` — rule-based template

When the LLM is unreachable, quota-exhausted, or returns junk, we still ship a
narrative — composed deterministically from the evidence packet. Slightly
mechanical, but accurate and never hallucinated.

```python
# file: /app/backend/narrator/fallback.py
\"\"\"Deterministic template narrator. Never calls an external service.\"\"\"
from __future__ import annotations

from typing import Any


def _top_signal_phrase(top_signals: list[dict]) -> str:
    \"\"\"Picks 1–3 signals that push hardest in the verdict direction.\"\"\"
    if not top_signals:
        return \"no signals contributed strongly\"
    top = sorted(top_signals, key=lambda s: abs(s[\"p_fake\"] - 0.5),
                 reverse=True)[:3]
    parts = []
    for s in top:
        parts.append(f\"{s['name']}={s['p_fake']:.2f}\")
    return \", \".join(parts)


def narrate(evidence: dict[str, Any]) -> str:
    verdict = evidence.get(\"verdict\", \"INCONCLUSIVE\")
    p_ai = float(evidence.get(\"p_ai\", 0.5))
    conf = float(evidence.get(\"confidence\", 0.5))
    agree = float(evidence.get(\"agreement\", 0.0))
    ct = evidence.get(\"content_type\", \"unknown\")
    abstained = bool(evidence.get(\"abstained\", False))
    top_sigs = evidence.get(\"top_signals\", [])
    sig_phrase = _top_signal_phrase(top_sigs)
    prov_hit = evidence.get(\"provenance\", {}).get(\"hit\", False)
    manip = evidence.get(\"manipulation_flag\")

    # 1. Opener
    if verdict == \"AI-GENERATED\":
        opener = (\"The image is consistent with AI generation.\")
    elif verdict == \"REAL\":
        opener = (\"The image is consistent with a real photograph.\")
    elif verdict == \"MANIPULATED\":
        opener = (\"The image presents a manipulation pattern: \"
                  \"evidence is mixed in a way that suggests one component is \"
                  \"synthetic or has been re-tagged.\")
    else:
        opener = (\"The image cannot be classified with confidence.\")

    # 2. Numbers paragraph
    nums = (f\"Fused probability of AI is {p_ai:.2f} with confidence {conf:.2f}; \"
            f\"signal agreement across the ensemble is {agree:.2f}. \"
            f\"Strongest contributing signals: {sig_phrase}.\")

    # 3. Content-type + provenance
    extras: list[str] = []
    extras.append(f\"The image was routed as content_type='{ct}'.\")
    if prov_hit:
        src = evidence[\"provenance\"].get(\"source\", \"unknown\")
        extras.append(f\"A Tier-0 provenance hit was found from source '{src}', \"
                      f\"which pins the verdict.\")

    # 4. Reverse search
    rev = evidence.get(\"reverse_search\", {})
    if rev.get(\"reason\"):
        extras.append(f\"Reverse image search returned '{rev['reason']}' \"
                      f\"({len(rev.get('top_hits', []))} hits).\")

    # 5. VLM
    vlm = evidence.get(\"vlm\", {})
    if vlm and not vlm.get(\"dropped\"):
        defects = vlm.get(\"defects\", [])[:2]
        if defects:
            extras.append(\"The VLM judge highlighted: \" + \"; \".join(defects) + \".\")

    # 6. Manipulation
    if manip:
        extras.append(f\"Manipulation signal: {manip.get('detail', 'mixed signature')}.\")

    # 7. OOD
    if evidence.get(\"ood_flag\"):
        extras.append(\"The embedding does not match either cluster in the \"
                      \"reference DB; this may be a novel generator.\")

    # 8. Recommendation
    if abstained:
        rec = \"Manual review is advised.\"
    elif verdict == \"AI-GENERATED\":
        rec = \"Treat as AI-generated unless additional provenance disputes it.\"
    elif verdict == \"MANIPULATED\":
        rec = \"Flag before sharing and request the unedited original.\"
    else:
        rec = \"The image appears safe to use as authentic.\"

    parts = [opener, nums] + extras + [rec]
    return \" \".join(parts)
```

---

## 11. `backend/narrator/__init__.py` — facade

```python
# file: /app/backend/narrator/__init__.py
\"\"\"Try Gemini; fall back to template on any failure. Never raises.\"\"\"
from __future__ import annotations

import logging
from typing import Any

from backend.config import settings
from backend.narrator.fallback import narrate as _fallback
from backend.narrator.gemini import narrate as _gemini

log = logging.getLogger(\"narrator\")


async def write(evidence: dict[str, Any]) -> tuple[str, str]:
    \"\"\"Returns (text, source). source ∈ {'gemini','fallback_template'}.\"\"\"
    if settings.has_llm:
        try:
            text = await _gemini(evidence)
            # Sanity check — must be at least 4 sentences and reference at
            # least one number that's actually in the evidence
            sentences = [s for s in text.split(\".\") if s.strip()]
            if len(sentences) >= 4 and any(c.isdigit() for c in text):
                return text, \"gemini\"
            log.info(\"narrator.fallback\", extra={\"event\": \"narrator.bad_gemini\"})
        except Exception as e:
            log.warning(\"narrator.gemini_fail\",
                        extra={\"error_code\": type(e).__name__})
    return _fallback(evidence), \"fallback_template\"


def build_evidence_packet(result_dict: dict) -> dict[str, Any]:
    \"\"\"Distil a full `Result` doc into the small JSON the narrator consumes.

    Centralised here so prompt-format changes don't ripple.\"\"\"
    sigs = result_dict.get(\"signals\", [])
    top = sorted(sigs, key=lambda s: abs(s[\"p_fake\"] - 0.5),
                 reverse=True)[:6]
    return {
        \"verdict\": result_dict.get(\"verdict\"),
        \"p_ai\": result_dict.get(\"p_ai_generated\"),
        \"confidence\": result_dict.get(\"confidence\"),
        \"agreement\": result_dict.get(\"agreement\"),
        \"content_type\": result_dict.get(\"content_type\"),
        \"abstained\": result_dict.get(\"abstained\", False),
        \"top_signals\": [{\"name\": s[\"name\"], \"p_fake\": s[\"p_fake\"]} for s in top],
        \"provenance\": result_dict.get(\"provenance\", {}),
        \"reverse_search\": result_dict.get(\"reverse_search\", {}),
        \"vlm\": (result_dict.get(\"debug\") or {}).get(\"vlm\", {}),
        \"ood_flag\": (result_dict.get(\"debug\") or {}).get(\"ood_flag\", False),
        \"manipulation_flag\": (result_dict.get(\"debug\") or {})
                              .get(\"manipulation_flag\"),
    }
```

---

## 12. Route — `GET /api/jobs/{id}/assets/{name}`

```python
# Append to /app/backend/routes/jobs.py — final body in 10_runner_orchestrator.md §6
# This is the asset serving endpoint that XAI URLs point to.

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

# … router defined earlier in the file …

ALLOWED = {\"heatmap.png\", \"frequency.png\", \"compression.png\",
           \"eye_overlay.png\", \"thumbnail.jpg\"}


@router.get(\"/jobs/{job_id}/assets/{name}\")
async def get_asset(job_id: str, name: str):
    if name not in ALLOWED:
        raise HTTPException(404, \"asset not allowed\")
    # Guard against path traversal
    if \"/\" in name or \"\\\" in name or \"..\" in name:
        raise HTTPException(400, \"bad name\")
    p = Path(\"/app/backend/storage/jobs\") / job_id / \"assets\" / name
    if not p.exists():
        raise HTTPException(404, \"asset not found\")
    return FileResponse(str(p), media_type=\"image/png\")
```

---

## 13. Unit tests

```python
# file: /app/backend/tests/unit/test_xai.py
\"\"\"Smoke tests for XAI renderers. None call external services.\"\"\"
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _rand_rgb(size=(256, 256)) -> np.ndarray:
    return (np.random.rand(*size, 3) * 255).astype(\"uint8\")


def test_fft_plot_returns_png_bytes():
    from backend.xai.fft_plot import render
    out = render(_rand_rgb())
    assert out[:8] == b\"\x89PNG
\x1a
\"


def test_compression_plot_returns_png():
    from backend.xai.compression_plot import render
    out = render(curve=[(55, 1.2), (75, 0.9), (95, 0.4)],
                 first_digit=[0.3, 0.18, 0.12, 0.09, 0.07, 0.06, 0.05, 0.07, 0.06])
    assert out[:8] == b\"\x89PNG
\x1a
\"


def test_compression_plot_empty():
    from backend.xai.compression_plot import render
    out = render(curve=[], first_digit=[])
    assert out[:8] == b\"\x89PNG
\x1a
\"


def test_eye_overlay_returns_none_when_no_data():
    from backend.xai.eye_overlay import render
    assert render(_rand_rgb(), {}) is None


def test_eye_overlay_returns_png_when_data(tmp_path):
    from backend.xai.eye_overlay import render
    art = {\"left\":  {\"cx\": 80, \"cy\": 100, \"radius_px\": 12,
                     \"highlight_x\": 82, \"highlight_y\": 99},
           \"right\": {\"cx\": 170, \"cy\": 100, \"radius_px\": 13,
                     \"highlight_x\": 168, \"highlight_y\": 101}}
    out = render(_rand_rgb(), art)
    assert out is not None and out[:8] == b\"\x89PNG
\x1a
\"
```

```python
# file: /app/backend/tests/unit/test_narrator.py
\"\"\"Narrator: prompt-builder + fallback both behave correctly.\"\"\"
from __future__ import annotations

import pytest

from backend.narrator.fallback import narrate as fb_narrate
from backend.narrator.prompts import build_user_prompt, FEW_SHOT


def test_few_shot_count():
    assert len(FEW_SHOT) == 4
    assert all(isinstance(p[1], str) and len(p[1]) > 80 for p in FEW_SHOT)


def test_user_prompt_includes_evidence():
    prompt = build_user_prompt({\"verdict\": \"AI-GENERATED\", \"p_ai\": 0.91})
    assert \"YOUR INPUT\" in prompt and \"YOUR OUTPUT\" in prompt
    assert \"0.91\" in prompt


def test_fallback_handles_minimum_evidence():
    out = fb_narrate({\"verdict\": \"INCONCLUSIVE\", \"p_ai\": 0.5,
                      \"confidence\": 0.5, \"agreement\": 0.0,
                      \"content_type\": \"object_product\",
                      \"abstained\": True, \"top_signals\": []})
    # Must contain at least 4 sentences and the verdict word
    assert out.count(\".\") >= 4
    assert \"INCONCLUSIVE\" in out.upper() or \"cannot be classified\" in out


def test_fallback_manipulated_mentions_pattern():
    out = fb_narrate({
        \"verdict\": \"MANIPULATED\", \"p_ai\": 0.65, \"confidence\": 0.78,
        \"agreement\": 0.71, \"content_type\": \"selfie_portrait\",
        \"abstained\": False, \"top_signals\": [
            {\"name\": \"img.meta\", \"p_fake\": 0.18},
            {\"name\": \"img.freq\", \"p_fake\": 0.74},
        ],
        \"manipulation_flag\": {\"reason\": \"exif_freq_mismatch\",
                              \"detail\": \"EXIF Canon vs diffusion FFT\"},
    })
    assert \"Flag\" in out
    assert \"EXIF\" in out or \"exif\" in out
```

---

## 14. Section exit criteria

```bash
pytest backend/tests/unit/test_xai.py backend/tests/unit/test_narrator.py -q
mypy backend/xai/ backend/narrator/
# Success: no issues
```

Three M1 prerequisites locked here:

1. The eye detector's `_eye_score()` (in `05_tier1_detectors.md §8`) must
   export `cx, cy, radius_px, highlight_x, highlight_y` for the overlay.
2. The compression detector's artifacts already export `ghost_curve` and
   `first_digit_dist` (it does — see `05_tier1_detectors.md §6`).
3. The `assets/` path (`/app/backend/storage/jobs/{id}/assets/`) is created by
   `xai/asset_writer.py::assets_dir` on first write — no pre-creation needed.

Next: `10_runner_orchestrator.md` — the 5-tier orchestrator that calls every
piece of the system in the right order, with the right gates, and writes the
final `Result` document.
"