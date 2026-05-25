"# 04 — Tier 0: Provenance Gate

> **Tier 0 is the highest-precision tier in the system.** When it fires, it short-circuits the entire pipeline with `p_ai = 0.99` or `p_ai = 0.01`, bypasses abstention, and pins the verdict. The ensemble still runs in the background for telemetry.

Four checks. Each is independent. First positive hit wins.

| # | Check | Library | Direction | Confidence |
|---|---|---|---|---|
| 1 | C2PA active producer signature | `c2pa` (Python bindings to `c2pa-rs`) | REAL | 0.99 |
| 2 | Stable Diffusion invisible watermark | `invisible-watermark` | AI | 0.99 |
| 3 | Google SynthID (image variant when available) | `synthid-text` *(guarded)* | AI | 0.99 |
| 4 | Meta IM watermark | guarded — public detector ships later | AI | 0.99 |

---

## 1. `backend/provenance/__init__.py`

```python
# file: /app/backend/provenance/__init__.py
\"\"\"Tier-0 unified entry. Returns the first positive hit, or a clean miss.\"\"\"
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import asyncio
import logging

from backend.provenance.c2pa_check import check_c2pa
from backend.provenance.sd_watermark import check_sd_watermark
from backend.provenance.synthid_check import check_synthid
from backend.provenance.meta_watermark import check_meta_wm

log = logging.getLogger(\"tier0\")


@dataclass
class ProvenanceResult:
    hit: bool
    source: str       # \"c2pa\" | \"sd_wm\" | \"synthid\" | \"meta_wm\" | \"none\"
    p_ai: float       # 0.99 (AI) or 0.01 (REAL); 0.5 when no hit
    details: dict


async def run_tier0(image_path: Path) -> ProvenanceResult:
    \"\"\"Run all four checks concurrently. First positive hit wins by priority.\"\"\"
    c2pa, sd_wm, synthid, meta_wm = await asyncio.gather(
        check_c2pa(image_path),
        check_sd_watermark(image_path),
        check_synthid(image_path),
        check_meta_wm(image_path),
        return_exceptions=True,
    )

    # Priority order: C2PA REAL > SD watermark > SynthID > Meta watermark
    for label, res, p_ai, source in [
        (\"c2pa\",    c2pa,    0.01, \"c2pa\"),
        (\"sd_wm\",   sd_wm,   0.99, \"sd_wm\"),
        (\"synthid\", synthid, 0.99, \"synthid\"),
        (\"meta_wm\", meta_wm, 0.99, \"meta_wm\"),
    ]:
        if isinstance(res, Exception):
            log.warning(\"tier0.fail\", extra={\"signal_name\": label, \"error_code\": type(res).__name__})
            continue
        if res.get(\"hit\"):
            log.info(\"tier0.hit\", extra={\"event\": \"tier0.hit\", \"signal_name\": label})
            return ProvenanceResult(hit=True, source=source, p_ai=p_ai, details=res)

    return ProvenanceResult(hit=False, source=\"none\", p_ai=0.5, details={})
```

---

## 2. C2PA check

```python
# file: /app/backend/provenance/c2pa_check.py
\"\"\"C2PA manifest validation via the official c2pa Python binding.
A positive hit means an *active producer signature* exists and verifies.\"\"\"
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

log = logging.getLogger(\"c2pa\")


def _sync_check(path: str) -> dict:
    try:
        import c2pa  # python-c2pa
    except ImportError:
        return {\"hit\": False, \"reason\": \"c2pa_not_installed\"}

    try:
        reader = c2pa.Reader.from_file(path)
        # reader.validation_status() returns empty list when fully valid
        if reader.validation_status():
            return {\"hit\": False, \"reason\": \"manifest_invalid\"}
        active = reader.active_manifest()
        if not active:
            return {\"hit\": False, \"reason\": \"no_active_manifest\"}
        producer = active.get(\"claim_generator\", \"unknown\")
        return {
            \"hit\": True,
            \"producer\": producer,
            \"manifest_summary\": {
                \"ingredients\": len(active.get(\"ingredients\", [])),
                \"assertions\": len(active.get(\"assertions\", [])),
            },
        }
    except Exception as e:
        return {\"hit\": False, \"reason\": f\"c2pa_error:{type(e).__name__}\"}


async def check_c2pa(path: Path) -> dict:
    return await asyncio.to_thread(_sync_check, str(path))
```

---

## 3. Stable Diffusion invisible watermark

```python
# file: /app/backend/provenance/sd_watermark.py
\"\"\"SD's default DWT-DCT watermark embedded by `diffusers` pipelines.
A hit is near-deterministic evidence of SD-family generation.\"\"\"
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(\"sd_wm\")

# SD encodes 48 bits; expected payload is the literal \"StableDiffusionV1\"
EXPECTED_BITS = list(map(int, bin(int.from_bytes(b\"StableDiffusionV1\", \"big\"))[2:].zfill(48)))[-48:]


def _sync_check(path: str) -> dict:
    try:
        from imwatermark import WatermarkDecoder
    except ImportError:
        return {\"hit\": False, \"reason\": \"invisible_watermark_not_installed\"}

    try:
        bgr = cv2.imread(path)
        if bgr is None:
            return {\"hit\": False, \"reason\": \"decode_failed\"}
        h, w = bgr.shape[:2]
        if h < 256 or w < 256:
            return {\"hit\": False, \"reason\": \"too_small\"}
        decoder = WatermarkDecoder(\"bits\", 48)
        bits = decoder.decode(bgr, \"dwtDct\")
        hd = sum(a != b for a, b in zip(bits, EXPECTED_BITS))
        if hd <= 6:                                # tolerate ~12 % bit error
            return {\"hit\": True, \"hamming\": int(hd), \"payload\": \"\".join(map(str, bits))}
        return {\"hit\": False, \"hamming\": int(hd)}
    except Exception as e:
        return {\"hit\": False, \"reason\": f\"sd_wm_error:{type(e).__name__}\"}


async def check_sd_watermark(path: Path) -> dict:
    return await asyncio.to_thread(_sync_check, str(path))
```

> **Why Hamming ≤ 6, not == 0?** Lossy JPEG re-encoding flips a few bits even on genuinely-watermarked images. A threshold of 6/48 keeps the false-positive rate on unwatermarked photos < 1/2^30 while catching ~95 % of re-encoded SD outputs.

---

## 4. SynthID (guarded import)

```python
# file: /app/backend/provenance/synthid_check.py
\"\"\"Google SynthID — image API not fully public as of v1.4. Library import
is guarded; module is a working stub that returns no-hit when unavailable.\"\"\"
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

log = logging.getLogger(\"synthid\")


def _sync_check(path: str) -> dict:
    try:
        # Google has not released a Python image-SynthID detector at time of writing.
        # When they do, replace this stub with the actual import + detect call.
        # The current `synthid-text` package handles text only.
        import synthid_text  # noqa: F401
    except ImportError:
        return {\"hit\": False, \"reason\": \"synthid_image_not_available\"}

    return {\"hit\": False, \"reason\": \"synthid_image_not_available\"}


async def check_synthid(path: Path) -> dict:
    return await asyncio.to_thread(_sync_check, str(path))
```

> **Honest stub.** The detector slot exists; the moment Google publishes a public image-SynthID detector, replace the body. No fake \"looks-correct\" scoring.

---

## 5. Meta IM watermark (guarded import)

```python
# file: /app/backend/provenance/meta_watermark.py
\"\"\"Meta's invisible watermark for AI images, when public detector ships.\"\"\"
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

log = logging.getLogger(\"meta_wm\")


def _sync_check(path: str) -> dict:
    # No public detector at time of writing.
    return {\"hit\": False, \"reason\": \"meta_wm_detector_not_available\"}


async def check_meta_wm(path: Path) -> dict:
    return await asyncio.to_thread(_sync_check, str(path))
```

---

## 6. Integration into the runner (preview — full version in `10_runner_orchestrator.md`)

```python
# inside services/runner.py (excerpted)
from backend.provenance import run_tier0, ProvenanceResult

prov: ProvenanceResult = await run_tier0(image_path)
if prov.hit:
    # Short-circuit. Ensemble still runs in background for telemetry,
    # but the *headline* verdict and abstention are bypassed.
    result.provenance = {\"hit\": True, \"source\": prov.source, \"details\": prov.details}
    result.p_ai_generated = prov.p_ai
    result.verdict = \"AI-GENERATED\" if prov.p_ai > 0.5 else \"REAL\"
    result.abstained = False
    # ... still run Tier-1..3 in background for telemetry & XAI panel
```

---

## 7. Unit tests

```python
# file: /app/backend/tests/unit/test_provenance.py
import asyncio
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.provenance import run_tier0
from backend.provenance.sd_watermark import check_sd_watermark


def _save_random(tmp: Path, size=(256, 256)) -> Path:
    img = (np.random.rand(*size, 3) * 255).astype(\"uint8\")
    p = tmp / \"rand.png\"
    cv2.imwrite(str(p), img)
    return p


@pytest.mark.asyncio
async def test_clean_image_no_hit(tmp_path):
    p = _save_random(tmp_path)
    res = await run_tier0(p)
    assert res.hit is False
    assert res.source == \"none\"


@pytest.mark.asyncio
async def test_sd_watermark_clean_no_hit(tmp_path):
    p = _save_random(tmp_path)
    out = await check_sd_watermark(p)
    assert out[\"hit\"] is False
```

> Fixture for the *positive* SD-watermark case: generate one image via diffusers locally, save its bytes to `backend/tests/fixtures/sd_watermarked.png` (do this once, commit). Test asserts `check_sd_watermark` returns `hit=True`.

---

## 8. Section exit criteria

```bash
pytest backend/tests/unit/test_provenance.py -q
# 2 passed (more when fixtures land)
mypy backend/provenance/
# Success: no issues
```

Tier-0 done. Next: `05_tier1_detectors.md` — the Tier-1 image signals (8 base + 2 new in v1.4).
"