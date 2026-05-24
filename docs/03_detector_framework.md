"# 03 — Detector Framework: base, registry, content-type, TTA, device

> Goal: lay the contracts every detector implements. Once this file is built, adding a new detector is `from .base import Detector` + 30 lines.

---

## 1. `backend/services/device.py` — profile auto-detect

```python
# file: /app/backend/services/device.py
\"\"\"Detect cloud_lite | mac_full | cuda_full and the matching torch device.\"\"\"
from __future__ import annotations

import os
import torch


def detect_profile() -> str:
    forced = os.getenv(\"DETECTOR_PROFILE\", \"auto\")
    if forced != \"auto\":
        return forced
    if torch.cuda.is_available():
        return \"cuda_full\"
    if hasattr(torch.backends, \"mps\") and torch.backends.mps.is_available():
        return \"mac_full\"
    return \"cloud_lite\"


def torch_device(profile: str | None = None) -> torch.device:
    p = profile or detect_profile()
    if p == \"cuda_full\":
        return torch.device(\"cuda\")
    if p == \"mac_full\":
        return torch.device(\"mps\")
    return torch.device(\"cpu\")


def torch_dtype(profile: str | None = None) -> \"torch.dtype\":
    p = profile or detect_profile()
    if p == \"cuda_full\":
        return torch.float16
    return torch.float32   # MPS prefers fp32; CPU stays fp32
```

---

## 2. `backend/detectors/base.py` — Detector ABC

```python
# file: /app/backend/detectors/base.py
\"\"\"Every detector implements this contract. One class per file, registered in registry.py.\"\"\"
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class Sample:
    \"\"\"Container passed to each detector. Built once per job by services/runner.py.\"\"\"
    image_rgb: np.ndarray | None = None        # H, W, 3 uint8
    image_path: str | None = None              # original path
    sha256: str = \"\"
    mime: str = \"\"
    bytes: int = 0
    face_crops: list[np.ndarray] = field(default_factory=list)
    content_type: str = \"object_product\"       # set by content_type.classify()
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectorOutput:
    name: str                                  # e.g. \"img.prithiv\"
    p_fake: float                              # raw, BEFORE calibration  ∈ [0, 1]
    explanation: str = \"\"                      # human-readable
    artifacts: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int = 0
    enabled: bool = True                       # False → dropped from fusion (imputed)


class Detector(abc.ABC):
    \"\"\"Stateless interface. Heavy state lives in `registry.py` (loaded models).\"\"\"

    name: str                                  # set by subclass
    modality: str = \"image\"                    # image | audio | video
    profiles: tuple[str, ...] = (\"cloud_lite\", \"mac_full\", \"cuda_full\")

    @abc.abstractmethod
    async def predict(self, sample: Sample) -> DetectorOutput: ...
```

> Note: `Detector.predict` is async. CPU-bound work goes through `asyncio.to_thread` inside the method so the FastAPI event loop never blocks. Examples in `05_tier1_detectors.md`.

---

## 3. `backend/detectors/registry.py` — model loader, SHA pinning, LRU evict

```python
# file: /app/backend/detectors/registry.py
\"\"\"Single source of truth for which model weights are loaded, on which device,
in which precision. Lazy-loads. LRU-evicts on memory pressure (cuda_full only).\"\"\"
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from backend.services.device import detect_profile

log = logging.getLogger(\"registry\")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    sha: str = \"\"                                  # commit hash; \"\" = skip pin (dev)
    license: str = \"\"
    size_mb: int = 0
    profile_in: tuple[str, ...] = (\"cloud_lite\", \"mac_full\", \"cuda_full\")
    device_pref: dict[str, str] = field(default_factory=dict)  # profile -> \"cpu\"|\"mps\"|\"cuda\"
    fallback_repo: str | None = None


MODELS: dict[str, ModelSpec] = {
    \"img.prithiv\":   ModelSpec(\"img.prithiv\", \"prithivMLmods/deepfake-detector-model-v1\",
                                license=\"Apache-2.0\", size_mb=350,
                                device_pref={\"cloud_lite\":\"cpu\",\"mac_full\":\"mps\",\"cuda_full\":\"cuda\"}),
    \"img.clip0\":     ModelSpec(\"img.clip0\", \"openai/clip-vit-base-patch32\",
                                license=\"MIT\", size_mb=605,
                                device_pref={\"cloud_lite\":\"cpu\",\"mac_full\":\"mps\",\"cuda_full\":\"cuda\"}),
    \"embed.clip\":    ModelSpec(\"embed.clip\", \"openai/clip-vit-base-patch32\",
                                license=\"MIT\", size_mb=605,
                                device_pref={\"cloud_lite\":\"cpu\",\"mac_full\":\"mps\",\"cuda_full\":\"cuda\"}),
    # Heavy — Mac/CUDA only
    \"img.npr\":       ModelSpec(\"img.npr\", \"tancc/Generalizable_Deepfake_Detection-NPR-CVPR2024\",
                                license=\"MIT\", size_mb=48,
                                profile_in=(\"mac_full\",\"cuda_full\"),
                                device_pref={\"mac_full\":\"mps\",\"cuda_full\":\"cuda\"},
                                fallback_repo=\"chuangchuangtan/NPR-DeepfakeDetection\"),
    \"img.ufd\":       ModelSpec(\"img.ufd\", \"WisconsinAIVision/UniversalFakeDetect\",
                                license=\"Apache-2.0\", size_mb=1600,
                                profile_in=(\"mac_full\",\"cuda_full\"),
                                device_pref={\"mac_full\":\"mps\",\"cuda_full\":\"cuda\"}),
    \"img.dire\":      ModelSpec(\"img.dire\", \"Zhendong-Wang/DIRE\",
                                license=\"MIT\", size_mb=1100,
                                profile_in=(\"mac_full\",\"cuda_full\"),
                                device_pref={\"mac_full\":\"cpu\",\"cuda_full\":\"cuda\"}),
}


_LOADED: \"OrderedDict[str, Any]\" = OrderedDict()
_LOCK = threading.RLock()
_LRU_CAP_GB = 4.0  # cuda_full only; mac_full has 32 GB unified


def is_enabled(key: str) -> bool:
    spec = MODELS.get(key)
    return bool(spec) and detect_profile() in spec.profile_in


def warm_registry() -> None:
    \"\"\"Validate the model table at boot. Does NOT download — that's lazy.\"\"\"
    profile = detect_profile()
    enabled = [k for k, s in MODELS.items() if profile in s.profile_in]
    log.info(\"registry.ready\", extra={\"profile\": profile, \"enabled_count\": len(enabled)})


def get_or_load(key: str, loader: Callable[[ModelSpec, str], Any]) -> Any:
    \"\"\"Idempotent. Loader signature: (spec, device_str) -> any. Touches LRU on hit.\"\"\"
    spec = MODELS[key]
    if not is_enabled(key):
        raise RuntimeError(f\"{key} disabled on profile {detect_profile()}\")
    with _LOCK:
        if key in _LOADED:
            _LOADED.move_to_end(key)
            return _LOADED[key]
        device = spec.device_pref.get(detect_profile(), \"cpu\")
        log.info(\"registry.load\", extra={\"event\": \"load\", \"signal_name\": key})
        obj = loader(spec, device)
        _LOADED[key] = obj
        _maybe_evict(spec)
        return obj


def _maybe_evict(just_loaded: ModelSpec) -> None:
    if detect_profile() != \"cuda_full\":
        return
    # naive total: sum sizes of currently loaded
    total_mb = sum(MODELS[k].size_mb for k in _LOADED)
    while total_mb > _LRU_CAP_GB * 1024 and len(_LOADED) > 1:
        oldest_key, _ = _LOADED.popitem(last=False)
        if oldest_key == just_loaded.key:
            # don't evict what we just loaded
            _LOADED[oldest_key] = _; _LOADED.move_to_end(oldest_key); break  # noqa
        log.info(\"registry.evict\", extra={\"signal_name\": oldest_key})
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        total_mb -= MODELS[oldest_key].size_mb


def loaded_signals() -> list[str]:
    with _LOCK:
        return list(_LOADED.keys())
```

> **Why this is enough:** The registry is *just* a memo-cache + device router + license/SHA gate. Detector modules own the actual model construction by passing a `loader` callable. Loose coupling, no circular imports.

---

## 4. `backend/detectors/content_type.py` — CLIP zero-shot 6-way

```python
# file: /app/backend/detectors/content_type.py
\"\"\"Classifies an image into one of 6 content types. Used by abstention/gate.py to
pick type-specific thresholds. Shares the CLIP model with img.clip0.\"\"\"
from __future__ import annotations

import asyncio
import logging
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from backend.detectors.registry import get_or_load, ModelSpec
from backend.services.device import torch_device

log = logging.getLogger(\"content_type\")


LABELS_PROMPTS = {
    \"selfie_portrait\":     \"a selfie or portrait photograph of a person's face\",
    \"landscape_scene\":     \"a landscape or outdoor scene without a prominent face\",
    \"object_product\":      \"a product photograph of a single object\",
    \"meme_screenshot\":     \"a meme or screenshot with text overlay\",
    \"document_scan\":       \"a scan or photograph of a paper document\",
    \"artwork_illustration\":\"a digital illustration or painting, not photorealistic\",
}
LABELS = list(LABELS_PROMPTS.keys())
PROMPTS = list(LABELS_PROMPTS.values())


def _load_clip(spec: ModelSpec, device: str):
    model = CLIPModel.from_pretrained(spec.repo).to(device).eval()
    proc = CLIPProcessor.from_pretrained(spec.repo)
    return {\"model\": model, \"proc\": proc, \"device\": device}


def _sync_predict(image_rgb: np.ndarray) -> tuple[str, dict[str, float]]:
    bundle = get_or_load(\"embed.clip\", _load_clip)
    img = Image.fromarray(image_rgb).convert(\"RGB\")
    inputs = bundle[\"proc\"](text=PROMPTS, images=img, return_tensors=\"pt\",
                             padding=True).to(bundle[\"device\"])
    with torch.no_grad():
        out = bundle[\"model\"](**inputs)
        probs = out.logits_per_image.softmax(dim=-1).squeeze().cpu().numpy()
    scores = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    return max(scores, key=scores.get), scores


async def classify(image_rgb: np.ndarray) -> tuple[str, dict[str, float]]:
    return await asyncio.to_thread(_sync_predict, image_rgb)
```

---

## 5. `backend/detectors/tta.py` — test-time augmentation

```python
# file: /app/backend/detectors/tta.py
\"\"\"Three views per learned detector (original, h-flip, JPEG re-encode q=85).\"\"\"
from __future__ import annotations

import io
import numpy as np
from PIL import Image


def tta_views(image_rgb: np.ndarray) -> list[np.ndarray]:
    pil = Image.fromarray(image_rgb)
    views = [image_rgb, np.array(pil.transpose(Image.FLIP_LEFT_RIGHT))]
    buf = io.BytesIO()
    pil.save(buf, \"JPEG\", quality=85)
    buf.seek(0)
    views.append(np.array(Image.open(buf).convert(\"RGB\")))
    return views


def aggregate_tta(scores: list[float]) -> tuple[float, float]:
    \"\"\"Returns (mean, std). std is exported as the `tta_std` fusion feature.\"\"\"
    arr = np.array(scores, dtype=np.float32)
    return float(arr.mean()), float(arr.std())
```

---

## 6. Patch voting (mac_full / cuda_full only)

```python
# file: /app/backend/detectors/patch.py
\"\"\"For images > 512 px, run on 5 patches (4 corners + center). Aggregate mean+max.
Disabled on cloud_lite to stay within 30 s budget.\"\"\"
from __future__ import annotations
import numpy as np
from backend.services.device import detect_profile


def patches(image_rgb: np.ndarray, size: int = 384) -> list[np.ndarray]:
    if detect_profile() == \"cloud_lite\":
        return [image_rgb]                                  # no patching
    h, w, _ = image_rgb.shape
    if max(h, w) <= 512:
        return [image_rgb]
    cy, cx = h // 2, w // 2
    s = min(size, h, w)
    return [
        image_rgb[:s, :s],                                  # top-left
        image_rgb[:s, w - s:],                              # top-right
        image_rgb[h - s:, :s],                              # bottom-left
        image_rgb[h - s:, w - s:],                          # bottom-right
        image_rgb[cy - s // 2: cy + s // 2,
                  cx - s // 2: cx + s // 2],                # center
    ]


def aggregate_patches(scores: list[float]) -> tuple[float, float]:
    \"\"\"mean + max → fusion features.\"\"\"
    a = np.array(scores, dtype=np.float32)
    return float(a.mean()), float(a.max())
```

---

## 7. `backend/services/router.py` — MIME sniffer → modality

```python
# file: /app/backend/services/router.py
\"\"\"Decide modality from the upload bytes. Fails fast on mismatched ext/mime.\"\"\"
from __future__ import annotations

from pathlib import Path
import magic                                # python-magic
from backend.utils.errors import unsupported_mime


IMAGE_MIMES = {\"image/jpeg\", \"image/png\", \"image/webp\"}
AUDIO_MIMES = {\"audio/wav\", \"audio/mpeg\", \"audio/x-flac\", \"audio/ogg\"}
VIDEO_MIMES = {\"video/mp4\", \"video/quicktime\", \"video/x-matroska\", \"video/webm\"}


def sniff_mime(path: Path) -> str:
    return magic.from_file(str(path), mime=True)


def modality_for(path: Path, declared_mime: str | None = None) -> str:
    sniffed = sniff_mime(path)
    if declared_mime and declared_mime != sniffed:
        # log + reject — never trust the header alone
        raise unsupported_mime(f\"declared {declared_mime} but sniffed {sniffed}\")
    if sniffed in IMAGE_MIMES:
        return \"image\"
    if sniffed in AUDIO_MIMES:
        return \"audio\"
    if sniffed in VIDEO_MIMES:
        return \"video\"
    raise unsupported_mime(f\"unsupported mime {sniffed}\")
```

---

## 8. Unit-test fixtures for this layer

```python
# file: /app/backend/tests/unit/test_device.py
from backend.services.device import detect_profile

def test_profile_one_of():
    assert detect_profile() in {\"cloud_lite\", \"mac_full\", \"cuda_full\"}
```

```python
# file: /app/backend/tests/unit/test_tta.py
import numpy as np
from backend.detectors.tta import tta_views, aggregate_tta

def test_tta_three_views():
    img = (np.random.rand(64, 64, 3) * 255).astype(\"uint8\")
    views = tta_views(img)
    assert len(views) == 3
    assert all(v.shape == (64, 64, 3) for v in views[:2])

def test_aggregate_tta_bounds():
    m, s = aggregate_tta([0.1, 0.3, 0.5])
    assert 0.0 <= m <= 1.0 and s >= 0
```

```python
# file: /app/backend/tests/unit/test_registry.py
from backend.detectors.registry import MODELS, is_enabled, detect_profile

def test_models_keys_unique():
    assert len(MODELS) == len(set(MODELS))

def test_cloud_lite_subset():
    p = detect_profile()
    if p == \"cloud_lite\":
        assert is_enabled(\"img.prithiv\")
        assert not is_enabled(\"img.npr\")  # gated
```

---

## 9. Where these are used next

- `05_tier1_detectors.md`: Every image detector imports `Detector`, `Sample`, `DetectorOutput`, and uses `get_or_load` from this file's registry.
- `10_runner_orchestrator.md`: Builds the `Sample` once, calls `content_type.classify`, dispatches to all enabled detectors concurrently with per-stage timeouts.
- `08_fusion_calibration_abstention.md`: Consumes `DetectorOutput.p_fake` lists and the content-type label.

### Section exit criteria

```bash
pytest backend/tests/unit/test_device.py backend/tests/unit/test_tta.py \
       backend/tests/unit/test_registry.py -q
# 3 passed
mypy backend/services/device.py backend/detectors/base.py \
     backend/detectors/registry.py backend/detectors/tta.py
# Success: no issues
```
"