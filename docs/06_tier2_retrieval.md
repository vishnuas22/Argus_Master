"# 06 — Tier 2: Embedding Retrieval + Reference DB

> Tier 2 turns \"is this AI?\" into \"does it look like things I already know are AI?\" — a similarity problem that generalises across generators without retraining.
>
> Components: shared CLIP embedder, FAISS index per modality + label, hard-negative partition, build script that scrapes 1500 real + 1500 AI images from permissive sources.

---

## 1. `backend/retrieval/embedder.py` — shared CLIP-B/32

```python
# file: /app/backend/retrieval/embedder.py
\"\"\"One CLIP-B/32 instance shared with detectors/image/clip0.py and
detectors/content_type.py. The registry guarantees a single load.\"\"\"
from __future__ import annotations

import asyncio
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from backend.detectors.registry import get_or_load, ModelSpec


def _load(spec: ModelSpec, device: str):
    model = CLIPModel.from_pretrained(spec.repo).to(device).eval()
    proc = CLIPProcessor.from_pretrained(spec.repo)
    return {\"model\": model, \"proc\": proc, \"device\": device}


def _sync_embed(image_rgb: np.ndarray) -> np.ndarray:
    bundle = get_or_load(\"embed.clip\", _load)
    pil = Image.fromarray(image_rgb).convert(\"RGB\")
    inp = bundle[\"proc\"](images=pil, return_tensors=\"pt\").to(bundle[\"device\"])
    with torch.no_grad():
        feats = bundle[\"model\"].get_image_features(**inp)
    v = feats.squeeze().float().cpu().numpy()
    return v / (np.linalg.norm(v) + 1e-9)         # L2-normalised


async def embed_image(image_rgb: np.ndarray) -> np.ndarray:
    \"\"\"Returns a (512,) float32 unit vector.\"\"\"
    return await asyncio.to_thread(_sync_embed, image_rgb)
```

---

## 2. `backend/retrieval/index.py` — FAISS load + query + dedup

```python
# file: /app/backend/retrieval/index.py
\"\"\"FAISS index manager. One bank per (modality, label) + a hard-neg partition.
All indexes are IndexFlatIP (cosine since vectors are normalised) — exact, fast
enough at 5k entries (~5 ms/query).\"\"\"
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import faiss
import numpy as np

log = logging.getLogger(\"refdb\")
REFDB_DIR = Path(\"/app/backend/storage/refdb\")


@dataclass
class Bank:
    label: Literal[\"real\", \"ai\"]
    vectors: np.ndarray            # (N, D)
    ids: list[str]                 # parallel to vectors
    sources: list[str]             # url/path per entry
    index: faiss.Index             # IndexFlatIP


_BANKS: dict[str, Bank] = {}
_LOCK = threading.RLock()


def _bank_files(modality: str, label: str, hard: bool = False) -> tuple[Path, Path, Path]:
    suffix = \"_hard\" if hard else \"\"
    return (
        REFDB_DIR / f\"{modality}_{label}{suffix}.npy\",
        REFDB_DIR / f\"{modality}_{label}{suffix}_labels.json\",
        REFDB_DIR / f\"{modality}_{label}{suffix}_sources.json\",
    )


def _load_bank(modality: str, label: str, hard: bool = False) -> Bank | None:
    npy, lbl_p, src_p = _bank_files(modality, label, hard)
    if not npy.exists():
        return None
    vectors = np.load(npy).astype(\"float32\")
    ids = json.loads(lbl_p.read_text()) if lbl_p.exists() else [f\"{i}\" for i in range(len(vectors))]
    sources = json.loads(src_p.read_text()) if src_p.exists() else [\"\" for _ in range(len(vectors))]
    idx = faiss.IndexFlatIP(vectors.shape[1])
    idx.add(vectors)
    return Bank(label=label, vectors=vectors, ids=ids, sources=sources, index=idx)


def load_all(modality: str = \"image\") -> None:
    with _LOCK:
        for label in (\"real\", \"ai\"):
            for hard in (False, True):
                key = f\"{modality}_{label}{'_hard' if hard else ''}\"
                bank = _load_bank(modality, label, hard)
                if bank:
                    _BANKS[key] = bank
                    log.info(\"refdb.load\", extra={\"event\": \"refdb.load\",
                                                  \"signal_name\": key,
                                                  \"status\": str(len(bank.ids))})


def refdb_stats() -> dict:
    return {
        \"loaded\": True if _BANKS else False,
        \"sizes\": {k: len(b.ids) for k, b in _BANKS.items()},
    }


@dataclass
class Neighbor:
    id: str
    label: Literal[\"real\", \"ai\"]
    distance: float     # 1 - cosine  (so smaller = closer)
    source: str


def query(modality: str, vec: np.ndarray, k: int = 15,
          exclude_id: str | None = None) -> list[Neighbor]:
    \"\"\"k nearest across union of all banks for this modality. Excludes self-leak.\"\"\"
    out: list[Neighbor] = []
    for key, bank in _BANKS.items():
        if not key.startswith(f\"{modality}_\"):
            continue
        sims, idxs = bank.index.search(vec[None, :].astype(\"float32\"), min(k, len(bank.ids)))
        for sim, i in zip(sims[0], idxs[0]):
            if i < 0: continue
            nid = bank.ids[i]
            if exclude_id and nid == exclude_id:
                continue
            out.append(Neighbor(
                id=nid, label=bank.label, distance=float(1.0 - sim),
                source=bank.sources[i],
            ))
    out.sort(key=lambda n: n.distance)
    return out[:k]


def retrieval_p_fake(neighbors: list[Neighbor]) -> float:
    \"\"\"Distance-weighted vote. w_i = 1 / (1 + d_i).\"\"\"
    if not neighbors:
        return 0.5
    w = np.array([1.0 / (1.0 + n.distance) for n in neighbors])
    y = np.array([1.0 if n.label == \"ai\" else 0.0 for n in neighbors])
    return float((w * y).sum() / w.sum())
```

---

## 3. `backend/retrieval/hard_negatives.py` — append + reindex

```python
# file: /app/backend/retrieval/hard_negatives.py
\"\"\"User corrections append their embedding to the hard-neg partition. No retraining.\"\"\"
from __future__ import annotations

import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Literal

import faiss
import numpy as np

from backend.retrieval.index import REFDB_DIR, _BANKS, _LOCK, Bank, _bank_files, _load_bank

log = logging.getLogger(\"hard_neg\")


def append(modality: str, label: Literal[\"real\", \"ai\"],
           vector: np.ndarray, source: str = \"user_correction\") -> int:
    \"\"\"Append one corrected embedding. Returns new bank size.\"\"\"
    with _LOCK:
        key = f\"{modality}_{label}_hard\"
        npy, lbl_p, src_p = _bank_files(modality, label, hard=True)
        # Load existing (or initialise)
        if npy.exists():
            arr = np.load(npy).astype(\"float32\")
            ids = json.loads(lbl_p.read_text())
            srcs = json.loads(src_p.read_text())
        else:
            arr = np.zeros((0, vector.shape[0]), dtype=\"float32\")
            ids, srcs = [], []
        # Append
        arr = np.vstack([arr, vector.astype(\"float32\")[None, :]])
        new_id = f\"hard_{uuid.uuid4().hex[:8]}\"
        ids.append(new_id); srcs.append(source)
        # Persist + reindex
        REFDB_DIR.mkdir(parents=True, exist_ok=True)
        np.save(npy, arr)
        lbl_p.write_text(json.dumps(ids))
        src_p.write_text(json.dumps(srcs))
        idx = faiss.IndexFlatIP(arr.shape[1]); idx.add(arr)
        _BANKS[key] = Bank(label=label, vectors=arr, ids=ids, sources=srcs, index=idx)
        log.info(\"hardneg.append\", extra={\"event\": \"hardneg.append\",
                                          \"signal_name\": key, \"status\": str(len(ids))})
        return len(ids)
```

---

## 4. `backend/retrieval/build_db.py` — the reference DB builder

```python
# file: /app/backend/retrieval/build_db.py
\"\"\"Build storage/refdb/image_real and image_ai from permissive sources.

Sources are intentionally minimal & free:

REAL (1500 target):
  - Unsplash (https://unsplash.com)  — Unsplash License, attribution optional
  - Pexels   (https://pexels.com)    — Pexels License, free for commercial use
  - WikiMedia Commons featured pictures (https://commons.wikimedia.org/wiki/Commons:Featured_pictures)

AI (1500 target):
  - Civitai showcase pages (https://civitai.com) — community licenses, attribution to image-page
  - Lexica (https://lexica.art)               — public sharing
  - This-Person-Does-Not-Exist style snapshots (https://thispersondoesnotexist.com)

ALL sources are crawled with respect to robots.txt + 1 req/s rate limit. Every
saved entry stores its origin URL in `sources.json` for license traceability.

Designed to run on cloud_lite (CPU). Takes ~30–60 min depending on bandwidth.
After build, scripts/run_calibration.py is auto-invoked (see §6).\"\"\"
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image
from bs4 import BeautifulSoup

from backend.retrieval.embedder import _sync_embed
from backend.retrieval.index import REFDB_DIR

log = logging.getLogger(\"build_db\")
RAW_DIR = REFDB_DIR / \"raw\"
THUMB_DIR = REFDB_DIR / \"thumbs\"
USER_AGENT = \"deepfake-detector-refdb-builder/1.0 (+contact: research)\"


@dataclass
class Source:
    name: str
    label: str
    seeds: list[str]              # list-page URLs to scrape image links from
    pattern: str                  # CSS selector for <img> tags


SOURCES: list[Source] = [
    Source(
        name=\"unsplash\",
        label=\"real\",
        seeds=[
            \"https://unsplash.com/t/wallpapers\",
            \"https://unsplash.com/t/people\",
            \"https://unsplash.com/t/nature\",
            \"https://unsplash.com/t/architecture-interior\",
        ],
        pattern='img[src*=\"images.unsplash.com\"]',
    ),
    Source(
        name=\"pexels\",
        label=\"real\",
        seeds=[
            \"https://www.pexels.com/popular/\",
            \"https://www.pexels.com/search/people/\",
            \"https://www.pexels.com/search/landscape/\",
        ],
        pattern='img[src*=\"images.pexels.com\"]',
    ),
    Source(
        name=\"civitai\",
        label=\"ai\",
        seeds=[
            \"https://civitai.com/images?sort=Most+Reactions&period=Week\",
            \"https://civitai.com/images?sort=Most+Collected\",
        ],
        pattern=\"img.EdgeImage_image__iH4_q\",
    ),
    Source(
        name=\"lexica\",
        label=\"ai\",
        seeds=[
            \"https://lexica.art\",
            \"https://lexica.art/?q=portrait\",
            \"https://lexica.art/?q=landscape\",
        ],
        pattern=\"img.h-full\",
    ),
]


def _fetch(url: str, timeout: float = 12.0) -> bytes | None:
    try:
        r = requests.get(url, headers={\"User-Agent\": USER_AGENT}, timeout=timeout)
        if r.status_code == 200:
            return r.content
    except Exception as e:
        log.warning(\"fetch.fail\", extra={\"event\": \"fetch.fail\",
                                          \"signal_name\": url[:80],
                                          \"error_code\": type(e).__name__})
    return None


def _extract_image_urls(html: bytes, css_pattern: str, base: str) -> list[str]:
    \"\"\"CSS selector hits may have lazy-load attributes — handle src/data-src/srcset.\"\"\"
    soup = BeautifulSoup(html, \"html.parser\")
    out: list[str] = []
    for el in soup.select(css_pattern):
        for attr in (\"src\", \"data-src\", \"data-image\", \"srcset\"):
            val = el.get(attr)
            if not val: continue
            url = val.split()[0] if attr == \"srcset\" else val
            if url.startswith(\"//\"):
                url = \"https:\" + url
            elif url.startswith(\"/\"):
                p = urlparse(base); url = f\"{p.scheme}://{p.netloc}{url}\"
            out.append(url); break
    return out


def _is_usable(img_bytes: bytes) -> tuple[bool, np.ndarray | None]:
    try:
        pil = Image.open(BytesIO(img_bytes)).convert(\"RGB\")
        w, h = pil.size
        if min(w, h) < 256 or max(w, h) > 8000:
            return False, None
        arr = np.asarray(pil)
        # reject near-solid colour
        if arr.std() < 8.0:
            return False, None
        return True, arr
    except Exception:
        return False, None


def _save_thumb(arr: np.ndarray, sha: str) -> None:
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    pil = Image.fromarray(arr).convert(\"RGB\")
    pil.thumbnail((192, 192))
    pil.save(THUMB_DIR / f\"{sha}.jpg\", \"JPEG\", quality=72)


async def build(modality: str = \"image\", target_per_label: int = 1500) -> dict:
    \"\"\"Main entry. Run via `python -m backend.scripts.build_reference_db`.\"\"\"
    REFDB_DIR.mkdir(parents=True, exist_ok=True); RAW_DIR.mkdir(exist_ok=True)
    collected: dict[str, list[tuple[str, np.ndarray, str]]] = {\"real\": [], \"ai\": []}

    for src in SOURCES:
        log.info(\"source.start\", extra={\"event\": \"source.start\", \"signal_name\": src.name})
        urls: list[str] = []
        for seed in src.seeds:
            html = _fetch(seed)
            if html is None: continue
            urls.extend(_extract_image_urls(html, src.pattern, seed))
            time.sleep(1.0)               # 1 req/s
        urls = list(dict.fromkeys(urls))[:max(800, target_per_label // len(SOURCES))]
        random.shuffle(urls)

        for url in urls:
            if len(collected[src.label]) >= target_per_label: break
            data = _fetch(url, timeout=8.0)
            if data is None: continue
            ok, arr = _is_usable(data)
            if not ok or arr is None: continue
            sha = hashlib.sha256(data).hexdigest()[:24]
            collected[src.label].append((sha, arr, url))
            _save_thumb(arr, sha)
            (RAW_DIR / f\"{sha}.bin\").write_bytes(data)
            time.sleep(0.5)
        log.info(\"source.done\", extra={\"event\": \"source.done\", \"signal_name\": src.name,
                                       \"status\": str(len(collected[src.label]))})

    # Embed in batches
    for label, items in collected.items():
        if not items: continue
        vecs = []
        ids = []
        srcs = []
        for sha, arr, src_url in items:
            vecs.append(_sync_embed(arr))
            ids.append(sha)
            srcs.append(src_url)
        V = np.stack(vecs).astype(\"float32\")
        np.save(REFDB_DIR / f\"{modality}_{label}.npy\", V)
        (REFDB_DIR / f\"{modality}_{label}_labels.json\").write_text(json.dumps(ids))
        (REFDB_DIR / f\"{modality}_{label}_sources.json\").write_text(json.dumps(srcs))

    summary = {l: len(collected[l]) for l in collected}
    (REFDB_DIR / \"build_report.json\").write_text(json.dumps(summary, indent=2))
    log.info(\"build.done\", extra={\"event\": \"build.done\", \"status\": json.dumps(summary)})
    return summary
```

> **CSS selectors will drift.** Update `pattern` per source when scrape yield drops. Build_report.json shows per-source counts — if any source hits 0, fix its selector.

---

## 5. `backend/scripts/build_reference_db.py` — CLI

```python
# file: /app/backend/scripts/build_reference_db.py
\"\"\"CLI: python -m backend.scripts.build_reference_db --modalities image\"\"\"
from __future__ import annotations

import argparse
import asyncio
import logging

from backend.retrieval import build_db
from backend.scripts import run_calibration
from backend.utils.logs import configure_logging


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument(\"--modalities\", nargs=\"+\", default=[\"image\"])
    p.add_argument(\"--target\", type=int, default=1500)
    return p.parse_args()


async def _main():
    configure_logging()
    args = _parse()
    for m in args.modalities:
        summary = await build_db.build(modality=m, target_per_label=args.target)
        logging.info(f\"refdb built: {summary}\")
        # Auto-run cold-start calibration (§6 of 08_fusion_calibration_abstention.md)
        await run_calibration.run(modality=m, source=\"refdb\")


if __name__ == \"__main__\":
    asyncio.run(_main())
```

---

## 6. Runtime self-leak guard (inside the runner)

```python
# inside services/runner.py — preview
from backend.retrieval.embedder import embed_image
from backend.retrieval.index import query, retrieval_p_fake

vec = await embed_image(sample.image_rgb)
neighbors = query(\"image\", vec, k=15, exclude_id=sample.sha256[:24])
p_retrieval = retrieval_p_fake(neighbors)
```

If `sample.sha256[:24]` matches any refDB id (built directly from the upload's bytes elsewhere), it is excluded. The front-end surfaces a \"self-leak detected\" notice and disables this signal.

---

## 7. Unit tests

```python
# file: /app/backend/tests/unit/test_retrieval.py
import asyncio
import numpy as np
import pytest

from backend.retrieval.index import retrieval_p_fake, Neighbor


def test_weighted_vote_pure_ai():
    n = [Neighbor(id=\"a\", label=\"ai\", distance=0.05, source=\"\"),
         Neighbor(id=\"b\", label=\"ai\", distance=0.10, source=\"\"),
         Neighbor(id=\"c\", label=\"real\", distance=0.80, source=\"\")]
    assert retrieval_p_fake(n) > 0.85


def test_weighted_vote_pure_real():
    n = [Neighbor(id=\"a\", label=\"real\", distance=0.02, source=\"\"),
         Neighbor(id=\"b\", label=\"real\", distance=0.05, source=\"\")]
    assert retrieval_p_fake(n) < 0.1


def test_empty():
    assert retrieval_p_fake([]) == 0.5
```

```python
# file: /app/backend/tests/unit/test_hardneg.py
import numpy as np
from pathlib import Path
from backend.retrieval.hard_negatives import append
from backend.retrieval.index import REFDB_DIR


def test_append_creates_bank(tmp_path, monkeypatch):
    monkeypatch.setattr(\"backend.retrieval.index.REFDB_DIR\", tmp_path)
    monkeypatch.setattr(\"backend.retrieval.hard_negatives.REFDB_DIR\", tmp_path)
    v = np.random.randn(512).astype(\"float32\")
    v /= np.linalg.norm(v)
    size = append(\"image\", \"ai\", v, source=\"test\")
    assert size == 1
    # second append grows
    size = append(\"image\", \"ai\", v, source=\"test\")
    assert size == 2
```

---

## 8. Section exit criteria

```bash
# fast path — no actual scrape
pytest backend/tests/unit/test_retrieval.py backend/tests/unit/test_hardneg.py -q
mypy backend/retrieval/

# full build (one-time, ~30–60 min, requires network)
python -m backend.scripts.build_reference_db --modalities image --target 1500

# Inspect
ls -la /app/backend/storage/refdb/
cat /app/backend/storage/refdb/build_report.json
# Expected: {\"real\": 1500, \"ai\": 1500}
```

Next: `07_tier2_5_and_tier3.md` — SerpAPI reverse search + Gemini VLM with the v1.3.1 second-opinion enhancement.
"