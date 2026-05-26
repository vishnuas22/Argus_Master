"# 12 — Scripts & Testing

> Goal: every operational script (build refDB, run calibration, tune
> thresholds, verify registry, license audit) and the full pytest /
> testing_agent_v3 plan. Each script is copy-paste with `# TODO(M*)` for
> deliberate gaps.
>
> **Reference DB scale (locked):** 5000 real + 5000 AI per modality.
> **Coverage target (P0 from AGENTS.md §5 + AGENTS_FRONTEND.md §14):** ≥80 %
> on `detectors/`, `fusion/`, `retrieval/`, `provenance/`, `reverse_search/`,
> `abstention/`, `third_party/`, `xai/narrator.py`, and the runner.

---

## 1. Script inventory

| Script | When run | Idempotent? | Time (cloud_lite) |
|---|---|---|---|
| `verify_registry.py` | CI + before every refDB build | yes | ~30 s |
| `build_reference_db.py` | once at install, again to expand | yes (hash-checked) | ~10 h for 5000+5000 |
| `run_calibration.py` | after refDB build, after every +100 labels | yes | ~2 min |
| `tune_thresholds.py` | after every +200 labels | yes | ~30 s |
| `license_audit.py` | CI | yes | ~5 s |
| `seed_e2e_job.py` | local + CI before Playwright | yes | ~20 s |

All scripts live under `/app/backend/scripts/` and are runnable as:
```bash
python -m backend.scripts.<name> [args]
```

---

## 2. `verify_registry.py` — SHA-pin sanity

```python
# file: /app/backend/scripts/verify_registry.py
\"\"\"Verify every model in detectors.registry.MODELS is reachable on HF Hub.

Exit code 0 = all reachable.
Exit code 1 = any required-profile model missing (CI fails).
Exit code 2 = optional model missing (warn, do not fail).

Run:
    python -m backend.scripts.verify_registry [--profile cloud_lite|mac_full|cuda_full]
\"\"\"
from __future__ import annotations

import argparse
import sys
from typing import Final

from huggingface_hub import HfApi, RepositoryNotFoundError

from backend.detectors.registry import MODELS, ModelSpec
from backend.utils.logs import logger

REQUIRED_PROFILES: Final = (\"cloud\", \"mac\", \"cuda\")


def check_model(api: HfApi, spec: ModelSpec, profile: str) -> tuple[str, bool, str]:
    if profile not in spec.profile_in:
        return spec.key, True, \"skipped (not in this profile)\"
    try:
        info = api.model_info(spec.repo)
        sha = info.sha or \"\"
        if spec.sha and sha != spec.sha:
            return spec.key, False, f\"SHA mismatch (pinned={spec.sha[:8]} live={sha[:8]})\"
        return spec.key, True, f\"reachable @ {sha[:8]}\"
    except RepositoryNotFoundError:
        if spec.fallback_repo:
            try:
                api.model_info(spec.fallback_repo)
                return spec.key, True, f\"primary missing, fallback {spec.fallback_repo} OK\"
            except RepositoryNotFoundError:
                return spec.key, False, \"primary AND fallback missing\"
        return spec.key, False, \"repo not found\"
    except Exception as e:  # network / auth
        return spec.key, False, f\"unexpected: {e}\"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(\"--profile\", default=\"cloud_lite\")
    args = p.parse_args()
    profile_short = args.profile.split(\"_\")[0]  # cloud_lite -> cloud
    if profile_short not in REQUIRED_PROFILES:
        logger.error(\"invalid_profile\", profile=args.profile)
        return 1

    api = HfApi()
    failures: list[tuple[str, str]] = []
    for spec in MODELS.values():
        key, ok, note = check_model(api, spec, profile_short)
        level = \"info\" if ok else \"warning\"
        getattr(logger, level)(\"registry_check\", key=key, ok=ok, note=note)
        if not ok:
            failures.append((key, note))

    if failures:
        print(\"
FAILED:\", file=sys.stderr)
        for k, n in failures:
            print(f\"  {k}: {n}\", file=sys.stderr)
        return 1
    print(f\"OK — {len(MODELS)} models verified for profile {args.profile}\")
    return 0


if __name__ == \"__main__\":
    sys.exit(main())
```

---

## 3. `build_reference_db.py` — the 5000+5000 builder

> Hybrid: curated permissive sources + supervised scraper. SHA-deduped at
> insert time. Resumable: re-running picks up where it stopped.

### 3.1 Source plan (image modality, target 5000+5000)

| Bucket | Source | License | Target rows |
|---|---|---|---|
| Real — photos | Flickr Commons (CC0/PDM) | public domain | 2000 |
| Real — photos | Wikimedia Commons (CC-BY / CC0) | permissive | 1500 |
| Real — portraits | FFHQ thumbnails (low-res 256×256 subset, CC-BY-NC research) | research-only — **excluded from default; enable via `--include-ffhq` only for non-commercial demo** | 500 (opt-in) |
| Real — illustrations | OpenClipArt + SVG-Repo PNG renders | CC0 | 1000 |
| Real — total |  |  | **≈ 5000** (4500 default + 500 opt-in) |
| AI — SD1.5/SDXL/Flux/MJ | Civitai public posts API | community CC0 / CC-BY by default | 1800 |
| AI — Lexica.art | Lexica REST API (CC0 prompts + outputs) | CC0 | 1500 |
| AI — Krea / OpenArt public | direct page scrape (rate-limited) | community licensed | 800 |
| AI — Ideogram / DALL-E pasted | reddit r/aiart top of all time | mixed — store **for retrieval only**, link source | 900 |
| AI — total |  |  | **≈ 5000** |

> **Licensing policy.** Each row records `{source, license, attribution_url}`.
> `license_audit.py` (§6) refuses to ship if any row has license `unknown` or
> `restrictive`. The refDB is **never redistributed**; it lives only on the
> deployer's disk. Only the FAISS index (anonymised embeddings + label) is
> what powers retrieval.

### 3.2 Script

```python
# file: /app/backend/scripts/build_reference_db.py
\"\"\"Build the curated reference DB.

Default targets: 5000 real + 5000 AI image samples.
Enforces: dedup by SHA256, license whitelist, per-bucket cap, resumability.

Run:
    python -m backend.scripts.build_reference_db \
        --modality image \
        --target-real 5000 --target-ai 5000 \
        --concurrency 8

Resume safely: existing `image_real.npy` / `image_ai.npy` are extended;
already-indexed SHA256s are skipped.
\"\"\"
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
import httpx

from backend.retrieval.embedder import build_embedder  # M3
from backend.retrieval.index import save_index           # M3
from backend.utils.logs import logger

# --- Sources -----------------------------------------------------------------
# Each source returns an async iterator of (image_bytes, meta_dict).

from backend.retrieval.sources import (  # implemented incrementally
    flickr_commons,
    wikimedia,
    openclipart_pngs,
    civitai_posts,
    lexica,
    krea_openart,
    reddit_ai_art,
)

# (source_fn, bucket, cap, license_tag)
REAL_SOURCES: list[tuple[Callable[..., Any], str, int, str]] = [
    (flickr_commons,    \"flickr_commons\", 2000, \"public_domain\"),
    (wikimedia,         \"wikimedia\",      1500, \"cc_by_or_cc0\"),
    (openclipart_pngs,  \"openclipart\",    1000, \"cc0\"),
]
AI_SOURCES: list[tuple[Callable[..., Any], str, int, str]] = [
    (civitai_posts,     \"civitai\",        1800, \"community\"),
    (lexica,            \"lexica\",         1500, \"cc0\"),
    (krea_openart,      \"krea_openart\",    800, \"community\"),
    (reddit_ai_art,     \"reddit_ai_art\",   900, \"mixed\"),
]

REFDB = Path(\"backend/storage/refdb\")
THUMBS = REFDB / \"thumbs\"


@dataclass
class Row:
    sha: str
    bucket: str
    license: str
    source_url: str
    generator_family: str | None
    label: str            # \"real\" | \"ai\"


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_existing_shas(label: str) -> set[str]:
    p = REFDB / f\"image_{label}_sources.json\"
    if not p.exists():
        return set()
    return {row[\"sha\"] for row in json.loads(p.read_text())}


def append_meta(label: str, rows: list[Row]) -> None:
    p = REFDB / f\"image_{label}_sources.json\"
    existing = json.loads(p.read_text()) if p.exists() else []
    existing.extend([r.__dict__ for r in rows])
    p.write_text(json.dumps(existing, indent=2))


async def fetch_bucket(
    fn: Callable[..., Any],
    bucket: str,
    cap: int,
    license_tag: str,
    label: str,
    existing: set[str],
    embedder: Any,
    client: httpx.AsyncClient,
) -> tuple[list[np.ndarray], list[Row]]:
    embs: list[np.ndarray] = []
    rows: list[Row] = []
    count = 0
    async for img_bytes, meta in fn(client, cap):
        if count >= cap:
            break
        sh = sha256(img_bytes)
        if sh in existing:
            continue
        try:
            img = Image.open_bytes := Image.open  # alias for type clarity
            with Image.open(__import__(\"io\").BytesIO(img_bytes)) as pil:
                pil = pil.convert(\"RGB\")
                vec = embedder.embed_pil(pil)             # (D,)
                # thumbnail
                pil.thumbnail((192, 192))
                (THUMBS / f\"{sh}.jpg\").write_bytes(
                    __import__(\"io\").BytesIO().getvalue() if False else b\"\",  # placeholder
                )
                pil.save(THUMBS / f\"{sh}.jpg\", \"JPEG\", quality=78)
        except Exception as e:
            logger.warning(\"decode_failed\", bucket=bucket, err=str(e))
            continue

        embs.append(vec.astype(np.float32))
        rows.append(
            Row(
                sha=sh,
                bucket=bucket,
                license=license_tag,
                source_url=meta.get(\"url\", \"\"),
                generator_family=meta.get(\"generator\"),
                label=label,
            )
        )
        existing.add(sh)
        count += 1
        if count % 50 == 0:
            logger.info(\"bucket_progress\", bucket=bucket, count=count, cap=cap)
    return embs, rows


async def run(args: argparse.Namespace) -> int:
    REFDB.mkdir(parents=True, exist_ok=True)
    THUMBS.mkdir(parents=True, exist_ok=True)

    embedder = build_embedder(\"image\")    # CLIP-B/32 frozen

    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout, headers={\"User-Agent\": \"Argus-refDB/1.0\"}) as cli:
        for label, sources, target in (
            (\"real\", REAL_SOURCES, args.target_real),
            (\"ai\",   AI_SOURCES,   args.target_ai),
        ):
            existing = load_existing_shas(label)
            collected_emb: list[np.ndarray] = []
            collected_row: list[Row] = []
            for fn, bucket, cap, lic in sources:
                if len(collected_emb) >= target:
                    break
                remaining = target - len(collected_emb)
                embs, rows = await fetch_bucket(
                    fn, bucket, min(cap, remaining), lic, label, existing, embedder, cli,
                )
                collected_emb.extend(embs)
                collected_row.extend(rows)
                logger.info(\"bucket_done\", bucket=bucket, label=label, added=len(rows))

            # Persist
            if collected_emb:
                stacked = np.vstack(collected_emb)
                npy = REFDB / f\"image_{label}.npy\"
                if npy.exists():
                    stacked = np.vstack([np.load(npy), stacked])
                np.save(npy, stacked)
                save_index(stacked, REFDB / f\"image_{label}.index\")
                append_meta(label, collected_row)
                logger.info(\"label_complete\", label=label, total=stacked.shape[0])

    # Optional: per-generator family stats
    summarise()
    return 0


def summarise() -> None:
    for label in (\"real\", \"ai\"):
        p = REFDB / f\"image_{label}_sources.json\"
        if not p.exists():
            continue
        rows = json.loads(p.read_text())
        gens: dict[str, int] = {}
        for r in rows:
            gens[r.get(\"generator_family\") or r[\"bucket\"]] = gens.get(r.get(\"generator_family\") or r[\"bucket\"], 0) + 1
        logger.info(\"refdb_summary\", label=label, total=len(rows), buckets=gens)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(\"--modality\", choices=[\"image\"], default=\"image\")  # M4 adds audio/video
    p.add_argument(\"--target-real\", type=int, default=5000)
    p.add_argument(\"--target-ai\",   type=int, default=5000)
    p.add_argument(\"--include-ffhq\", action=\"store_true\", help=\"opt-in non-commercial real bucket\")
    p.add_argument(\"--concurrency\", type=int, default=8)
    args = p.parse_args()
    return asyncio.run(run(args))


if __name__ == \"__main__\":
    sys.exit(main())
```

> **NOTE on source modules.** `backend/retrieval/sources/*.py` are
> implemented in M3 alongside the embedder. Each source module exports an
> async generator with the signature
> `async def source_name(client, cap) -> AsyncIterator[tuple[bytes, dict]]:`.
> Templates for the four busiest sources are appended in §3.3–3.6.

### 3.3 `sources/flickr_commons.py` template

```python
# file: /app/backend/retrieval/sources/flickr_commons.py
\"\"\"Flickr Commons — public-domain photo archive.

Uses Flickr API method `flickr.photos.search` with `is_commons=1`.
No API key needed for public Commons photos via the `api_key=` query param;
Flickr enforces 3600 req/hour. We page through random groups for variety.
\"\"\"
from __future__ import annotations
from typing import AsyncIterator
import random
import httpx

BASE = \"https://api.flickr.com/services/rest/\"

async def flickr_commons(
    client: httpx.AsyncClient, cap: int,
) -> AsyncIterator[tuple[bytes, dict]]:
    page = 1
    yielded = 0
    while yielded < cap:
        r = await client.get(BASE, params={
            \"method\": \"flickr.photos.getRecent\",
            \"is_commons\": 1, \"per_page\": 100, \"page\": page, \"format\": \"json\",
            \"nojsoncallback\": 1, \"extras\": \"url_l,license,owner_name\",
        })
        if r.status_code != 200:
            return
        data = r.json().get(\"photos\", {}).get(\"photo\", [])
        if not data:
            return
        random.shuffle(data)
        for ph in data:
            if yielded >= cap:
                return
            url = ph.get(\"url_l\")
            if not url:
                continue
            try:
                img = await client.get(url)
                if img.status_code != 200 or len(img.content) < 4096:
                    continue
            except Exception:
                continue
            yield img.content, {
                \"url\": url, \"license\": \"public_domain\",
                \"attribution_url\": f\"https://flickr.com/photos/{ph['owner']}/{ph['id']}\",
            }
            yielded += 1
        page += 1
```

### 3.4 `sources/civitai_posts.py` template

```python
# file: /app/backend/retrieval/sources/civitai_posts.py
\"\"\"Civitai public posts — diverse AI-generated images with generator tags.\"\"\"
from __future__ import annotations
from typing import AsyncIterator
import httpx

BASE = \"https://civitai.com/api/v1/images\"
ALLOWED_GENERATORS = {
    \"Stable Diffusion 1.5\": \"sd15\",
    \"SDXL 1.0\":            \"sdxl\",
    \"SD 3\":                \"sd3\",
    \"Flux.1\":              \"flux\",
    \"Midjourney\":          \"midjourney\",
    \"DALL-E 3\":            \"dalle3\",
}

async def civitai_posts(
    client: httpx.AsyncClient, cap: int,
) -> AsyncIterator[tuple[bytes, dict]]:
    cursor = None
    yielded = 0
    while yielded < cap:
        params = {\"limit\": 100, \"sort\": \"Most Reactions\", \"nsfw\": \"None\"}
        if cursor:
            params[\"cursor\"] = cursor
        r = await client.get(BASE, params=params)
        if r.status_code != 200:
            return
        body = r.json()
        cursor = body.get(\"metadata\", {}).get(\"nextCursor\")
        for item in body.get(\"items\", []):
            if yielded >= cap:
                return
            url = item.get(\"url\")
            base_model = (item.get(\"baseModel\") or \"\").strip()
            family = next((v for k, v in ALLOWED_GENERATORS.items() if k in base_model), \"other\")
            if family == \"other\":
                continue
            try:
                img = await client.get(url)
                if img.status_code != 200 or len(img.content) < 4096:
                    continue
            except Exception:
                continue
            yield img.content, {
                \"url\": url, \"generator\": family,
                \"attribution_url\": f\"https://civitai.com/posts/{item.get('postId','')}\",
            }
            yielded += 1
        if not cursor:
            return
```

### 3.5 `sources/lexica.py` template

```python
# file: /app/backend/retrieval/sources/lexica.py
\"\"\"Lexica.art — CC0 SD outputs with prompt metadata.\"\"\"
from __future__ import annotations
from typing import AsyncIterator
import httpx
import random

SEEDS = [\"portrait\", \"landscape\", \"product\", \"cat\", \"city\", \"abstract\", \"fantasy\",
         \"photo\", \"concept art\", \"anime\", \"still life\", \"vehicle\"]

async def lexica(
    client: httpx.AsyncClient, cap: int,
) -> AsyncIterator[tuple[bytes, dict]]:
    yielded = 0
    for seed in random.sample(SEEDS, k=len(SEEDS)):
        if yielded >= cap:
            return
        r = await client.get(\"https://lexica.art/api/v1/search\", params={\"q\": seed})
        if r.status_code != 200:
            continue
        for img_meta in r.json().get(\"images\", []):
            if yielded >= cap:
                return
            url = img_meta.get(\"srcSmall\") or img_meta.get(\"src\")
            try:
                img = await client.get(url)
                if img.status_code != 200 or len(img.content) < 2048:
                    continue
            except Exception:
                continue
            yield img.content, {
                \"url\": url, \"generator\": \"sd15\",
                \"license\": \"cc0\",
                \"attribution_url\": f\"https://lexica.art/prompt/{img_meta.get('id','')}\",
            }
            yielded += 1
```

### 3.6 Remaining sources

`wikimedia.py`, `openclipart_pngs.py`, `krea_openart.py`, `reddit_ai_art.py`
follow the same shape:

1. Async iterator over a public API or scraped index
2. Yield `(image_bytes, {\"url\":..., \"generator\": ..., \"license\": ...})`
3. Respect server rate limits (`asyncio.sleep(0.2)` per request)
4. Hard cap on requests per run via `cap`

> Detailed bodies are deferred to M3 implementation; each is ~40 lines and
> mirrors the templates above. Adding a new source is **plug-in** — append a
> tuple to `REAL_SOURCES` / `AI_SOURCES` in `build_reference_db.py`.

---

## 4. `run_calibration.py` — Platt scaling on refDB

> Fits per-signal Platt scaling on an 80/20 split of refDB. Writes
> `backend/fusion/platt.json` and `backend/calibration/report.md`.

```python
# file: /app/backend/scripts/run_calibration.py
\"\"\"Per-signal Platt scaling on the reference DB.

Reads:
  - backend/storage/refdb/image_real.npy + image_ai.npy
  - backend/storage/refdb/image_real_sources.json + image_ai_sources.json

For each Tier-1 signal we re-run its `predict(...)` on every refDB item
(features are bundled at refDB build time in `image_*_signals.json`,
generated by --emit-signals mode below).

Writes:
  - backend/fusion/platt.json   (per-signal A,B)
  - backend/calibration/report.md
\"\"\"
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from backend.utils.logs import logger

REFDB = Path(\"backend/storage/refdb\")
FUSION = Path(\"backend/fusion\")
CAL = Path(\"backend/calibration\")


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def fit_platt(s: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    \"\"\"Return (A, B) so that p = sigmoid(A*s + B).\"\"\"
    if s.std() < 1e-9 or len(np.unique(y)) < 2:
        return 1.0, 0.0
    lr = LogisticRegression(C=1.0, solver=\"lbfgs\", max_iter=200)
    lr.fit(s.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def run(modality: str) -> int:
    sig_file = REFDB / f\"{modality}_signals.json\"
    if not sig_file.exists():
        logger.error(\"signals_missing\", path=str(sig_file),
                     hint=\"Run build_reference_db.py --emit-signals first\")
        return 1
    rows: list[dict[str, Any]] = json.loads(sig_file.read_text())

    signal_names = sorted({k for r in rows for k in r[\"signals\"]})
    y = np.array([1 if r[\"label\"] == \"ai\" else 0 for r in rows], dtype=np.int8)

    platt: dict[str, dict[str, float]] = {}
    per_signal_report: list[tuple[str, float, float]] = []

    for name in signal_names:
        s = np.array([r[\"signals\"].get(name, np.nan) for r in rows], dtype=np.float64)
        mask = ~np.isnan(s)
        if mask.sum() < 100:
            logger.warning(\"signal_too_sparse\", name=name, count=int(mask.sum()))
            continue
        st, sv, yt, yv = train_test_split(
            s[mask], y[mask], test_size=0.2, random_state=42, stratify=y[mask],
        )
        A, B = fit_platt(st, yt)
        pv = 1.0 / (1.0 + np.exp(-(A * sv + B)))
        auroc = float(roc_auc_score(yv, pv))
        ece = expected_calibration_error(pv, yv)
        platt[name] = {\"A\": A, \"B\": B, \"auroc_holdout\": auroc, \"ece_holdout\": ece}
        per_signal_report.append((name, auroc, ece))

    FUSION.mkdir(parents=True, exist_ok=True)
    (FUSION / \"platt.json\").write_text(json.dumps(platt, indent=2))

    CAL.mkdir(parents=True, exist_ok=True)
    md = [\"# Calibration report
\"]
    md.append(f\"- modality: `{modality}`
- samples: {len(rows)}

\")
    md.append(\"| signal | AUROC holdout | ECE holdout |
|---|---:|---:|
\")
    for n, a, e in sorted(per_signal_report, key=lambda x: -x[1]):
        md.append(f\"| `{n}` | {a:.3f} | {e:.3f} |
\")
    (CAL / \"report.md\").write_text(\"\".join(md))

    logger.info(\"calibration_done\", n=len(per_signal_report))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(\"--modality\", default=\"image\")
    p.add_argument(\"--source\", default=\"refdb\", choices=[\"refdb\", \"mixed\"])
    p.add_argument(\"--min-labels\", type=int, default=100)
    args = p.parse_args()
    return run(args.modality)


if __name__ == \"__main__\":
    sys.exit(main())
```

---

## 5. `tune_thresholds.py` — content-type-aware

> Re-tunes per-content-type `{high, low, agree}` thresholds to hit a target
> precision (default 0.95) using accumulated labelled jobs.

```python
# file: /app/backend/scripts/tune_thresholds.py
\"\"\"Per-content-type abstention threshold tuner.

Optimises {high, low, agree} per content_type to hit
  precision @ AI >= --target-precision  AND  precision @ REAL >= --target-precision
while maximising coverage (1 - abstain rate).

Reads from MongoDB:
  - results (must have content_type, p_ai_generated, agreement, verdict, fusion)
  - labels  (consumed=false yet useful regardless)

Writes:
  - backend/abstention/thresholds.json

Algorithm: per content_type, grid search over (high, low, agree) on a
2D parabolic neighbourhood centred on the current value; pick the point
with highest coverage subject to precision constraints.
\"\"\"
from __future__ import annotations
import argparse
import asyncio
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from backend.db.mongo import db
from backend.utils.logs import logger

ABSTAIN_FILE = Path(\"backend/abstention/thresholds.json\")

DEFAULTS = {
    \"selfie_portrait\":      {\"high\": 0.78, \"low\": 0.22, \"agree\": 0.55},
    \"landscape_scene\":      {\"high\": 0.72, \"low\": 0.28, \"agree\": 0.55},
    \"object_product\":       {\"high\": 0.75, \"low\": 0.25, \"agree\": 0.55},
    \"meme_screenshot\":      {\"high\": 0.82, \"low\": 0.18, \"agree\": 0.50},
    \"document_scan\":        {\"high\": 0.80, \"low\": 0.20, \"agree\": 0.55},
    \"artwork_illustration\": {\"high\": 0.85, \"low\": 0.15, \"agree\": 0.50},
}


async def collect(min_per_type: int) -> dict[str, list[dict[str, Any]]]:
    \"\"\"Return {content_type: [{p_ai, agreement, label}]} from joined results+labels.\"\"\"
    out: dict[str, list[dict[str, Any]]] = {k: [] for k in DEFAULTS}
    cursor = db.labels.find({})
    async for lab in cursor:
        res = await db.results.find_one({\"job_id\": lab[\"job_id\"]})
        if not res or \"content_type\" not in res:
            continue
        ct = res[\"content_type\"]
        if ct not in out:
            continue
        out[ct].append({
            \"p_ai\": res[\"p_ai_generated\"],
            \"agree\": res[\"agreement\"],
            \"label\": 1 if lab[\"user_label\"] == \"ai\" else 0,
        })
    return {k: v for k, v in out.items() if len(v) >= min_per_type}


def grid_search(rows: list[dict[str, Any]], target_p: float, base: dict[str, float]) -> dict[str, float]:
    p = np.array([r[\"p_ai\"] for r in rows])
    a = np.array([r[\"agree\"] for r in rows])
    y = np.array([r[\"label\"] for r in rows])

    best = base
    best_cov = -1.0
    for high in np.arange(0.60, 0.95, 0.02):
        for low in np.arange(0.05, 0.40, 0.02):
            for agree in np.arange(0.40, 0.80, 0.04):
                pred_ai = (p >= high) & (a >= agree)
                pred_real = (p <= low) & (a >= agree)
                if pred_ai.sum() == 0 or pred_real.sum() == 0:
                    continue
                prec_ai = (y[pred_ai] == 1).mean()
                prec_real = (y[pred_real] == 0).mean()
                if prec_ai >= target_p and prec_real >= target_p:
                    cov = (pred_ai.sum() + pred_real.sum()) / len(rows)
                    if cov > best_cov:
                        best_cov = cov
                        best = {\"high\": float(high), \"low\": float(low), \"agree\": float(agree)}
    return best


async def run(args: argparse.Namespace) -> int:
    data = await collect(args.min_per_type)
    if not data:
        logger.warning(\"not_enough_labels\", min_per_type=args.min_per_type)
        return 0
    new = dict(DEFAULTS)
    for ct, rows in data.items():
        tuned = grid_search(rows, args.target_precision, DEFAULTS[ct])
        new[ct] = tuned
        logger.info(\"type_tuned\", content_type=ct, n=len(rows), thresholds=tuned)
    ABSTAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    ABSTAIN_FILE.write_text(json.dumps(new, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(\"--target-precision\", type=float, default=0.95)
    p.add_argument(\"--min-per-type\", type=int, default=20)
    p.add_argument(\"--modality\", default=\"image\")
    args = p.parse_args()
    return asyncio.run(run(args))


if __name__ == \"__main__\":
    sys.exit(main())
```

---

## 6. `license_audit.py` — refuse to ship without provenance

```python
# file: /app/backend/scripts/license_audit.py
\"\"\"Enumerate every model + refDB row license. Fail CI on `unknown`/`restrictive`.

Output: `backend/licenses.txt` (committed).
\"\"\"
from __future__ import annotations
import json
import sys
from pathlib import Path

from backend.detectors.registry import MODELS
from backend.utils.logs import logger

WHITELIST = {
    \"MIT\", \"Apache-2.0\", \"BSD-3-Clause\", \"CC0\", \"public_domain\",
    \"cc0\", \"cc_by_or_cc0\", \"community\",
}
SOFT_OK = {\"mixed\"}  # warn, do not fail


def main() -> int:
    failures: list[str] = []
    out: list[str] = [\"# License audit
\"]

    out.append(\"## Models
\")
    for spec in MODELS.values():
        ok = spec.license in WHITELIST
        out.append(f\"- `{spec.key}` ({spec.repo}) — {spec.license} {'OK' if ok else '⚠'}
\")
        if not ok and spec.license not in SOFT_OK:
            failures.append(f\"model {spec.key}: {spec.license}\")

    out.append(\"
## refDB (image)
\")
    refdb = Path(\"backend/storage/refdb\")
    for label in (\"real\", \"ai\"):
        p = refdb / f\"image_{label}_sources.json\"
        if not p.exists():
            out.append(f\"- {label}: not built yet
\")
            continue
        rows = json.loads(p.read_text())
        buckets: dict[str, dict[str, int]] = {}
        for r in rows:
            b = r[\"bucket\"]
            lic = r[\"license\"]
            buckets.setdefault(b, {})[lic] = buckets[b].get(lic, 0) + 1
            if lic not in WHITELIST and lic not in SOFT_OK:
                failures.append(f\"refDB {label}/{b}: {lic}\")
        out.append(f\"- {label}: {len(rows)} rows · \" + \" · \".join(
            f\"{b}={sum(v.values())}\" for b, v in buckets.items()) + \"
\")

    Path(\"backend/licenses.txt\").write_text(\"\".join(out))

    if failures:
        print(\"FAIL — license audit:
  \" + \"
  \".join(failures), file=sys.stderr)
        return 1
    logger.info(\"license_audit_ok\", models=len(MODELS))
    return 0


if __name__ == \"__main__\":
    sys.exit(main())
```

---

## 7. `seed_e2e_job.py` — deterministic fixture for Playwright

```python
# file: /app/backend/scripts/seed_e2e_job.py
\"\"\"Create one completed job from a bundled fixture image so Playwright has
something to navigate to. Prints the seeded job_id to stdout.

Run:
    JOB_ID=$(python -m backend.scripts.seed_e2e_job)
    export E2E_SEED_JOB_ID=$JOB_ID
\"\"\"
from __future__ import annotations
import asyncio
import sys
from pathlib import Path

from backend.services.runner import run_job
from backend.db.mongo import db
from backend.utils.logs import logger

FIXTURE = Path(\"backend/tests/fixtures/real_photo.jpg\")


async def main() -> int:
    if not FIXTURE.exists():
        logger.error(\"fixture_missing\", path=str(FIXTURE))
        return 1
    job_id = await run_job(FIXTURE.read_bytes(), filename=FIXTURE.name, mime=\"image/jpeg\")
    # Wait for done
    for _ in range(60):
        j = await db.jobs.find_one({\"_id\": job_id})
        if j and j[\"status\"] == \"done\":
            print(job_id)
            return 0
        await asyncio.sleep(1)
    return 2


if __name__ == \"__main__\":
    sys.exit(asyncio.run(main()))
```

---

## 8. Pytest configuration

```toml
# file: /app/backend/pyproject.toml (partial — testing section)
[tool.pytest.ini_options]
testpaths = [\"tests\"]
asyncio_mode = \"auto\"
addopts = [
  \"-ra\",
  \"--strict-markers\",
  \"--cov=backend\",
  \"--cov-report=term-missing\",
  \"--cov-report=xml\",
  \"--cov-fail-under=80\",
]
filterwarnings = [\"ignore::DeprecationWarning\"]

[tool.coverage.run]
branch = true
source = [\"backend\"]
omit = [
  \"backend/scripts/*\",
  \"backend/retrieval/sources/*\",   # network code, integration-tested instead
  \"backend/**/__init__.py\",
]

[tool.coverage.report]
exclude_lines = [
  \"pragma: no cover\",
  \"raise NotImplementedError\",
  \"if __name__ == .__main__.:\",
]

[tool.ruff]
line-length = 100
target-version = \"py311\"
select = [\"E\", \"F\", \"W\", \"I\", \"UP\", \"B\", \"SIM\", \"PL\"]
ignore = [\"PLR0913\"]

[tool.mypy]
python_version = \"3.11\"
strict = true
ignore_missing_imports = true
plugins = [\"pydantic.mypy\"]
```

---

## 9. Unit test plan (per module, ≥80 % coverage)

| Package | Critical tests | Fixtures |
|---|---|---|
| `detectors/image/prithiv.py` | predict() on real photo (`p<0.4`) + sdxl export (`p>0.6`); ONNX path on cloud_lite; timeout fallback returns NaN | `tests/fixtures/img/real_*.jpg`, `tests/fixtures/img/sdxl_*.png` |
| `detectors/image/frequency.py` | radial FFT shape `(64,)`; flat-noise → `p≈0.5`; AI-suspect kurtosis pattern → `p>0.6` | synthetic images |
| `detectors/image/clip0.py` | softmax > 0; zero-shot returns one of `real|ai` | one of each fixture |
| `detectors/image/meta.py` | EXIF intact → `p<0.4`; EXIF stripped → `p>0.5`; C2PA partial returns flag | real_photo, ai_image fixtures |
| `detectors/image/compression.py` | PNG fixture with `bit_depth=8`, `color_type=2`, no `tEXt` → `flag=ai_signature` `p>0.7`; camera JPEG with EXIF + APP0/APP1 → `flag=camera_signature` `p<0.3` | curated fixtures |
| `detectors/image/ocr_gibberish.py` (v1.4) | gibberish text → `p≈0.85`; clean text → `p≈0.5`; no text → `p=0.5` | synthetic via PIL.ImageDraw |
| `detectors/image/eye_forensics.py` (v1.4) | runs only when content_type=selfie_portrait; outputs in `[0,1]` | portrait fixture |
| `detectors/content_type.py` | each fixture classified into its expected of 6 buckets | one fixture per content type |
| `detectors/tta.py` | mean over 3 views == 1/3 sum; std reported | mock predictor |
| `detectors/registry.py` | LRU eviction kicks in past budget; SHA mismatch raises; fallback_repo used when primary 404 | mock HfApi |
| `provenance/c2pa_check.py` | signed fixture → `hit=true, source=\"c2pa\"`; clean photo → `hit=false` | manifest fixture |
| `provenance/sd_watermark.py` | watermarked SDXL output → `hit=true`; same image scrubbed → `hit=false` | fixture |
| `provenance/synthid_check.py` | import-guarded: returns `hit=false, source=\"none\"` when lib absent | mock import |
| `retrieval/embedder.py` | embedding shape `(512,)`; same image twice → same vector | one fixture |
| `retrieval/index.py` | k=15 returns ≤15; SHA dedup excludes self; hard-neg partition queried jointly | synthetic vectors |
| `retrieval/hard_negatives.py` | append + reindex; survives restart | tmp_path |
| `reverse_search/serpapi_client.py` | mocked SerpAPI 200 → parsed; 429 retried twice; 402 surfaced as `SERPAPI_QUOTA` | respx |
| `reverse_search/interpreter.py` | each rule in Appendix D table: `ai_gallery → 0.93`, `pre_ai_news → 0.07`, ... | hand-built dicts |
| `reverse_search/cache.py` | SHA-keyed 24h TTL; hit skips network call | freezegun |
| `vlm/judge.py` | mocked Gemini JSON → parsed; malformed → returns `None`; timeout → returns `None` | mock client |
| `fusion/calibrate.py` | identity at `A=1,B=0`; monotone otherwise | numpy |
| `fusion/fuse.py` | uniform mode == mean of present signals; LR mode uses weights; mean-imputation on missing | mock weights |
| `fusion/selector.py` | thresholds at n=0/100/500/5000 | parametrize |
| `fusion/crossmodal_bonus.py` | bonus capped at +0.10; never below 0 | numpy |
| `abstention/gate.py` | truth table for every content type × `(high, low, agree, p, a)` | parametrize |
| `xai/heatmap.py` | output is PNG bytes; shape matches input | mock backbone |
| `xai/narrator.py` | Gemini path + fallback path; both produce ≤5 sentences | mock client |
| `services/runner.py` | full pipeline runs on 1 image fixture < 30s on cloud_lite; missing Gemini key → no crash; missing SerpAPI key → no crash | integration |
| `services/device.py` | each `DETECTOR_PROFILE` value → expected profile string | monkeypatch env |
| `db/repos.py` | CRUD round-trips for jobs, results, labels, serpapi_cache | mongomock-motor |
| `routes/*` | every endpoint: happy path + 4 error envelopes | httpx ASGITransport |
| `utils/retry.py` | exponential backoff math; jitter within bounds | freezegun |
| `utils/logs.py` | redacts known secret env keys (GEMINI_API_KEY, SERPAPI_KEY, HF_TOKEN) | capsys |

---

## 10. Integration tests

```python
# file: /app/backend/tests/integration/test_pipeline.py
\"\"\"End-to-end pipeline on a fixture image, no network mocks for SerpAPI/Gemini
(they are gated off via .env feature flags during test runs).\"\"\"
from __future__ import annotations
import asyncio
import os
from pathlib import Path
from httpx import AsyncClient, ASGITransport
import pytest

from backend.server import app

FIXTURE = Path(__file__).parent.parent / \"fixtures\" / \"real_photo.jpg\"


@pytest.fixture(autouse=True)
def disable_external(monkeypatch):
    monkeypatch.setenv(\"ENABLE_VLM_TIEBREAKER\", \"false\")
    monkeypatch.setenv(\"ENABLE_REVERSE_SEARCH\", \"false\")
    monkeypatch.setenv(\"DETECTOR_PROFILE\", \"cloud_lite\")
    yield


@pytest.mark.asyncio
async def test_analyze_image_real_photo_returns_done():
    async with AsyncClient(transport=ASGITransport(app=app), base_url=\"http://test\") as cli:
        with open(FIXTURE, \"rb\") as fh:
            r = await cli.post(\"/api/analyze\", files={\"file\": (\"real.jpg\", fh, \"image/jpeg\")})
        assert r.status_code == 202, r.text
        job_id = r.json()[\"job_id\"]
        for _ in range(40):
            j = (await cli.get(f\"/api/jobs/{job_id}\")).json()
            if j[\"status\"] in (\"done\", \"failed\"):
                break
            await asyncio.sleep(1)
        assert j[\"status\"] == \"done\", j
        res = (await cli.get(f\"/api/jobs/{job_id}/result\")).json()
        assert res[\"modality\"] == \"image\"
        assert \"signals\" in res and len(res[\"signals\"]) >= 4
        assert res[\"xai\"][\"narrative\"]


@pytest.mark.asyncio
async def test_analyze_rejects_oversize(monkeypatch):
    monkeypatch.setenv(\"MAX_UPLOAD_MB\", \"0\")  # force 4xx
    async with AsyncClient(transport=ASGITransport(app=app), base_url=\"http://test\") as cli:
        with open(FIXTURE, \"rb\") as fh:
            r = await cli.post(\"/api/analyze\", files={\"file\": (\"real.jpg\", fh, \"image/jpeg\")})
        assert r.status_code == 413
        body = r.json()
        assert body[\"error\"] == \"UPLOAD_TOO_LARGE\"
        assert \"request_id\" in body


@pytest.mark.asyncio
async def test_health_envelope_shape():
    async with AsyncClient(transport=ASGITransport(app=app), base_url=\"http://test\") as cli:
        r = (await cli.get(\"/api/health\")).json()
        for k in (\"status\", \"profile\", \"db_ok\", \"calibration\", \"fusion_mode\",
                  \"ece_refdb_holdout\", \"auroc_refdb_holdout\", \"refdb_size\"):
            assert k in r, k
```

Additional integration tests (one file each):

- `test_correct_endpoint.py` — corrects a verdict; refDB hard-neg counter increments; reload survives.
- `test_provenance_short_circuit.py` — C2PA-signed fixture short-circuits; `provenance.hit=True`; abstention bypassed.
- `test_sd_watermark_fixture.py` — watermarked fixture flagged AI with `source=sd_wm`.
- `test_asset_route.py` — `/jobs/{id}/assets/heatmap.png` returns `image/png`; path-traversal `../etc/passwd` → 400.
- `test_serpapi_disabled.py` — with `ENABLE_REVERSE_SEARCH=false`, pipeline completes; signal absent from fusion vector.
- `test_vlm_disabled.py` — with `GEMINI_API_KEY` unset, pipeline completes; narrative uses fallback template.

---

## 11. testing_agent_v3 plan (M3 exit gate)

After all unit + integration tests pass, **run testing_agent_v3 once** with
the following JSON payload (filled in by the main implementer):

```json
{
  \"original_problem_statement_and_user_choices_inputs\":
    \"Multimodal AI/deepfake detection with 5-tier COEF. Target ≥95% accuracy on non-abstained. v1.4 plan, image-only first-finish.\",
  \"features_or_bugs_to_test\": [
    \"POST /api/analyze with a real-photo fixture returns 202 + job_id; status reaches done within 30s on cloud_lite\",
    \"POST /api/analyze with an AI image (SDXL fixture) eventually produces verdict AI-GENERATED\",
    \"POST /api/analyze with C2PA-signed fixture produces verdict REAL with provenance.hit=true\",
    \"POST /api/analyze with SD-watermarked fixture produces verdict AI-GENERATED with provenance.source=sd_wm\",
    \"GET /api/jobs/{id}/result includes signals[], xai.heatmap_url, xai.narrative, retrieval.neighbors, content_type\",
    \"POST /api/jobs/{id}/correct with user_label=real increments refdb_hard_size in /api/health\",
    \"GET /api/health returns status=ok, db_ok=true, ece_refdb_holdout<0.10, signals_loaded length matches profile\",
    \"Frontend UploadPage: media-upload-dropzone accepts a file and navigates to /job/{id}\",
    \"Frontend JobPage: while running, job-progress-steps visible; when done, verdict-card-container + signal-bar-chart-section + metadata-technical-table visible\",
    \"Frontend developer mode: Ctrl/Cmd+D shows developer-panel with dev-raw-signal-row-* rows and threshold-slider-high/low/agree; live verdict updates client-side\",
    \"Frontend axe-core: zero serious/critical violations on / and /about\",
    \"Mobile Safari project: dropzone tappable target ≥44px, navigation usable\",
    \"Removing GEMINI_API_KEY: pipeline completes, narrative-source shows fallback_template\",
    \"Removing SERPAPI_KEY: pipeline completes, reverse-search-badge absent\"
  ],
  \"files_of_reference\": [
    \"backend/services/runner.py (Tier-0→3 orchestrator)\",
    \"backend/routes/analyze.py\",
    \"backend/routes/jobs.py\",
    \"backend/fusion/fuse.py + selector.py + crossmodal_bonus.py\",
    \"backend/abstention/gate.py\",
    \"frontend/src/pages/JobPage.tsx\",
    \"frontend/src/components/VerdictCard.tsx\",
    \"frontend/src/components/DeveloperPanel.tsx\",
    \"frontend/src/lib/api.ts\"
  ],
  \"required_credentials\": [
    \"GEMINI_API_KEY (or EMERGENT_LLM_KEY) — for VLM tiebreaker tests; tests with key missing also required\",
    \"SERPAPI_KEY — for reverse-search tests; tests with key missing also required\",
    \"MongoDB local URL (already in .env)\",
    \"Pre-seeded job_id from seed_e2e_job.py for dev-mode E2E\"
  ],
  \"testing_type\": \"both\",
  \"agent_to_agent_context_note\": \"Reference DB must be built (5000+5000) before running. If refdb_loaded=false in /health, skip Tier-2 assertions and report this clearly in the report.\",
  \"prev_test_files_and_folder\": null,
  \"mocked_api\": {
    \"Mocked API\": \"Inform if you have mocked some APIs\",
    \"value\": {
      \"has_mocked_apis\": false,
      \"mocked_apis_list\": []
    }
  },
  \"other_misc_info\":
    \"Use REACT_APP_BACKEND_URL for all external test calls (not localhost). data-testid registry is in /app/docs/11_frontend.md §16.\"
}
```

---

## 12. CI gates (P0 cumulative)

Before declaring M3 done, all of the following must pass on a clean
machine:

1. `ruff check backend/`
2. `mypy backend/`
3. `pytest backend/tests/unit --cov-fail-under=80`
4. `pytest backend/tests/integration`
5. `python -m backend.scripts.verify_registry --profile cloud_lite`
6. `python -m backend.scripts.license_audit`
7. `cd frontend && yarn lint && yarn tsc --noEmit && yarn vitest run --coverage`
8. `cd frontend && yarn playwright test`
9. `testing_agent_v3` returns no high-severity blockers

If any of 1–9 fails: **stop, fix, regress, then advance.** This is the
AGENTS.md \"test-driven development\" mandate in operational form.

---

## 13. AGENTS.md mapping for this file

| Standard | Where honored |
|---|---|
| Testing & QA (§5) | §9 unit plan + §10 integration + §11 e2e |
| Code quality tools | §8 ruff + mypy + pytest-cov; pre-commit binds them |
| Coverage ≥80 % | §8 `--cov-fail-under=80`; §12 CI gate |
| Compliance / license | §6 license_audit.py refuses ship on `unknown` |
| Idempotency | §3 SHA-deduped resumable refDB build; §5 cache HIT skip |
| Distributed tracing surfaces | `request_id` log field in every script (`utils/logs.py`) |
| Documentation | this file IS the script doc; each script has docstring with run command |

---

End of `12_scripts_and_testing.md`.
"