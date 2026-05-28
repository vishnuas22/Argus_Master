"# 02 — Backend Skeleton: `server.py`, config, DB, utils, schemas

> Goal: a runnable `server.py` that exposes `/api/health`, `/api/profile`, `/api/modalities`, `/api/refdb/stats` (stubs OK at M0) — and the foundational modules every later file imports.

---

## 1. `backend/server.py`

```python
# file: /app/backend/server.py
\"\"\"FastAPI entrypoint. All routes mounted under /api.\"\"\"
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()  # MUST be first

from backend.config import settings
from backend.db.mongo import close_mongo, init_mongo
from backend.detectors.registry import warm_registry
from backend.routes import analyze, correct, health, history, jobs, refdb
from backend.services.device import detect_profile
from backend.utils.errors import AppError
from backend.utils.logs import configure_logging

configure_logging()
log = logging.getLogger(\"server\")

_START = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    \"\"\"Boot + shutdown hooks.\"\"\"
    log.info(\"server.start\", extra={\"profile\": detect_profile()})
    await init_mongo()
    warm_registry()  # lazy by default; this just validates the table is well-formed
    yield
    await close_mongo()
    log.info(\"server.stop\")


app = FastAPI(title=\"Deepfake Detection\", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=[\"*\"],
    allow_headers=[\"*\"],
    allow_credentials=False,
)


# ─────────────────────────── request-id middleware ───────────────────────────
@app.middleware(\"http\")
async def request_id_mw(request: Request, call_next):
    rid = request.headers.get(\"x-request-id\") or str(uuid.uuid4())
    request.state.request_id = rid
    request.state.t0 = time.time()
    try:
        response = await call_next(request)
    except AppError as e:
        return JSONResponse(
            status_code=e.status,
            content={\"error\": e.code, \"message\": e.message, \"request_id\": rid},
        )
    except Exception as exc:  # noqa: BLE001  (top-level safety net)
        log.exception(\"unhandled\", extra={\"request_id\": rid})
        return JSONResponse(
            status_code=500,
            content={\"error\": \"INTERNAL\", \"message\": str(exc), \"request_id\": rid},
        )
    response.headers[\"x-request-id\"] = rid
    response.headers[\"x-elapsed-ms\"] = str(int((time.time() - request.state.t0) * 1000))
    return response


# ───────────────────────────────── routes ────────────────────────────────────
app.include_router(analyze.router, prefix=\"/api\")
app.include_router(jobs.router,    prefix=\"/api\")
app.include_router(history.router, prefix=\"/api\")
app.include_router(refdb.router,   prefix=\"/api\")
app.include_router(correct.router, prefix=\"/api\")
app.include_router(health.router,  prefix=\"/api\")


@app.get(\"/api/uptime\")
async def _uptime() -> dict:
    return {\"uptime_s\": int(time.time() - _START)}
```

> **No `if __name__ == \"__main__\"` block.** Supervisor runs Uvicorn directly; never start your own server.

---

## 2. `backend/config.py` — Pydantic Settings

```python
# file: /app/backend/config.py
\"\"\"All runtime config lives here. Read once from env; immutable after boot.\"\"\"
from __future__ import annotations

import os
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=\".env\", extra=\"ignore\")

    # Mongo
    mongo_url: str = Field(alias=\"MONGO_URL\")
    db_name: str = Field(alias=\"DB_NAME\")
    cors_origins: list[str] = Field(default_factory=lambda: [\"*\"])

    # LLM
    gemini_api_key: str | None = Field(default=None, alias=\"GEMINI_API_KEY\")
    emergent_llm_key: str | None = Field(default=None, alias=\"EMERGENT_LLM_KEY\")
    gemini_model: str = \"gemini-3-flash-preview\"

    # Reverse search
    serpapi_key: str | None = Field(default=None, alias=\"SERPAPI_KEY\")

    # Profile / device
    detector_profile: str = Field(default=\"auto\", alias=\"DETECTOR_PROFILE\")
    torch_device: str = Field(default=\"auto\", alias=\"TORCH_DEVICE\")
    hf_home: str = Field(default=\"/app/backend/storage/models\", alias=\"HF_HOME\")
    hf_token: str | None = Field(default=None, alias=\"HF_TOKEN\")

    # Feature flags
    enable_vlm: bool = Field(default=True, alias=\"ENABLE_VLM_TIEBREAKER\")
    enable_vlm_second_opinion: bool = Field(default=True, alias=\"ENABLE_VLM_SECOND_OPINION\")
    enable_reverse_search: bool = Field(default=True, alias=\"ENABLE_REVERSE_SEARCH\")
    enable_dire_mps: bool = Field(default=False, alias=\"ENABLE_DIRE_MPS\")

    # Gates
    vlm_extremity_thr: float = Field(default=0.25, alias=\"VLM_EXTREMITY_THRESHOLD\")
    vlm_agreement_thr: float = Field(default=0.63, alias=\"VLM_AGREEMENT_THRESHOLD\")
    rev_extremity_thr: float = Field(default=0.30, alias=\"REVERSE_EXTREMITY_THRESHOLD\")
    rev_agreement_thr: float = Field(default=0.70, alias=\"REVERSE_AGREEMENT_THRESHOLD\")

    # Abstention defaults
    abstain_high: float = Field(default=0.75, alias=\"ABSTAIN_HIGH\")
    abstain_low: float = Field(default=0.25, alias=\"ABSTAIN_LOW\")
    abstain_agree: float = Field(default=0.55, alias=\"ABSTAIN_AGREE\")

    # Uploads
    max_upload_mb: int = Field(default=200, alias=\"MAX_UPLOAD_MB\")

    @property
    def llm_key(self) -> str | None:
        \"\"\"User-supplied Gemini key wins; Emergent universal key as fallback.\"\"\"
        return self.gemini_api_key or self.emergent_llm_key

    @property
    def has_llm(self) -> bool:
        return bool(self.llm_key)


@lru_cache(maxsize=1)
def _settings() -> Settings:
    # Side-effect: ensure HF cache exists.
    s = Settings()  # type: ignore[call-arg]
    os.makedirs(s.hf_home, exist_ok=True)
    os.environ[\"HF_HOME\"] = s.hf_home  # propagate to transformers/hf_hub
    return s


settings = _settings()
```

> CORS origins: in production, swap `[\"*\"]` for `[settings_origin]` parsed from `CORS_ORIGINS=https://example.com,https://other.com`. AGENTS.md §6.

---

## 3. `backend/db/mongo.py` — async Motor singleton

```python
# file: /app/backend/db/mongo.py
\"\"\"Motor client singleton + lifecycle hooks.\"\"\"
from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from backend.config import settings

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def init_mongo() -> None:
    global _client, _db
    _client = AsyncIOMotorClient(settings.mongo_url, serverSelectionTimeoutMS=2000)
    _db = _client[settings.db_name]
    # indexes — idempotent
    await _db.jobs.create_index([(\"created_at\", -1)])
    await _db.jobs.create_index([(\"status\", 1)])
    await _db.results.create_index([(\"job_id\", 1)], unique=True)
    await _db.labels.create_index([(\"consumed\", 1)])
    await _db.serpapi_cache.create_index(\"ttl_until\", expireAfterSeconds=0)


async def close_mongo() -> None:
    global _client
    if _client:
        _client.close()
        _client = None


def db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError(\"Mongo not initialized — call init_mongo() first.\")
    return _db
```

---

## 4. `backend/db/repos.py` — thin repo layer (zero ORM)

```python
# file: /app/backend/db/repos.py
\"\"\"All Mongo writes go through here. No raw collection access from routes/services.\"\"\"
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from backend.db.mongo import db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── jobs ──────────────────────────────────────────────────────────────
async def create_job(job: dict[str, Any]) -> None:
    job[\"created_at\"] = _now()
    job[\"updated_at\"] = _now()
    await db().jobs.insert_one(job)


async def update_job(job_id: str, fields: dict[str, Any]) -> None:
    fields[\"updated_at\"] = _now()
    await db().jobs.update_one({\"_id\": job_id}, {\"$set\": fields})


async def get_job(job_id: str) -> dict | None:
    return await db().jobs.find_one({\"_id\": job_id})


async def list_jobs(limit: int = 20) -> list[dict]:
    cur = db().jobs.find({}, sort=[(\"created_at\", -1)]).limit(limit)
    return [j async for j in cur]


# ─── results ───────────────────────────────────────────────────────────
async def save_result(result: dict) -> None:
    await db().results.replace_one({\"job_id\": result[\"job_id\"]}, result, upsert=True)


async def get_result(job_id: str) -> dict | None:
    return await db().results.find_one({\"job_id\": job_id})


# ─── labels (user corrections) ─────────────────────────────────────────
async def save_label(label: dict) -> None:
    label[\"submitted_at\"] = _now()
    label[\"consumed\"] = False
    await db().labels.insert_one(label)


async def unconsumed_labels() -> list[dict]:
    cur = db().labels.find({\"consumed\": False})
    return [d async for d in cur]


async def mark_label_consumed(label_id: str) -> None:
    await db().labels.update_one({\"_id\": label_id}, {\"$set\": {\"consumed\": True}})


# ─── serpapi cache (TTL-managed by index) ──────────────────────────────
async def get_serpapi_cache(sha: str) -> dict | None:
    doc = await db().serpapi_cache.find_one({\"_id\": sha})
    return doc[\"response\"] if doc else None


async def put_serpapi_cache(sha: str, response: dict, ttl_seconds: int = 86400) -> None:
    from datetime import timedelta
    ttl = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
    await db().serpapi_cache.replace_one(
        {\"_id\": sha},
        {\"_id\": sha, \"response\": response, \"fetched_at\": _now(), \"ttl_until\": ttl},
        upsert=True,
    )
```

> **No `_id` ever leaves the API boundary** (AGENTS.md). `_id` here is always the same UUID we generate ourselves — never the Mongo ObjectId. Result endpoints strip nothing because we never insert ObjectId-typed fields.

---

## 5. `backend/utils/` — logs, errors, retry, timing

### 5.1 `backend/utils/logs.py`
```python
# file: /app/backend/utils/logs.py
\"\"\"Structured JSON logging, one line per event. Redacts secrets.\"\"\"
from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone


_REDACT = re.compile(r\"(sk-[A-Za-z0-9_-]{6,}|[A-Za-z0-9]{32,})\")


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            \"ts\": datetime.now(timezone.utc).isoformat(),
            \"level\": record.levelname,
            \"logger\": record.name,
            \"msg\": record.getMessage(),
        }
        for k in (\"request_id\", \"job_id\", \"route\", \"event\", \"dur_ms\",
                  \"signal_name\", \"status\", \"error_code\", \"profile\"):
            v = getattr(record, k, None)
            if v is not None:
                base[k] = v
        if record.exc_info:
            base[\"exc\"] = self.formatException(record.exc_info)
        return _REDACT.sub(\"[REDACTED]\", json.dumps(base, ensure_ascii=False))


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    logging.getLogger(\"uvicorn.access\").setLevel(logging.WARNING)
```

### 5.2 `backend/utils/errors.py`
```python
# file: /app/backend/utils/errors.py
\"\"\"Single error envelope. Every API failure shape is identical.\"\"\"
from __future__ import annotations


class AppError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


# Common factories
def unsupported_mime(detail: str) -> AppError:
    return AppError(\"UNSUPPORTED_MIME\", detail, 415)


def too_large(mb: int) -> AppError:
    return AppError(\"UPLOAD_TOO_LARGE\", f\"Max {mb} MB\", 413)


def corrupt(detail: str) -> AppError:
    return AppError(\"CORRUPT_MEDIA\", detail, 422)


def refdb_missing() -> AppError:
    return AppError(\"REFDB_MISSING\",
                    \"Reference DB not built. Run scripts/build_reference_db.py\",
                    503)


def model_load_failed(name: str) -> AppError:
    return AppError(\"MODEL_LOAD_FAILED\", f\"{name} failed to load\", 503)
```

### 5.3 `backend/utils/retry.py`
```python
# file: /app/backend/utils/retry.py
\"\"\"Exponential-backoff retry helper for external calls.\"\"\"
from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, TypeVar

T = TypeVar(\"T\")
log = logging.getLogger(\"retry\")


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    factor: float = 2.0,
    jitter: float = 0.25,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    label: str = \"call\",
) -> T:
    last: BaseException | None = None
    for i in range(attempts):
        try:
            return await fn()
        except retry_on as e:
            last = e
            if i == attempts - 1:
                break
            delay = base_delay * (factor ** i) + random.uniform(0, jitter)
            log.warning(\"retry\", extra={\"event\": label, \"attempt\": i + 1, \"dur_ms\": int(delay * 1000)})
            await asyncio.sleep(delay)
    raise last  # type: ignore[misc]
```

### 5.4 `backend/utils/timing.py`
```python
# file: /app/backend/utils/timing.py
\"\"\"Per-stage timer context manager — accumulates `durations_ms`.\"\"\"
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class Timings:
    def __init__(self) -> None:
        self.data: dict[str, int] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.time()
        try:
            yield
        finally:
            self.data[name] = int((time.time() - t0) * 1000)
```

---

## 6. `backend/schemas/jobs.py`

```python
# file: /app/backend/schemas/jobs.py
\"\"\"Pydantic v2 schemas for jobs and results.\"\"\"
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


Modality = Literal[\"image\", \"audio\", \"video\", \"text\"]
Verdict = Literal[\"AI-GENERATED\", \"REAL\", \"INCONCLUSIVE\"]
Status = Literal[\"queued\", \"running\", \"done\", \"failed\"]
Profile = Literal[\"cloud_lite\", \"mac_full\", \"cuda_full\"]


class JobInput(BaseModel):
    filename: str
    sha256: str
    bytes: int
    mime: str
    path: str


class JobDoc(BaseModel):
    \"\"\"Stored in `jobs` collection.\"\"\"
    id: str = Field(..., alias=\"_id\")
    created_at: str
    updated_at: str
    status: Status
    stage: str = \"\"
    progress: float = 0.0
    modality: Modality
    profile: Profile
    input: JobInput
    error: str | None = None


class JobStatus(BaseModel):
    \"\"\"Response to GET /jobs/{id}.\"\"\"
    job_id: str
    modality: Modality
    status: Status
    progress: float
    stage: str
    started_at: str | None = None
    finished_at: str | None = None
```

## 7. `backend/schemas/results.py`

```python
# file: /app/backend/schemas/results.py
\"\"\"Pydantic v2 schemas for the result payload returned by /jobs/{id}/result.\"\"\"
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class SignalOut(BaseModel):
    name: str            # e.g. \"img.prithiv\"
    p_fake: float        # calibrated probability of AI
    weight: float        # fusion weight applied
    explanation: str     # short human-readable string


class RetrievalNeighbor(BaseModel):
    id: str
    label: Literal[\"real\", \"ai\"]
    distance: float
    thumb_url: str


class ReverseHit(BaseModel):
    url: str
    domain: str
    date: str | None = None
    title: str | None = None


class XAI(BaseModel):
    heatmap_url: str | None = None
    frequency_plot_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    compression_fingerprint: dict[str, Any] = Field(default_factory=dict)
    narrative: str = \"\"
    narrative_source: Literal[\"gemini\", \"fallback_template\"] = \"fallback_template\"


class Result(BaseModel):
    job_id: str
    modality: Literal[\"image\", \"audio\", \"video\"]
    profile: Literal[\"cloud_lite\", \"mac_full\", \"cuda_full\"]
    calibration: Literal[\"platt_refdb\", \"platt_blended\", \"isotonic\"]
    fusion_model: Literal[\"uniform\", \"lr_l2\", \"gbdt\"]
    content_type: str
    verdict: Literal[\"AI-GENERATED\", \"REAL\", \"INCONCLUSIVE\", \"MANIPULATED\"]
    p_ai_generated: float
    confidence: float
    agreement: float
    extremity: float
    cross_modal_bonus: float = 0.0
    abstained: bool
    provenance: dict[str, Any]
    vlm_invoked: bool
    reverse_invoked: bool
    signals: list[SignalOut]
    retrieval: dict[str, Any]
    reverse_search: dict[str, Any]
    third_party: list[dict[str, Any]] = Field(default_factory=list)
    xai: XAI
    input: dict[str, Any]
    durations_ms: dict[str, int]
    debug: dict[str, Any] | None = None

      # NOTE: `MANIPULATED` is set by the runner cross-check
    # (see 10_runner_orchestrator.md §2, `_manipulation_check`) when EXIF claims
    # a real camera but frequency + compression signatures both match a diffusion
    # signature. `third_party` carries Tier-1.5 provider results (Hive /
    # SightEngine / AI-or-Not). Both fields are emitted by the runner —
    # do not remove.
```

---

## 8. Minimal routes for M0 (`health`, `profile`/`modalities`/`refdb_stats`)

### 8.1 `backend/routes/health.py`
```python
# file: /app/backend/routes/health.py
from __future__ import annotations

import time
from fastapi import APIRouter
from backend.config import settings
from backend.db.mongo import db
from backend.services.device import detect_profile

router = APIRouter()
_START = time.time()


@router.get(\"/health\")
async def health() -> dict:
    try:
        await db().command(\"ping\")
        db_ok = True
    except Exception:
        db_ok = False

    profile = detect_profile()
    return {
        \"status\": \"ok\" if db_ok else \"degraded\",
        \"profile\": profile,
        \"signals_loaded\": [],            # populated post-M1 by registry.loaded_signals()
        \"db_ok\": db_ok,
        \"gemini_ok\": settings.has_llm,
        \"serpapi_ok\": bool(settings.serpapi_key),
        \"refdb_loaded\": False,            # post-M3
        \"refdb_size\": {},
        \"fusion_mode\": \"uniform\",
        \"calibration\": \"platt_refdb\",
        \"ece_refdb_holdout\": None,
        \"auroc_refdb_holdout\": None,
        \"n_user_labels\": 0,
        \"uptime_s\": int(time.time() - _START),
    }


@router.get(\"/profile\")
async def profile() -> dict:
    return {\"profile\": detect_profile(), \"device\": settings.torch_device}


@router.get(\"/modalities\")
async def modalities() -> dict:
    return {
        \"supported\": [\"image\"],          # post-M5: + audio, video
        \"enabled_signals\": {
            \"image\": [\"img.prithiv\", \"img.freq\", \"img.clip0\",
                      \"img.meta\", \"img.compression\", \"img.retrieval\"]
        },
    }
```

### 8.2 `backend/routes/refdb.py` (stubs at M0; filled at M3)
```python
# file: /app/backend/routes/refdb.py
from fastapi import APIRouter
from backend.retrieval.index import refdb_stats

router = APIRouter()


@router.get(\"/refdb/stats\")
async def stats() -> dict:
    return refdb_stats()
```

### 8.3 `backend/routes/analyze.py`, `jobs.py`, `history.py`, `correct.py`
These come online during M1/M3. Stubs here so `server.py` imports cleanly at M0:
```python
# file: /app/backend/routes/analyze.py
from fastapi import APIRouter
router = APIRouter()
# TODO(M1): POST /analyze — implemented in 10_runner_orchestrator.md §4
```
```python
# file: /app/backend/routes/jobs.py
from fastapi import APIRouter
router = APIRouter()
# TODO(M1): GET /jobs/{id}, /jobs/{id}/result, /jobs/{id}/assets/{name}, /jobs/{id}/report.json
```
```python
# file: /app/backend/routes/history.py
from fastapi import APIRouter
router = APIRouter()
# TODO(M1): GET /history
```
```python
# file: /app/backend/routes/correct.py
from fastapi import APIRouter
router = APIRouter()
# TODO(M3): POST /jobs/{id}/correct
```

> The full bodies of these 4 routes are in **`10_runner_orchestrator.md`**, alongside the orchestrator they call.

---

## 9. M0 exit check

```bash
sudo supervisorctl restart backend
curl -sf http://localhost:8001/api/health | python3 -m json.tool
curl -sf http://localhost:8001/api/profile
curl -sf http://localhost:8001/api/modalities
```

All three must return 200 with valid JSON. If yes → M0 complete. Proceed to `03_detector_framework.md`.
"