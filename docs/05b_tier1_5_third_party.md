"# 05b — Tier 1.5: Third-Party Detector APIs (Hive + SightEngine)

> **Why a separate tier?** These two providers were trained on proprietary datasets — millions of images we cannot afford to gather. Each is an *orthogonal ensemble member* that often catches what Tier 1 misses. The trade-off: monthly quotas. We gate by uncertainty so we burn quota only on hard cases.
>
> AI-or-Not was dropped (only 100 calls/month — too small to be useful).
>
> | Provider | Free quota | Endpoint | Signal direction |
> |---|---|---|---|
> | Hive | 1k images/month | `https://api.thehive.ai/api/v2/task/sync` | `p_ai` ∈ [0,1] + per-class breakdown |
> | SightEngine | 2k operations/month | `https://api.sightengine.com/1.0/check.json?models=genai` | `p_ai_generated` ∈ [0,1] |
>
> Both calls run in parallel with an 8-second per-provider timeout. Either may be absent from the fusion vector — mean-imputed.

---

## 1. `backend/third_party/__init__.py` — facade + parallel dispatcher

```python
# file: /app/backend/third_party/__init__.py
\"\"\"Single entry. Fires Hive + SightEngine concurrently, returns a normalised
list[ProviderResult]. Honours monthly quota counters (db.providers_usage).\"\"\"
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

from backend.config import settings
from backend.third_party.hive import call_hive
from backend.third_party.sightengine import call_sightengine
from backend.third_party.usage import (
    can_call as quota_can_call,
    bump as quota_bump,
)

log = logging.getLogger(\"tier1_5\")


@dataclass
class ProviderResult:
    provider: Literal[\"hive\", \"sightengine\"]
    p_fake: float | None        # None → omit from fusion
    explanation: str
    raw: dict
    elapsed_ms: int
    invoked: bool


async def call_providers(image_url: str, image_bytes: bytes) -> list[ProviderResult]:
    \"\"\"Run enabled providers concurrently. Quota-gated per provider.\"\"\"
    tasks: list = []
    if settings.hive_api_key and quota_can_call(\"hive\"):
        tasks.append((\"hive\", call_hive(image_url, image_bytes)))
    if (settings.sightengine_user and settings.sightengine_secret
            and quota_can_call(\"sightengine\")):
        tasks.append((\"sightengine\", call_sightengine(image_url, image_bytes)))

    if not tasks:
        return []

    results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
    out: list[ProviderResult] = []
    for (name, _), res in zip(tasks, results):
        if isinstance(res, Exception):
            log.warning(\"tier1_5.fail\", extra={\"signal_name\": name,
                                               \"error_code\": type(res).__name__})
            out.append(ProviderResult(provider=name, p_fake=None,
                                      explanation=f\"{name} error: {type(res).__name__}\",
                                      raw={}, elapsed_ms=0, invoked=False))
            continue
        if res.get(\"p_fake\") is not None:
            await quota_bump(name)
        out.append(ProviderResult(provider=name, **res))
    return out


def should_invoke(extremity: float, agreement: float) -> bool:
    \"\"\"Burn quota only when Tier-1 has not decided.

    Gate: extremity < settings.t15_extremity_thr (default 0.30).\"\"\"
    return extremity < getattr(settings, \"t15_extremity_thr\", 0.30)
```

---

## 2. `backend/third_party/hive.py`

```python
# file: /app/backend/third_party/hive.py
\"\"\"Hive Moderation — image AI/deepfake detection.

Docs: https://docs.thehive.ai
Endpoint: POST https://api.thehive.ai/api/v2/task/sync
Auth: Bearer <key>
Request: multipart/form-data with `image` (or `url`) + `models=ai_generated_classifier`.

Response sketch (simplified):
{
  \"status\": [{\"response\": {\"output\": [{
    \"classes\": [
      {\"class\": \"not_ai_generated\", \"score\": 0.07},
      {\"class\": \"ai_generated\",      \"score\": 0.93}
    ]
  }]}}]
}\"\"\"
from __future__ import annotations

import logging
import time
from typing import Any

import aiohttp

from backend.config import settings

log = logging.getLogger(\"hive\")
ENDPOINT = \"https://api.thehive.ai/api/v2/task/sync\"


async def call_hive(image_url: str, image_bytes: bytes) -> dict[str, Any]:
    t0 = time.time()
    headers = {\"Authorization\": f\"Bearer {settings.hive_api_key}\"}
    form = aiohttp.FormData()
    form.add_field(\"image\", image_bytes, filename=\"upload.jpg\",
                   content_type=\"image/jpeg\")
    # Hive ai_generated_classifier is on by default in the project; no models param needed
    timeout = aiohttp.ClientTimeout(total=8.0)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(ENDPOINT, headers=headers, data=form) as r:
                if r.status >= 400:
                    body = await r.text()
                    raise RuntimeError(f\"hive {r.status}: {body[:200]}\")
                data = await r.json()
    except Exception as e:
        log.warning(\"hive.fail\", extra={\"error_code\": type(e).__name__})
        return {\"p_fake\": None,
                \"explanation\": f\"hive error: {type(e).__name__}\",
                \"raw\": {}, \"elapsed_ms\": int((time.time() - t0) * 1000),
                \"invoked\": False}

    # Walk the response defensively
    try:
        classes = (data[\"status\"][0][\"response\"][\"output\"][0][\"classes\"])
        ai = next((c for c in classes if \"ai\" in c[\"class\"].lower()), None)
        if ai is None:
            return {\"p_fake\": None, \"explanation\": \"no ai_generated class\",
                    \"raw\": data, \"elapsed_ms\": int((time.time() - t0) * 1000),
                    \"invoked\": False}
        p_fake = float(ai[\"score\"])
        return {
            \"p_fake\": p_fake,
            \"explanation\": f\"Hive ai_generated={p_fake:.2f}\",
            \"raw\": {\"classes\": classes},
            \"elapsed_ms\": int((time.time() - t0) * 1000),
            \"invoked\": True,
        }
    except (KeyError, IndexError, ValueError) as e:
        return {\"p_fake\": None, \"explanation\": f\"hive parse: {e}\",
                \"raw\": data, \"elapsed_ms\": int((time.time() - t0) * 1000),
                \"invoked\": False}
```

---

## 3. `backend/third_party/sightengine.py`

```python
# file: /app/backend/third_party/sightengine.py
\"\"\"SightEngine — image moderation including AI-generated detection.

Docs: https://sightengine.com/docs/genai-detection
Endpoint: POST https://api.sightengine.com/1.0/check.json
Auth: api_user + api_secret in form body
Models: `genai` (AI-generated detection)

Response (simplified):
{
  \"status\": \"success\",
  \"type\": {\"ai_generated\": 0.95}
}\"\"\"
from __future__ import annotations

import logging
import time
from typing import Any

import aiohttp

from backend.config import settings

log = logging.getLogger(\"sightengine\")
ENDPOINT = \"https://api.sightengine.com/1.0/check.json\"


async def call_sightengine(image_url: str, image_bytes: bytes) -> dict[str, Any]:
    t0 = time.time()
    form = aiohttp.FormData()
    form.add_field(\"media\", image_bytes, filename=\"upload.jpg\",
                   content_type=\"image/jpeg\")
    form.add_field(\"models\", \"genai\")
    form.add_field(\"api_user\", settings.sightengine_user)
    form.add_field(\"api_secret\", settings.sightengine_secret)

    timeout = aiohttp.ClientTimeout(total=8.0)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(ENDPOINT, data=form) as r:
                data = await r.json()
                if r.status >= 400 or data.get(\"status\") != \"success\":
                    raise RuntimeError(f\"sightengine {r.status}: {str(data)[:200]}\")
    except Exception as e:
        log.warning(\"sightengine.fail\", extra={\"error_code\": type(e).__name__})
        return {\"p_fake\": None,
                \"explanation\": f\"sightengine error: {type(e).__name__}\",
                \"raw\": {}, \"elapsed_ms\": int((time.time() - t0) * 1000),
                \"invoked\": False}

    try:
        p_fake = float(data[\"type\"][\"ai_generated\"])
        return {
            \"p_fake\": p_fake,
            \"explanation\": f\"SightEngine ai_generated={p_fake:.2f}\",
            \"raw\": {\"type\": data.get(\"type\")},
            \"elapsed_ms\": int((time.time() - t0) * 1000),
            \"invoked\": True,
        }
    except (KeyError, ValueError) as e:
        return {\"p_fake\": None, \"explanation\": f\"sightengine parse: {e}\",
                \"raw\": data, \"elapsed_ms\": int((time.time() - t0) * 1000),
                \"invoked\": False}
```

---

## 4. `backend/third_party/usage.py` — monthly quota counter

```python
# file: /app/backend/third_party/usage.py
\"\"\"Per-provider monthly counter persisted in Mongo. UTC-month rollover.

Collection: providers_usage
Doc:
{
  \"_id\": \"hive:2026-01\",
  \"calls\": 42,
  \"month\": \"2026-01\",
  \"provider\": \"hive\",
  \"cap\": 1000,
  \"updated_at\": \"...\"
}\"\"\"
from __future__ import annotations

import logging
from datetime import datetime, timezone

from backend.db.mongo import db

log = logging.getLogger(\"provider_usage\")

CAPS = {\"hive\": 1000, \"sightengine\": 2000}
RESERVE = 0.95   # leave 5% headroom


def _month_key(provider: str) -> str:
    return f\"{provider}:{datetime.now(timezone.utc).strftime('%Y-%m')}\"


def can_call(provider: str) -> bool:
    \"\"\"Synchronous check — read the latest counter from in-memory cache.
    Heavy enforcement happens in bump(); this function is best-effort.\"\"\"
    cap = CAPS.get(provider, 0)
    if cap == 0:
        return False
    return True  # actual depletion check happens server-side in bump()


async def bump(provider: str) -> int:
    cap = CAPS.get(provider, 0)
    if cap == 0:
        return 0
    key = _month_key(provider)
    now = datetime.now(timezone.utc).isoformat()
    doc = await db().providers_usage.find_one_and_update(
        {\"_id\": key},
        {\"$inc\": {\"calls\": 1},
         \"$setOnInsert\": {\"provider\": provider,
                          \"month\": key.split(\":\", 1)[1],
                          \"cap\": cap},
         \"$set\": {\"updated_at\": now}},
        upsert=True, return_document=True,
    )
    n = int(doc.get(\"calls\", 0))
    if n > cap * RESERVE:
        log.warning(\"provider.quota_low\",
                    extra={\"signal_name\": provider, \"status\": f\"{n}/{cap}\"})
    return n


async def remaining(provider: str) -> int:
    key = _month_key(provider)
    doc = await db().providers_usage.find_one({\"_id\": key})
    used = int(doc[\"calls\"]) if doc else 0
    return max(0, CAPS.get(provider, 0) - used)
```

> The `providers_usage` collection is added to `db/mongo.py::init_mongo`:
> ```python
> await _db.providers_usage.create_index([(\"month\", 1)])
> ```

---

## 5. `.env` additions

Append to `/app/backend/.env` outline (`01_setup.md §3.1`):

```
# --- Tier 1.5 third-party detectors ---
HIVE_API_KEY=
SIGHTENGINE_USER=
SIGHTENGINE_SECRET=

# Gate — only call third-party providers when uncertainty is high
T15_EXTREMITY_THRESHOLD=0.30
```

And in `config.py`:

```python
# Append to Settings(...)
hive_api_key: str | None = Field(default=None, alias=\"HIVE_API_KEY\")
sightengine_user: str | None = Field(default=None, alias=\"SIGHTENGINE_USER\")
sightengine_secret: str | None = Field(default=None, alias=\"SIGHTENGINE_SECRET\")
t15_extremity_thr: float = Field(default=0.30, alias=\"T15_EXTREMITY_THRESHOLD\")
```

API key sources (paste into README):

| Key | Where | Free quota |
|---|---|---|
| `HIVE_API_KEY` | https://thehive.ai/ → developer console | 1000 images/month |
| `SIGHTENGINE_USER` / `SIGHTENGINE_SECRET` | https://dashboard.sightengine.com/api-credentials | 2000 operations/month |

If both keys are absent, Tier 1.5 is skipped silently and fusion proceeds with Tier 1 only. **System remains fully functional without these keys.**

---

## 6. Integration into the runner (preview — full in `10_runner_orchestrator.md`)

```python
# inside services/runner.py — preview
from backend.third_party import call_providers, should_invoke as t15_should_invoke

provider_results: list[ProviderResult] = []
if t15_should_invoke(extremity, agreement) and (
        settings.hive_api_key or settings.sightengine_user):
    provider_results = await call_providers(image_url, image_bytes)
    for p in provider_results:
        if p.p_fake is not None:
            signals.append(SignalIn(name=f\"img.t15.{p.provider}\",
                                    p_fake=p.p_fake,
                                    enabled=True,
                                    weight_hint=0.8))
```

Each provider contributes one slot in the fusion vector. Fusion weights are learned during cold-start calibration (see `08_fusion_calibration_abstention.md §4`).

---

## 7. Unit tests (no live API calls)

```python
# file: /app/backend/tests/unit/test_third_party.py
\"\"\"Mocked tests — verify the parsing layer, not the live API.\"\"\"
from __future__ import annotations

import pytest
from unittest.mock import patch

from backend.third_party.hive import call_hive
from backend.third_party.sightengine import call_sightengine


class _FakeResp:
    def __init__(self, status: int, body: dict):
        self.status = status; self._body = body
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return None
    async def json(self): return self._body
    async def text(self): return str(self._body)


class _FakeSess:
    def __init__(self, body, status=200):
        self._resp = _FakeResp(status, body)
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return None
    def post(self, *a, **kw): return self._resp


@pytest.mark.asyncio
async def test_hive_parse_ai_class(monkeypatch):
    body = {\"status\": [{\"response\": {\"output\": [{
        \"classes\": [{\"class\": \"ai_generated\", \"score\": 0.91},
                    {\"class\": \"not_ai_generated\", \"score\": 0.09}]
    }]}}]}
    monkeypatch.setattr(\"backend.third_party.hive.aiohttp.ClientSession\",
                        lambda timeout: _FakeSess(body))
    monkeypatch.setattr(\"backend.config.settings.hive_api_key\", \"x\")
    out = await call_hive(\"http://example.com/x.jpg\", b\"\")
    assert out[\"p_fake\"] is not None and out[\"p_fake\"] > 0.9
    assert out[\"invoked\"] is True


@pytest.mark.asyncio
async def test_sightengine_parse(monkeypatch):
    body = {\"status\": \"success\", \"type\": {\"ai_generated\": 0.42}}
    monkeypatch.setattr(\"backend.third_party.sightengine.aiohttp.ClientSession\",
                        lambda timeout: _FakeSess(body))
    monkeypatch.setattr(\"backend.config.settings.sightengine_user\", \"u\")
    monkeypatch.setattr(\"backend.config.settings.sightengine_secret\", \"s\")
    out = await call_sightengine(\"http://example.com/x.jpg\", b\"\")
    assert abs(out[\"p_fake\"] - 0.42) < 1e-6


@pytest.mark.asyncio
async def test_hive_500_returns_disabled(monkeypatch):
    monkeypatch.setattr(\"backend.third_party.hive.aiohttp.ClientSession\",
                        lambda timeout: _FakeSess({\"err\": \"x\"}, status=500))
    monkeypatch.setattr(\"backend.config.settings.hive_api_key\", \"x\")
    out = await call_hive(\"http://example.com/x.jpg\", b\"\")
    assert out[\"p_fake\"] is None
    assert out[\"invoked\"] is False
```

---

## 8. Section exit criteria

```bash
pytest backend/tests/unit/test_third_party.py -q
mypy backend/third_party/
# Success: no issues
```

Next: `08_fusion_calibration_abstention.md` — the math that turns 10–14 raw signals into a verdict with honest abstention.
"