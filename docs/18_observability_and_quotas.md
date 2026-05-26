"# 18 — Observability & Quotas

> Goal: never get silently degraded. Every external dependency (Gemini,
> SerpAPI, Hive, SightEngine, AI-or-Not, HuggingFace) has a finite free
> quota. When one runs out, our accuracy drops — *quietly* — unless we
> surface it. This doc specifies the **counters, drift detectors, alerts,
> and UI banners** that make degradation visible within seconds.
>
> Status: **P0 for M3**. Without `/api/usage` and the UI banner, the
> system \"silently slips below 95 %\" the moment SerpAPI hits its 100/mo
> ceiling.

---

## 1. What we measure (the four pillars)

| Pillar | Metric | Why it matters |
|---|---|---|
| **Quota** | calls per provider per month vs. limit | Predicts when accuracy will drop |
| **Drift** | KL-divergence of `p_ai` distribution vs. baseline; OOD-trigger rate; agreement-median | Detects new generators in the wild |
| **Calibration** | rolling ECE on user-corrected jobs; ECE on GoldenEval-mini | Catches stale Platt / conformal |
| **Latency** | per-stage p50 / p95 from `result.durations_ms` | Detects degraded HF cache / slow third-party |

Each pillar emits **structured-log events** (one JSON line per event) AND
populates an in-memory aggregate exposed at `/api/health` and `/api/usage`.

---

## 2. Quota tracking — Mongo schema

### 2.1 Collection `provider_usage`

```json
{
  \"_id\": \"hive_2026-02\",                     // <provider>_<YYYY-MM>
  \"provider\": \"hive\",
  \"month\": \"2026-02\",
  \"calls\": 487,
  \"errors\": 3,
  \"limit_monthly\": 1000,
  \"first_call_at\": \"2026-02-01T00:14:02Z\",
  \"last_call_at\": \"2026-02-15T13:42:09Z\",
  \"exhausted_at\": null,                       // ISO ts when calls >= limit
  \"events\": [                                 // last 50 — rolling
    {\"ts\": \"...\", \"type\": \"call\", \"status\": 200, \"dur_ms\": 612},
    {\"ts\": \"...\", \"type\": \"rate_limited\", \"status\": 429, \"dur_ms\": 800},
    {\"ts\": \"...\", \"type\": \"exhausted\"}
  ]
}
```

Index: `{provider:1, month:1}` unique. TTL: 13 months (`expireAfterSeconds`
on `last_call_at`).

### 2.2 Providers tracked

| Provider | Free monthly limit | Where called from |
|---|---|---|
| `gemini`        | (effectively free; rate-limited per minute) | `narrator/gemini.py`, `vlm/judge.py` |
| `serpapi`       | 100  | `reverse_search/serpapi_client.py` |
| `hive`          | 1000 | `third_party/hive.py` |
| `sightengine`   | 2000 | `third_party/sightengine.py` |
| `aiornot`       | 100  | `third_party/aiornot.py` (best-effort tier) |
| `huggingface`   | n/a (token-gated; large free pool) | model downloads + `inference_api` fallback |

### 2.3 Recorder — `backend/services/usage.py`

```python
# file: /app/backend/services/usage.py
\"\"\"Single entry point used by every external-API caller.

Usage:
    async with track(\"hive\") as t:
        resp = await client.post(...)
        t.status = resp.status_code

Persists a single update per call, in-memory aggregate updated atomically.\"\"\"
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator

from backend.db.mongo import get_db

log = logging.getLogger(\"usage\")

LIMITS = {
    \"gemini\":      int(os.getenv(\"LIMIT_GEMINI_MONTHLY\",      \"999999\")),
    \"serpapi\":     int(os.getenv(\"LIMIT_SERPAPI_MONTHLY\",     \"100\")),
    \"hive\":        int(os.getenv(\"LIMIT_HIVE_MONTHLY\",        \"1000\")),
    \"sightengine\": int(os.getenv(\"LIMIT_SIGHTENGINE_MONTHLY\", \"2000\")),
    \"aiornot\":     int(os.getenv(\"LIMIT_AIORNOT_MONTHLY\",     \"100\")),
    \"huggingface\": int(os.getenv(\"LIMIT_HF_MONTHLY\",          \"999999\")),
}


@dataclass
class _Track:
    provider: str
    status: int = 0
    error: str | None = None
    started_at: float = field(default_factory=lambda: __import__(\"time\").time())


@asynccontextmanager
async def track(provider: str) -> AsyncIterator[_Track]:
    t = _Track(provider)
    try:
        yield t
    except Exception as exc:                       # noqa: BLE001 — boundary
        t.error = str(exc)
        t.status = -1
        raise
    finally:
        await _persist(t)


async def _persist(t: _Track) -> None:
    db = get_db()
    month = datetime.now(timezone.utc).strftime(\"%Y-%m\")
    key = f\"{t.provider}_{month}\"
    dur_ms = int((__import__(\"time\").time() - t.started_at) * 1000)
    update = {
        \"$setOnInsert\": {
            \"provider\": t.provider, \"month\": month,
            \"limit_monthly\": LIMITS.get(t.provider, 0),
            \"first_call_at\": datetime.now(timezone.utc).isoformat(),
        },
        \"$inc\": {\"calls\": 1, \"errors\": 0 if t.status == 200 else 1},
        \"$set\": {\"last_call_at\": datetime.now(timezone.utc).isoformat()},
        \"$push\": {\"events\": {\"$each\": [
            {\"ts\": datetime.now(timezone.utc).isoformat(),
             \"type\": \"call\" if t.status == 200 else (\"rate_limited\" if t.status == 429 else \"error\"),
             \"status\": t.status, \"dur_ms\": dur_ms,
             **({\"error\": t.error} if t.error else {})}
        ], \"$slice\": -50}},
    }
    await db.provider_usage.update_one({\"_id\": key}, update, upsert=True)
    log.info(\"usage_event provider=%s status=%d dur_ms=%d month=%s\",
             t.provider, t.status, dur_ms, month)
    # Cheap \"exhausted_at\" stamp on transition
    doc = await db.provider_usage.find_one({\"_id\": key})
    if doc and doc.get(\"calls\", 0) >= doc.get(\"limit_monthly\", 1) and not doc.get(\"exhausted_at\"):
        await db.provider_usage.update_one(
            {\"_id\": key, \"exhausted_at\": None},
            {\"$set\": {\"exhausted_at\": datetime.now(timezone.utc).isoformat()}})
        log.warning(\"provider_exhausted provider=%s month=%s\", t.provider, month)
```

### 2.4 Wiring requirement

Every external caller (Gemini, SerpAPI, Hive, SightEngine, AI-or-Not, HF
`InferenceApi`) **must** wrap its outbound call in `async with track(...)`.
Enforced via lint rule (`tools/lint_usage_wrap.py` — grep-based, see §9).

---

## 3. `/api/usage` endpoint

```python
# file: /app/backend/routes/usage.py
\"\"\"Read-only view of current month's quota state, for the UI banner.\"\"\"
from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter

from backend.db.mongo import get_db
from backend.services.usage import LIMITS

router = APIRouter(prefix=\"/api/usage\", tags=[\"usage\"])


@router.get(\"\")
async def get_usage() -> dict:
    db = get_db()
    month = datetime.now(timezone.utc).strftime(\"%Y-%m\")
    docs = {d[\"provider\"]: d async for d in db.provider_usage.find({\"month\": month})}
    providers = []
    for prov, lim in LIMITS.items():
        d = docs.get(prov, {})
        used = int(d.get(\"calls\", 0))
        errors = int(d.get(\"errors\", 0))
        pct = used / max(1, lim)
        providers.append({
            \"provider\": prov,
            \"limit_monthly\": lim,
            \"used\": used,
            \"errors\": errors,
            \"pct\": round(pct, 4),
            \"exhausted\": d.get(\"exhausted_at\") is not None,
            \"last_call_at\": d.get(\"last_call_at\"),
            \"severity\": (
                \"exhausted\" if pct >= 1.0
                else \"critical\" if pct >= 0.95
                else \"warning\" if pct >= 0.80
                else \"ok\"
            ),
        })
    return {\"month\": month, \"providers\": providers}
```

### 3.1 Response example

```json
{
  \"month\": \"2026-02\",
  \"providers\": [
    {\"provider\":\"gemini\",      \"limit_monthly\":999999,\"used\":421, \"pct\":0.0004,\"severity\":\"ok\"},
    {\"provider\":\"serpapi\",     \"limit_monthly\":100,   \"used\":83,  \"pct\":0.83,  \"severity\":\"warning\"},
    {\"provider\":\"hive\",        \"limit_monthly\":1000,  \"used\":612, \"pct\":0.61,  \"severity\":\"ok\"},
    {\"provider\":\"sightengine\", \"limit_monthly\":2000,  \"used\":154, \"pct\":0.077, \"severity\":\"ok\"},
    {\"provider\":\"aiornot\",     \"limit_monthly\":100,   \"used\":99,  \"pct\":0.99,  \"severity\":\"critical\"}
  ]
}
```

### 3.2 Severity thresholds

| pct of monthly limit | severity | UI behaviour |
|---|---|---|
| < 0.80 | `ok`        | no banner |
| 0.80 – 0.94 | `warning`   | yellow banner: \"{provider}: {used}/{limit} this month\" |
| 0.95 – 0.99 | `critical`  | orange banner + freezes that signal preemptively |
| ≥ 1.00       | `exhausted` | red banner + signal dropped from fusion vector |

`critical` proactively freezes a signal at 95 % to leave room for retries
without slipping over the cap silently.

---

## 4. UI banner (frontend)

```tsx
// file: /app/frontend/src/components/QuotaBanner.tsx
// Polled every 60 s; shown only when any provider is non-ok.
// data-testid: quota-banner | quota-banner-row-<provider>
```

Behaviour:
- Renders inside `App.tsx` shell, above the main router.
- Polls `/api/usage` every 60 s via TanStack Query.
- Renders zero DOM when all providers are `ok` (no layout shift).
- Each row links to the relevant provider dashboard for top-up.
- Dismissible per-session via localStorage; reappears next page-load if
  severity unchanged.

---

## 5. `/api/metrics` (Prometheus-style, optional)

For users running their own monitoring:

```python
# file: /app/backend/routes/metrics.py
from fastapi import APIRouter, Response

router = APIRouter(prefix=\"/api/metrics\", tags=[\"metrics\"])


@router.get(\"\", include_in_schema=False)
async def metrics() -> Response:
    body = _render_prometheus()        # see §5.1
    return Response(content=body, media_type=\"text/plain; version=0.0.4\")
```

### 5.1 Exposed series

```
# HELP detector_calls_total Detector invocations
# TYPE detector_calls_total counter
detector_calls_total{detector=\"img.prithiv\"} 4218

# HELP detector_p_ai_quantile Quantiles of p_ai per detector
# TYPE detector_p_ai_quantile gauge
detector_p_ai_quantile{detector=\"img.prithiv\",q=\"0.5\"} 0.418

# HELP provider_calls_total
provider_calls_total{provider=\"hive\"} 612
provider_calls_total{provider=\"serpapi\"} 83

# HELP fusion_verdict_total
fusion_verdict_total{verdict=\"AI-GENERATED\"} 1841
fusion_verdict_total{verdict=\"REAL\"} 1612
fusion_verdict_total{verdict=\"INCONCLUSIVE\"} 765

# HELP pipeline_duration_seconds_bucket
pipeline_duration_seconds_bucket{le=\"5\"} 1023
pipeline_duration_seconds_bucket{le=\"10\"} 3014
pipeline_duration_seconds_bucket{le=\"30\"} 4187
pipeline_duration_seconds_bucket{le=\"+Inf\"} 4218

# HELP eval_macro_auroc Last full GoldenEval result
# TYPE eval_macro_auroc gauge
eval_macro_auroc 0.892

# HELP eval_ece Last full GoldenEval ECE
eval_ece 0.043
```

Disabled by default; flip via `ENABLE_PROM_METRICS=true`. Designed for
`prometheus-fastapi-instrumentator` if user wants the full toolkit later.

---

## 6. Drift detection

### 6.1 In-process rolling buffer

```python
# file: /app/backend/services/drift.py
\"\"\"Lightweight in-memory drift monitor — bounded ring buffer of last 1000 jobs.

Computes on every 50th job:
- KL-div(p_ai_distribution) vs. baseline P_baseline = uniform(0,1)
- agreement-median
- ood_trigger_rate
- ece_on_user_labels (when corrections exist)

When any metric crosses threshold, emit structured-log `event=\"drift_alert\"`.\"\"\"
from __future__ import annotations

import logging
import math
from collections import deque
from typing import Final

import numpy as np

log = logging.getLogger(\"drift\")
BUF: Final[deque[dict]] = deque(maxlen=1000)
EVERY_N: Final = 50


def record(p_ai: float, agreement: float, ood_triggered: bool,
           user_label: int | None = None) -> None:
    BUF.append({\"p\": p_ai, \"a\": agreement, \"ood\": ood_triggered, \"y\": user_label})
    if len(BUF) % EVERY_N == 0: _evaluate()


def _evaluate() -> None:
    p = np.array([x[\"p\"] for x in BUF])
    a = np.array([x[\"a\"] for x in BUF])
    ood = np.mean([x[\"ood\"] for x in BUF])
    kl = _kl_uniform(p)
    alerts = []
    if kl > 0.05:               alerts.append((\"p_ai_kl_baseline\", kl))
    if a.size and np.median(a) < 0.45: alerts.append((\"agreement_median_low\", float(np.median(a))))
    if ood > 0.12:              alerts.append((\"ood_rate_high\", float(ood)))

    labeled = [x for x in BUF if x[\"y\"] is not None]
    if len(labeled) >= 50:
        ece_user = _ece(np.array([x[\"p\"] for x in labeled]),
                        np.array([x[\"y\"] for x in labeled]))
        if ece_user > 0.10: alerts.append((\"ece_user_corrections_high\", ece_user))

    for name, val in alerts:
        log.warning(\"drift_alert metric=%s value=%.4f n=%d\", name, val, len(BUF))


def _kl_uniform(p: np.ndarray, bins: int = 20) -> float:
    h, _ = np.histogram(p, bins=bins, range=(0, 1), density=True)
    h = h / h.sum() + 1e-9
    q = np.ones(bins) / bins
    return float(np.sum(h * np.log(h / q)))


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0; n = len(p)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if not m.any(): continue
        e += abs(p[m].mean() - y[m].mean()) * m.sum() / n
    return float(e)
```

### 6.2 Surfaced in `/api/health`

Adds to existing health payload (see doc 02 §8.1):

```json
{
  \"drift\": {
    \"window_size\": 1000,
    \"p_ai_kl_baseline\": 0.027,
    \"agreement_median\": 0.62,
    \"ood_rate\": 0.094,
    \"ece_on_corrections\": 0.061,
    \"alerts\": []
  }
}
```

---

## 7. `/api/health.eval_mini` — canary

Every 6 hours (or on demand via `?force=1`), runs a 50-sample stratified
mini-eval against the on-disk GoldenEval set. Cheap, gives an early
warning when the full eval would fail.

```json
{
  \"eval_mini\": {
    \"ran_at\": \"2026-02-15T12:00:00Z\",
    \"n\": 50,
    \"auroc\": 0.873,
    \"abstain\": 0.22,
    \"passed_gate\": true
  }
}
```

Implementation: `services/eval_canary.py` schedules via `asyncio.create_task`
on startup; gated by `ENABLE_EVAL_CANARY=true`.

---

## 8. Alert routing (optional)

| Sink | When | How |
|---|---|---|
| Structured log | always | `log.warning(\"drift_alert ...\")` |
| `/api/health.alerts[]` | always | in-memory ring, last 50 |
| Webhook (Slack/Discord/etc.) | severity >= warning | `ALERT_WEBHOOK_URL` POST JSON |
| Email | optional | not in M3 |

Webhook payload (when configured):

```json
{
  \"ts\": \"...\", \"level\": \"warning\",
  \"event\": \"provider_exhausted\", \"provider\": \"serpapi\",
  \"month\": \"2026-02\", \"used\": 100, \"limit\": 100,
  \"host\": \"argus-prod-1\"
}
```

---

## 9. Lint rule — enforce `track()` wrapping

Tiny grep-based check; runs in CI:

```python
# file: /app/tools/lint_usage_wrap.py
\"\"\"Refuse to merge code that calls a known third-party endpoint without
wrapping it in `usage.track(...)`. Cheap regex linter.\"\"\"
from __future__ import annotations

import re
import sys
from pathlib import Path

URLS = {
    \"hive\":        r\"api\.thehive\.ai\",
    \"sightengine\": r\"api\.sightengine\.com\",
    \"aiornot\":     r\"api\.aiornot\.com\",
    \"serpapi\":     r\"serpapi\.com\",
    \"gemini\":      r\"generativelanguage\.googleapis\.com\",
}

BACKEND = Path(\"/app/backend\")


def main() -> int:
    bad: list[str] = []
    for f in BACKEND.rglob(\"*.py\"):
        txt = f.read_text()
        for prov, pat in URLS.items():
            if re.search(pat, txt) and f\"track(\\"{prov}\\")\" not in txt:
                if f.name not in {\"usage.py\", \"lint_usage_wrap.py\"}:
                    bad.append(f\"{f}: hits {prov} without usage.track()\")
    if bad:
        print(\"
\".join(bad)); return 1
    return 0


if __name__ == \"__main__\": sys.exit(main())
```

Hook in `pytest` (`conftest.py` runs it as a session fixture in CI mode).

---

## 10. Folder layout

```
backend/
├── routes/
│   ├── usage.py                            # NEW v1.5
│   └── metrics.py                          # NEW v1.5 (opt-in)
├── services/
│   ├── usage.py                            # NEW v1.5 — track() context
│   ├── drift.py                            # NEW v1.5
│   └── eval_canary.py                      # NEW v1.5
└── ... (existing)

frontend/src/components/
└── QuotaBanner.tsx                         # NEW v1.5
```

```
tools/
└── lint_usage_wrap.py                      # NEW v1.5 (CI gate)
```

---

## 11. ENV additions

Append to `/app/backend/.env` (no defaults; ship empty in `.env.example`):

```
# --- Usage limits (override per deployment) ---
LIMIT_GEMINI_MONTHLY=999999
LIMIT_SERPAPI_MONTHLY=100
LIMIT_HIVE_MONTHLY=1000
LIMIT_SIGHTENGINE_MONTHLY=2000
LIMIT_AIORNOT_MONTHLY=100
LIMIT_HF_MONTHLY=999999

# --- Metrics & alerts (opt-in) ---
ENABLE_PROM_METRICS=false
ENABLE_EVAL_CANARY=true
ALERT_WEBHOOK_URL=
```

---

## 12. AGENTS.md mapping

| AGENTS.md principle | Where addressed |
|---|---|
| §7 Observability & Monitoring | Drift detector + Prom metrics + health canary |
| §7 Distributed tracing | `request_id` propagated; per-stage `dur_ms` in result |
| §7 Alert management | Severity tiers + webhook sink |
| §7 SLA/SLO | Gate thresholds in `eval_mini` are SLO baseline |
| §8 Caching | Provider responses cached (SerpAPI); quota counter cached in-process |
| §11 Graceful degradation | `critical`/`exhausted` preemptively freezes signal — no surprise failure |
| §14 AI/ML — Token usage optimization | Hive/SightEngine gated by `extremity`; usage logged per-provider |

---

## 13. Section exit criteria

```bash
# Inspect quota state
curl localhost:8001/api/usage | jq .

# Force a drift evaluation (in tests)
pytest backend/tests/unit/test_drift.py -k \"test_drift_eval_emits_alerts\"

# Lint usage-wrap rule
python tools/lint_usage_wrap.py

# Prometheus metrics (when ENABLE_PROM_METRICS=true)
curl localhost:8001/api/metrics | head -20

# Banner renders in UI when severity != ok
# (frontend test in 11_frontend.md)
```

All four pass → §18 obligations met for M3 close.
"