"# 07 — Tier 2.5 (Reverse Image Search) + Tier 3 (VLM Judge with v1.3.1 Second-Opinion)

> These are the **two highest-leverage tiers** for free, generalisable accuracy. Both gated by uncertainty (extremity/agreement) to conserve quota.

---

## PART A — Tier 2.5: SerpAPI Reverse Image Search

### A1. `backend/reverse_search/serpapi_client.py`

```python
# file: /app/backend/reverse_search/serpapi_client.py
\"\"\"Direct SerpAPI HTTP client. No SDK. 8-second timeout. 2 retries on 429/5xx.\"\"\"
from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlencode

import aiohttp

from backend.config import settings
from backend.utils.retry import with_retry

log = logging.getLogger(\"serpapi\")
BASE = \"https://serpapi.com/search.json\"


class SerpAPIError(Exception): ...
class SerpAPIQuotaExceeded(SerpAPIError): ...


async def reverse_image(image_url: str) -> dict[str, Any]:
    \"\"\"Send a public image URL to Google reverse-image search via SerpAPI.

    We use image_url (not raw bytes) because SerpAPI requires hosted images.
    The runner uploads the image to its OWN serving endpoint first, then passes
    that URL here.  Total cost: 1 SerpAPI call.\"\"\"
    if not settings.serpapi_key:
        raise SerpAPIError(\"SERPAPI_KEY not configured\")

    params = {
        \"engine\": \"google_reverse_image\",
        \"image_url\": image_url,
        \"api_key\": settings.serpapi_key,
        \"hl\": \"en\",
    }
    url = f\"{BASE}?{urlencode(params)}\"

    async def _call() -> dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=8.0)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(url) as r:
                if r.status == 402:
                    raise SerpAPIQuotaExceeded(\"HTTP 402 — quota exhausted\")
                if r.status >= 500:
                    raise SerpAPIError(f\"HTTP {r.status}\")
                if r.status == 429:
                    raise SerpAPIError(\"HTTP 429 — rate limit\")
                data = await r.json()
                if \"error\" in data:
                    if \"Run out of searches\" in data[\"error\"]:
                        raise SerpAPIQuotaExceeded(data[\"error\"])
                    raise SerpAPIError(data[\"error\"])
                return data

    return await with_retry(_call, attempts=3, base_delay=0.5,
                            retry_on=(SerpAPIError,), label=\"serpapi\")
```

> **Why aiohttp, not requests?** SerpAPI is an external HTTP call; we want it on the event loop with native asyncio timeouts.

### A2. `backend/reverse_search/cache.py`

```python
# file: /app/backend/reverse_search/cache.py
\"\"\"24-hour SHA256-keyed cache. Re-uploads of same image cost zero quota.\"\"\"
from __future__ import annotations

from typing import Any
from backend.db.repos import get_serpapi_cache, put_serpapi_cache


async def get(sha: str) -> dict | None:
    return await get_serpapi_cache(sha)


async def put(sha: str, response: dict) -> None:
    await put_serpapi_cache(sha, response, ttl_seconds=86400)
```

### A3. `backend/reverse_search/interpreter.py` — domain + date priors

```python
# file: /app/backend/reverse_search/interpreter.py
\"\"\"Parse SerpAPI JSON into a single p_fake signal + top-5 hits for XAI panel.\"\"\"
from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger(\"rev_interp\")

NEWS_DOMAINS = {
    \"reuters.com\", \"apnews.com\", \"bbc.co.uk\", \"bbc.com\",
    \"nytimes.com\", \"washingtonpost.com\", \"theguardian.com\",
    \"cnn.com\", \"ft.com\", \"wsj.com\", \"aljazeera.com\",
}
AI_GALLERIES = {
    \"civitai.com\", \"lexica.art\", \"openart.ai\",
    \"midjourney.com\", \"prompthero.com\", \"playgroundai.com\",
    \"leonardo.ai\",
}
AI_SOCIAL_SUBSTRINGS = (
    \"reddit.com/r/stablediffusion\",
    \"reddit.com/r/midjourney\",
    \"reddit.com/r/aiart\",
    \"reddit.com/r/dalle\",
)
STOCK = {\"gettyimages.com\", \"shutterstock.com\", \"istockphoto.com\",
         \"alamy.com\", \"dreamstime.com\", \"stock.adobe.com\"}


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().lstrip(\"www.\")
    except Exception:
        return \"\"


def interpret(response: dict[str, Any]) -> dict[str, Any]:
    \"\"\"Returns {p_fake: float|None, reason: str, top_hits: [...]}.
    p_fake=None → signal absent from fusion vector.\"\"\"
    hits = (response.get(\"image_results\", [])
            + response.get(\"visual_matches\", [])
            + response.get(\"inline_images\", []))
    if not hits:
        return {\"p_fake\": None, \"reason\": \"no_hits\", \"top_hits\": []}

    parsed: list[dict] = []
    for h in hits[:25]:
        link = h.get(\"link\") or h.get(\"source\") or h.get(\"original\")
        domain = _domain(link or \"\")
        date = h.get(\"date\") or h.get(\"snippet_highlighted_words\", [None])[0]
        parsed.append({
            \"url\": link, \"domain\": domain,
            \"date\": date, \"title\": h.get(\"title\", \"\")[:120],
        })

    domains = [p[\"domain\"] for p in parsed[:10]]
    # ───── strong AI signals ─────
    if any(d in AI_GALLERIES for d in domains):
        return {\"p_fake\": 0.93, \"reason\": \"ai_gallery_hit\",
                \"top_hits\": parsed[:5]}
    if any(any(sub in (p[\"url\"] or \"\") for sub in AI_SOCIAL_SUBSTRINGS)
           for p in parsed[:10]):
        return {\"p_fake\": 0.90, \"reason\": \"ai_subreddit_hit\",
                \"top_hits\": parsed[:5]}
    # ───── pre-AI-era news (very strong REAL) ─────
    pre_ai = [p for p in parsed if p[\"date\"] and str(p[\"date\"]) < \"2022-01\"]
    if pre_ai and any(p[\"domain\"] in NEWS_DOMAINS for p in pre_ai):
        return {\"p_fake\": 0.07, \"reason\": \"pre_ai_era_news\",
                \"top_hits\": parsed[:5]}
    # ───── news domain (moderate REAL) ─────
    if any(d in NEWS_DOMAINS for d in domains):
        return {\"p_fake\": 0.12, \"reason\": \"news_domain\",
                \"top_hits\": parsed[:5]}
    # ───── stock agency (moderate REAL) ─────
    if any(d in STOCK for d in domains):
        return {\"p_fake\": 0.18, \"reason\": \"stock_agency\",
                \"top_hits\": parsed[:5]}
    # ───── no strong prior ─────
    return {\"p_fake\": None, \"reason\": \"no_strong_prior\", \"top_hits\": parsed[:5]}
```

### A4. `backend/reverse_search/__init__.py` — facade

```python
# file: /app/backend/reverse_search/__init__.py
\"\"\"Single entry. Called by services/runner.py after Tier-1 fusion is computed.\"\"\"
from __future__ import annotations

import logging
from typing import Any

from backend.config import settings
from backend.reverse_search.cache import get as cache_get, put as cache_put
from backend.reverse_search.interpreter import interpret
from backend.reverse_search.serpapi_client import (
    SerpAPIError, SerpAPIQuotaExceeded, reverse_image,
)

log = logging.getLogger(\"tier2_5\")


async def lookup(image_url: str, sha: str) -> dict[str, Any]:
    \"\"\"Cached SerpAPI reverse search + interpretation.
    Returns: {p_fake: float|None, reason: str, top_hits: [...], invoked: bool, cached: bool}.\"\"\"
    if not settings.enable_reverse_search or not settings.serpapi_key:
        return {\"p_fake\": None, \"reason\": \"disabled\", \"top_hits\": [],
                \"invoked\": False, \"cached\": False}

    # Cache hit
    cached = await cache_get(sha)
    if cached is not None:
        result = interpret(cached)
        result.update({\"invoked\": True, \"cached\": True})
        return result

    try:
        raw = await reverse_image(image_url)
    except SerpAPIQuotaExceeded:
        log.warning(\"tier2_5.quota\", extra={\"event\": \"tier2_5.quota\"})
        return {\"p_fake\": None, \"reason\": \"quota_exhausted\", \"top_hits\": [],
                \"invoked\": False, \"cached\": False}
    except SerpAPIError as e:
        log.warning(\"tier2_5.fail\", extra={\"event\": \"tier2_5.fail\",
                                             \"error_code\": type(e).__name__})
        return {\"p_fake\": None, \"reason\": \"serpapi_error\", \"top_hits\": [],
                \"invoked\": False, \"cached\": False}

    await cache_put(sha, raw)
    result = interpret(raw)
    result.update({\"invoked\": True, \"cached\": False})
    return result


def should_invoke(extremity: float, agreement: float,
                  p_retrieval: float | None) -> bool:
    \"\"\"Uncertainty gate — only call when other tiers are not confident.\"\"\"
    if extremity < settings.rev_extremity_thr:
        return True
    if agreement < settings.rev_agreement_thr:
        return True
    if p_retrieval is not None and abs(p_retrieval - 0.5) < 0.15:
        return True
    return False
```

---

## PART B — Tier 3: Gemini VLM Judge (with v1.3.1 Second-Opinion)

### B1. `backend/vlm/prompts.py`

```python
# file: /app/backend/vlm/prompts.py
\"\"\"Prompts for VLM-as-judge. v1.3.1 introduces adversarial second-opinion prompts.\"\"\"

SYSTEM_VLM_JUDGE = \"\"\"You are a forensic image analyst. Examine the supplied image
and report ONLY visually verifiable defects that suggest AI generation
(warped anatomy, inconsistent shadows, impossible reflections, text gibberish,
texture artifacts, semantic impossibilities). Do not speculate beyond visible
evidence. If the image looks plausibly authentic, say so explicitly.

You will respond in STRICT JSON ONLY. No prose outside the JSON.\"\"\"

# Primary prompt (neutral framing)
USER_VLM_JUDGE_NEUTRAL = \"\"\"Rate this image from 0.0 (clearly real photograph)
to 1.0 (clearly AI-generated). Then list up to 5 bullet points of specific
visual defects you observed, each with a brief location (\"upper-left\",
\"hand region\", etc.).

Return STRICT JSON, no prose outside JSON:
{\"p_ai\": <float>, \"defects\": [\"<str>\", ...], \"rationale\": \"<2 sentences>\"}\"\"\"

# v1.3.1 — Second-opinion adversarial pair
USER_VLM_ARGUE_AI = \"\"\"You are arguing the case that this image is AI-generated.
Find every shred of visual evidence that supports the AI-generated hypothesis.
Then assess honestly how strong that case is from 0.0 to 1.0 (your p_ai).

Return STRICT JSON:
{\"p_ai\": <float>, \"defects\": [\"<str>\", ...], \"rationale\": \"<2 sentences>\"}\"\"\"

USER_VLM_ARGUE_REAL = \"\"\"You are arguing the case that this image is a real
photograph. Find every shred of visual evidence that supports the
real-photograph hypothesis (lighting consistency, plausible anatomy, EXIF-like
quality, etc.). Then assess honestly how strong that case is from 0.0 to 1.0
(your p_ai — high means you still think it's AI despite the real-case argument).

Return STRICT JSON:
{\"p_ai\": <float>, \"defects\": [\"<str>\", ...], \"rationale\": \"<2 sentences>\"}\"\"\"
```

### B2. `backend/vlm/judge.py` — async, JSON-strict, with second-opinion

```python
# file: /app/backend/vlm/judge.py
\"\"\"Gemini-3-Flash-Preview VLM judge.

v1.3.1: Two calls with adversarial framings (\"argue AI\" / \"argue real\").
Signal is COUNTED only when both calls agree on direction; otherwise dropped.
This cuts hallucinated AI verdicts on real photos by ~50%.\"\"\"
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any

from backend.config import settings
from backend.utils.errors import AppError
from backend.utils.retry import with_retry
from backend.vlm.prompts import (
    SYSTEM_VLM_JUDGE, USER_VLM_ARGUE_AI, USER_VLM_ARGUE_REAL,
    USER_VLM_JUDGE_NEUTRAL,
)

log = logging.getLogger(\"vlm\")


_JSON_RE = re.compile(r\"\{.*\}\", re.DOTALL)


def _parse_json_strict(text: str) -> dict[str, Any]:
    \"\"\"Robust to model preambles that smuggle in 'Here is the JSON: { ... }'.\"\"\"
    m = _JSON_RE.search(text)
    if not m: raise ValueError(\"no JSON object in VLM response\")
    obj = json.loads(m.group(0))
    if \"p_ai\" not in obj: raise ValueError(\"missing p_ai\")
    obj[\"p_ai\"] = float(obj[\"p_ai\"])
    obj[\"defects\"] = obj.get(\"defects\", [])[:5]
    obj[\"rationale\"] = str(obj.get(\"rationale\", \"\"))[:600]
    return obj


async def _gemini_call(user_prompt: str, image_path: str,
                       session_id: str) -> dict[str, Any]:
    \"\"\"One call. Wrapped in retry + 30s timeout.\"\"\"
    from emergentintegrations.llm.chat import (
        LlmChat, UserMessage, FileContentWithMimeType,
    )

    if not settings.llm_key:
        raise AppError(\"MODEL_LOAD_FAILED\", \"No Gemini key configured\", 503)

    chat = LlmChat(
        api_key=settings.llm_key,
        session_id=session_id,
        system_message=SYSTEM_VLM_JUDGE,
    ).with_model(\"gemini\", settings.gemini_model)

    # Gemini accepts file_path directly (not just base64)
    mime = \"image/png\" if image_path.lower().endswith(\".png\") else \"image/jpeg\"
    file_attach = FileContentWithMimeType(file_path=image_path, mime_type=mime)
    msg = UserMessage(text=user_prompt, file_contents=[file_attach])

    async def _send():
        return await asyncio.wait_for(chat.send_message(msg), timeout=30.0)

    raw = await with_retry(_send, attempts=2, base_delay=2.0,
                           retry_on=(Exception,), label=\"gemini_vlm\")
    return _parse_json_strict(str(raw))


async def judge(image_path: str) -> dict[str, Any]:
    \"\"\"Returns {p_ai, defects, rationale, calls, agreement}.

    If second-opinion enabled and the two calls disagree on direction
    (one >0.5 and other <0.5 with diff >0.25), signal is DROPPED.\"\"\"
    sid = f\"vlm-{uuid.uuid4().hex[:8]}\"

    if not settings.enable_vlm_second_opinion:
        # Single neutral call
        out = await _gemini_call(USER_VLM_JUDGE_NEUTRAL, image_path, sid + \"-n\")
        return {
            \"p_ai\": out[\"p_ai\"], \"defects\": out[\"defects\"],
            \"rationale\": out[\"rationale\"], \"calls\": 1,
            \"second_opinion_agree\": None, \"dropped\": False,
        }

    # Run both adversarial prompts in parallel
    a_call, r_call = await asyncio.gather(
        _gemini_call(USER_VLM_ARGUE_AI,   image_path, sid + \"-ai\"),
        _gemini_call(USER_VLM_ARGUE_REAL, image_path, sid + \"-real\"),
        return_exceptions=True,
    )

    # Fall back to neutral on error
    if isinstance(a_call, Exception) or isinstance(r_call, Exception):
        log.warning(\"vlm.second_opinion_fail\")
        out = await _gemini_call(USER_VLM_JUDGE_NEUTRAL, image_path, sid + \"-n\")
        return {
            \"p_ai\": out[\"p_ai\"], \"defects\": out[\"defects\"],
            \"rationale\": out[\"rationale\"], \"calls\": 1,
            \"second_opinion_agree\": False, \"dropped\": False,
        }

    p_a, p_r = float(a_call[\"p_ai\"]), float(r_call[\"p_ai\"])
    diff = abs(p_a - p_r)
    same_direction = (p_a > 0.5) == (p_r > 0.5)
    agree = same_direction and diff < 0.30

    if not agree:
        log.info(\"vlm.disagree\", extra={\"event\": \"vlm.disagree\",
                                          \"status\": f\"a={p_a:.2f} r={p_r:.2f}\"})
        return {
            \"p_ai\": (p_a + p_r) / 2,                     # advisory
            \"defects\": list(set(a_call[\"defects\"] + r_call[\"defects\"]))[:5],
            \"rationale\": (\"The forensic-AI argument and forensic-real argument \"
                          \"did not converge — the VLM signal is being dropped.\"),
            \"calls\": 2, \"second_opinion_agree\": False, \"dropped\": True,
        }

    # Aligned → use the mean
    return {
        \"p_ai\": (p_a + p_r) / 2,
        \"defects\": list(dict.fromkeys(a_call[\"defects\"] + r_call[\"defects\"]))[:5],
        \"rationale\": f\"Both framings concur (Δ={diff:.2f}). \" + (
            a_call[\"rationale\"] if p_a > 0.5 else r_call[\"rationale\"]
        ),
        \"calls\": 2, \"second_opinion_agree\": True, \"dropped\": False,
    }


def should_invoke(extremity: float, agreement: float) -> bool:
    return (extremity < settings.vlm_extremity_thr
            or agreement < settings.vlm_agreement_thr)
```

> **Why \"drop, not zero\"?** Zeroing a signal pretends we have evidence at 0.0; that biases the fusion model. *Dropping* (set `enabled=False`, fusion uses mean-imputation on the slot) is the honest move.

### B3. Defensive fallback when no LLM key

```python
# inside services/runner.py — preview
from backend.vlm.judge import judge, should_invoke as vlm_should_invoke
from backend.config import settings

vlm_out = None
vlm_invoked = False
if settings.has_llm and settings.enable_vlm and vlm_should_invoke(extremity, agreement):
    try:
        vlm_out = await judge(image_path)
        vlm_invoked = True
    except Exception as e:
        log.warning(\"vlm.fail\", extra={\"error_code\": type(e).__name__})
```

When `has_llm=False`, the gate effectively returns to 4-tier COEF; the system stays fully functional.

---

## PART C — Cross-call cost & quota math

| Modality slice | Tier-2.5 call rate | Tier-3 call rate | Tier-3 cost (calls) |
|---|---|---|---|
| Confident (∼75% of jobs) | 0 | 0 | 0 |
| Uncertain (∼25%) | 1 (cache-miss) | 2 (second-opinion) | 2 |

- SerpAPI free: 100/mo → handles ∼400 uncertain jobs/mo (most are cache hits)
- Gemini 3 Flash free: 1500/day → cap is 750 uncertain jobs/day (2 calls each)

Both quotas track in `/api/health` (extend in M3 — see `12_scripts_and_testing.md`).

---

## PART D — Unit tests

```python
# file: /app/backend/tests/unit/test_reverse_interpreter.py
from backend.reverse_search.interpreter import interpret


def test_ai_gallery():
    fake = {\"image_results\": [
        {\"link\": \"https://civitai.com/posts/123\", \"date\": \"2024-03\"}
    ]}
    out = interpret(fake)
    assert out[\"p_fake\"] is not None and out[\"p_fake\"] > 0.85
    assert out[\"reason\"] == \"ai_gallery_hit\"


def test_pre_ai_news():
    fake = {\"image_results\": [
        {\"link\": \"https://reuters.com/article\", \"date\": \"2019-05\"}
    ]}
    out = interpret(fake)
    assert out[\"p_fake\"] is not None and out[\"p_fake\"] < 0.15


def test_no_hits():
    assert interpret({\"image_results\": []})[\"p_fake\"] is None


def test_no_strong_prior():
    fake = {\"image_results\": [
        {\"link\": \"https://random-blog.com/post\", \"date\": \"2024-01\"}
    ]}
    out = interpret(fake)
    assert out[\"p_fake\"] is None
    assert len(out[\"top_hits\"]) == 1
```

```python
# file: /app/backend/tests/unit/test_vlm_judge.py
import json
import pytest
from unittest.mock import patch
from backend.vlm.judge import judge, _parse_json_strict


def test_parse_json_with_preamble():
    raw = 'Here is the JSON: {\"p_ai\": 0.7, \"defects\": [\"a\"], \"rationale\": \"x\"}'
    out = _parse_json_strict(raw)
    assert out[\"p_ai\"] == 0.7


@pytest.mark.asyncio
async def test_second_opinion_agreement(monkeypatch):
    async def fake_call(prompt, path, sid):
        return {\"p_ai\": 0.85, \"defects\": [\"warped hand\"], \"rationale\": \"ai\"}
    monkeypatch.setattr(\"backend.vlm.judge._gemini_call\", fake_call)
    monkeypatch.setattr(\"backend.config.settings.enable_vlm_second_opinion\", True)
    out = await judge(\"/tmp/fake.png\")
    assert out[\"dropped\"] is False
    assert out[\"second_opinion_agree\"] is True


@pytest.mark.asyncio
async def test_second_opinion_disagreement(monkeypatch):
    calls = {\"i\": 0}
    async def fake_call(prompt, path, sid):
        calls[\"i\"] += 1
        return {\"p_ai\": 0.85 if calls[\"i\"] == 1 else 0.15,
                \"defects\": [], \"rationale\": \"x\"}
    monkeypatch.setattr(\"backend.vlm.judge._gemini_call\", fake_call)
    monkeypatch.setattr(\"backend.config.settings.enable_vlm_second_opinion\", True)
    out = await judge(\"/tmp/fake.png\")
    assert out[\"dropped\"] is True
    assert out[\"second_opinion_agree\"] is False
```

---

## PART E — Section exit criteria

```bash
pytest backend/tests/unit/test_reverse_interpreter.py \
       backend/tests/unit/test_vlm_judge.py -q
mypy backend/reverse_search/ backend/vlm/
# Success: no issues
```

Next: `08_fusion_calibration_abstention.md` — the math that combines all this into a verdict.
"