"# 10 — Runner Orchestrator + API Routes (M1 Heart)

> Goal: the single file that calls every other module in the correct order,
> with the correct gates, the correct concurrency, and the correct
> short-circuits. This is the file the testing-agent stress-tests at M1
> exit.
>
> **Pipeline overview** (image modality, first-finish):
>
> ```
>   ingest (upload, sha, MIME)
>      │
>      ▼
>   preprocess (decode, EXIF rotate, downscale ≤1024)
>      │
>      ▼
>   Tier 0 — Provenance gate  ────────► HIT? pin verdict, run rest as telemetry
>      │
>      ▼
>   content_type classifier (CLIP zero-shot)
>      │
>      ▼
>   Tier 1 (parallel, gated):
>     prithiv / freq / clip0 / meta / compression
>     ocr_gibberish / eye_forensics (selfie-gated)
>     npr / ufd / dire (heavy — Mac/CUDA only)
>      │
>      ▼
>   Tier 2 — CLIP embedding + FAISS retrieval + OOD-IF
>      │
>      ▼
>   Cold fuse #1 (uniform/LR over Tier-1+2)
>      │
>      ▼
>   Uncertain?   ┐
>      yes       │   ──► Tier 1.5 — Hive + SightEngine (parallel, quota-gated)
>      yes       │   ──► Tier 2.5 — SerpAPI reverse search (cached)
>      yes       │   ──► Tier 3   — Gemini VLM second-opinion
>                ▼
>   Final fuse + cross-modal bonus
>      │
>      ▼
>   Manipulation cross-check (EXIF camera vs FFT/compression)
>      │
>      ▼
>   Abstention (OOD → conformal → content-type gate)
>      │
>      ▼
>   XAI assets + narrator
>      │
>      ▼
>   Persist Result, update Job, return.
> ```

---

## 1. Schema patch — add `MANIPULATED` verdict class

**Before writing the runner**, update `02_backend_skeleton.md §7` Pydantic schemas
to include the fourth class:

```python
# file: /app/backend/schemas/results.py (DELTA — replace the Verdict literal)
Verdict = Literal[\"AI-GENERATED\", \"REAL\", \"INCONCLUSIVE\", \"MANIPULATED\"]
```

And in `08_fusion_calibration_abstention.md §1`:

```python
# file: /app/backend/fusion/types.py (DELTA on Verdict dataclass field)
@dataclass
class Verdict:
    label: Literal[\"AI-GENERATED\", \"REAL\", \"INCONCLUSIVE\", \"MANIPULATED\"]
    confidence: float
    abstained: bool
    rationale: str
```

The MANIPULATED verdict is set by a runner-stage cross-check (this file §3.10),
not by any single detector. Reasoning is in `05c_v15_addendum.md §6` (trick G).

---

## 2. `backend/services/runner.py` — the orchestrator

```python
# file: /app/backend/services/runner.py
\"\"\"Multi-tier evidence orchestrator for image jobs.

Single public coroutine: `run_image_job(job_id: str)`. Reads the job document,
runs all tiers in order with the right gates, writes the Result document,
updates Job state. Never raises — failures are logged + Job goes to status=failed.

This module is the ONLY caller of:
  - backend.provenance.run_tier0
  - backend.detectors.image.image_detectors
  - backend.detectors.content_type.classify
  - backend.retrieval.embed / query / OOD
  - backend.third_party.call_providers
  - backend.reverse_search.lookup
  - backend.vlm.judge
  - backend.fusion.fuse + add_cross_modal
  - backend.abstention.decide
  - backend.xai.build_artefacts
  - backend.narrator.write
\"\"\"
from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from backend.abstention import decide
from backend.abstention.ood import is_ood
from backend.config import settings
from backend.db.repos import (
    get_job, save_result, update_job,
)
from backend.detectors.base import Sample
from backend.detectors.content_type import classify as classify_content_type
from backend.detectors.image import image_detectors
from backend.detectors.image._io import load_rgb
from backend.fusion import add_cross_modal, fuse
from backend.fusion.types import SignalIn
from backend.narrator import build_evidence_packet, write as write_narrative
from backend.provenance import run_tier0
from backend.retrieval.embedder import embed_image
from backend.retrieval.index import query as faiss_query, retrieval_p_fake
from backend.reverse_search import (
    lookup as reverse_lookup, should_invoke as reverse_should_invoke,
)
from backend.third_party import (
    call_providers, should_invoke as t15_should_invoke,
)
from backend.utils.timing import Timings
from backend.utils.upload import public_url_for
from backend.vlm.judge import judge as vlm_judge, should_invoke as vlm_should_invoke
from backend.xai import build_artefacts

log = logging.getLogger(\"runner\")

# Per-detector wall-clock cap — keeps a hung model from blocking the job
PER_DETECTOR_TIMEOUT_S = 25.0


# ─────────────────────── helpers ──────────────────────────────────────────
async def _safe_detector(d, sample: Sample) -> \"SignalIn | None\":
    \"\"\"Run one detector with a hard timeout. Convert DetectorOutput→SignalIn.\"\"\"
    try:
        out = await asyncio.wait_for(d.predict(sample),
                                     timeout=PER_DETECTOR_TIMEOUT_S)
    except asyncio.TimeoutError:
        log.warning(\"detector.timeout\", extra={\"signal_name\": d.name})
        return None
    except Exception as e:
        log.warning(\"detector.error\", extra={\"signal_name\": d.name,
                                              \"error_code\": type(e).__name__})
        return None
    if not out.enabled:
        return SignalIn(name=out.name, p_fake=out.p_fake,
                        enabled=False, weight_hint=1.0)
    return SignalIn(name=out.name, p_fake=out.p_fake, enabled=True,
                    weight_hint=1.0)


def _detector_output_to_signal_dict(detector_name: str,
                                    p_fake: float,
                                    explanation: str,
                                    weight: float) -> dict:
    return {\"name\": detector_name, \"p_fake\": float(p_fake),
            \"weight\": float(weight), \"explanation\": explanation}


def _manipulation_check(signals_raw: dict[str, dict],
                        provenance_hit: bool) -> dict | None:
    \"\"\"Cross-check: EXIF claims a real camera + FFT/compression says AI.

    Returns a flag dict when triggered; otherwise None. Pure logic, no I/O.\"\"\"
    if provenance_hit:
        return None
    meta = signals_raw.get(\"img.meta\")
    freq = signals_raw.get(\"img.freq\")
    comp = signals_raw.get(\"img.compression\")
    if not (meta and freq and comp):
        return None
    # Camera-shape EXIF → meta p_fake should be < 0.30 (REAL-leaning)
    if meta[\"p_fake\"] >= 0.30:
        return None
    # But forensic signals contradict
    if freq[\"p_fake\"] >= 0.65 and comp[\"p_fake\"] >= 0.65:
        return {
            \"reason\": \"exif_freq_mismatch\",
            \"detail\": (\"EXIF metadata is camera-shape (real-leaning) but \"
                       \"frequency-domain and compression fingerprints both \"
                       \"match a diffusion-model signature.\"),
            \"meta_p_fake\": meta[\"p_fake\"],
            \"freq_p_fake\": freq[\"p_fake\"],
            \"compression_p_fake\": comp[\"p_fake\"],
        }
    return None


# ─────────────────────── main entry ───────────────────────────────────────
async def run_image_job(job_id: str) -> None:
    \"\"\"Read job, run pipeline, persist result. Never raises out.\"\"\"
    timings = Timings()
    try:
        await _run(job_id, timings)
    except Exception as e:
        log.exception(\"runner.crash\", extra={\"job_id\": job_id,
                                              \"error_code\": type(e).__name__})
        await update_job(job_id, {\"status\": \"failed\",
                                  \"error\": f\"{type(e).__name__}: {e}\",
                                  \"stage\": \"crash\",
                                  \"progress\": 1.0})


async def _run(job_id: str, timings: Timings) -> None:
    job = await get_job(job_id)
    if job is None:
        log.error(\"runner.no_job\", extra={\"job_id\": job_id})
        return

    inp = job[\"input\"]
    image_path = inp[\"path\"]
    sha = inp[\"sha256\"]
    modality = job[\"modality\"]
    profile = job[\"profile\"]

    if modality != \"image\":
        # Phase-1 follow-up will route audio/video. For now: fail clean.
        await update_job(job_id, {\"status\": \"failed\",
                                  \"error\": \"modality not supported in M1\",
                                  \"stage\": \"ingest\", \"progress\": 1.0})
        return

    await update_job(job_id, {\"status\": \"running\", \"stage\": \"preprocess\",
                              \"progress\": 0.05})

    # ─── Stage 1: preprocess ──────────────────────────────────────────────
    with timings.stage(\"preprocess\"):
        image_rgb = load_rgb(image_path)
    log.info(\"runner.preprocess\", extra={\"job_id\": job_id,
                                          \"dur_ms\": timings.data[\"preprocess\"]})

    # ─── Stage 2: Tier 0 provenance gate ─────────────────────────────────
    await update_job(job_id, {\"stage\": \"tier0\", \"progress\": 0.10})
    with timings.stage(\"tier0\"):
        prov = await run_tier0(Path(image_path))

    # ─── Stage 3: content-type classify ──────────────────────────────────
    await update_job(job_id, {\"stage\": \"content_type\", \"progress\": 0.15})
    with timings.stage(\"content_type\"):
        try:
            content_type, ct_scores = await classify_content_type(image_rgb)
        except Exception as e:
            log.warning(\"content_type.fail\",
                        extra={\"error_code\": type(e).__name__})
            content_type, ct_scores = \"object_product\", {}

    sample = Sample(image_rgb=image_rgb, image_path=image_path,
                    sha256=sha, mime=inp[\"mime\"], bytes=inp[\"bytes\"],
                    content_type=content_type)

    # ─── Stage 4: Tier 1 detectors (parallel) ────────────────────────────
    await update_job(job_id, {\"stage\": \"tier1\", \"progress\": 0.25})
    with timings.stage(\"tier1\"):
        detectors = image_detectors()
        # Build raw DetectorOutput list (we keep both formats)
        raw_outputs: dict[str, Any] = {}
        signal_ins: list[SignalIn] = []

        async def _run_one(d):
            try:
                out = await asyncio.wait_for(d.predict(sample),
                                              timeout=PER_DETECTOR_TIMEOUT_S)
                return d.name, out
            except asyncio.TimeoutError:
                log.warning(\"detector.timeout\",
                            extra={\"signal_name\": d.name})
                return d.name, None
            except Exception as e:
                log.warning(\"detector.error\",
                            extra={\"signal_name\": d.name,
                                   \"error_code\": type(e).__name__})
                return d.name, None

        results = await asyncio.gather(*[_run_one(d) for d in detectors])
        for name, out in results:
            if out is None:
                continue
            raw_outputs[name] = out
            signal_ins.append(SignalIn(
                name=name, p_fake=float(out.p_fake),
                enabled=bool(out.enabled), weight_hint=1.0,
            ))

    # Snapshot of raw signals as plain dicts (used by manipulation check + result)
    signals_raw: dict[str, dict] = {
        name: {\"p_fake\": float(out.p_fake),
               \"explanation\": out.explanation,
               \"artifacts\": out.artifacts,
               \"elapsed_ms\": out.elapsed_ms,
               \"enabled\": out.enabled}
        for name, out in raw_outputs.items()
    }

    # ─── Stage 5: Tier 2 retrieval (CLIP embed + FAISS) ──────────────────
    await update_job(job_id, {\"stage\": \"tier2\", \"progress\": 0.42})
    p_retrieval = None
    retrieval_meta: dict[str, Any] = {\"neighbors\": [], \"self_leak\": False}
    ood_flag = False
    ood_scores: dict[str, Any] = {}
    with timings.stage(\"tier2\"):
        try:
            vec = await embed_image(image_rgb)
            # Self-leak guard
            self_leak_id = sha[:24]
            neighbors = faiss_query(\"image\", vec, k=15,
                                     exclude_id=self_leak_id)
            p_retrieval = retrieval_p_fake(neighbors)
            retrieval_meta[\"neighbors\"] = [
                {\"id\": n.id, \"label\": n.label,
                 \"distance\": float(n.distance),
                 \"thumb_url\": f\"/api/refdb/thumb/{n.id}.jpg\"}
                for n in neighbors[:8]
            ]
            signal_ins.append(SignalIn(
                name=\"img.retrieval\", p_fake=float(p_retrieval), enabled=True,
                weight_hint=1.0,
            ))
            signals_raw[\"img.retrieval\"] = {
                \"p_fake\": float(p_retrieval),
                \"explanation\": f\"retrieval k=15, top1 distance=\"
                               f\"{neighbors[0].distance:.3f}\" if neighbors
                               else \"no neighbours\",
                \"artifacts\": {\"k\": 15, \"n_neighbors\": len(neighbors)},
                \"elapsed_ms\": timings.data.get(\"tier2\", 0),
                \"enabled\": True,
            }
            # OOD check
            ood_flag, ood_scores = is_ood(\"image\", vec)
        except Exception as e:
            log.warning(\"tier2.fail\", extra={\"error_code\": type(e).__name__})

    # ─── Stage 6: cold fuse to compute extremity/agreement gates ─────────
    await update_job(job_id, {\"stage\": \"cold_fuse\", \"progress\": 0.50})
    with timings.stage(\"cold_fuse\"):
        cold = fuse(signal_ins)
        extremity = float(cold.extremity)
        agreement = float(cold.agreement)
    log.info(\"runner.cold_fuse\", extra={\"job_id\": job_id,
                                         \"status\": f\"p_ai={cold.p_ai:.2f}, \"
                                                   f\"extr={extremity:.2f}, \"
                                                   f\"agree={agreement:.2f}\"})

    # ─── Stage 7: Tier 1.5 third-party (gated) ───────────────────────────
    third_party_results: list = []
    if not prov.hit and t15_should_invoke(extremity, agreement) and (
            settings.hive_api_key or settings.sightengine_user):
        await update_job(job_id, {\"stage\": \"tier1_5\", \"progress\": 0.55})
        with timings.stage(\"tier1_5\"):
            try:
                # We need a public URL for some providers; pass the bytes too
                image_url = public_url_for(job_id, image_path)
                image_bytes = Path(image_path).read_bytes()
                third_party_results = await call_providers(image_url,
                                                            image_bytes)
                for pr in third_party_results:
                    if pr.p_fake is None or not pr.invoked:
                        continue
                    sig_name = f\"img.t15.{pr.provider}\"
                    signal_ins.append(SignalIn(
                        name=sig_name, p_fake=float(pr.p_fake),
                        enabled=True, weight_hint=1.0,
                    ))
                    signals_raw[sig_name] = {
                        \"p_fake\": float(pr.p_fake),
                        \"explanation\": pr.explanation,
                        \"artifacts\": pr.raw,
                        \"elapsed_ms\": pr.elapsed_ms,
                        \"enabled\": True,
                    }
            except Exception as e:
                log.warning(\"tier1_5.fail\",
                            extra={\"error_code\": type(e).__name__})

    # ─── Stage 8: Tier 2.5 reverse search (gated) ────────────────────────
    reverse_out: dict[str, Any] = {\"p_fake\": None, \"reason\": \"not_invoked\",
                                    \"top_hits\": [], \"invoked\": False,
                                    \"cached\": False}
    if not prov.hit and reverse_should_invoke(extremity, agreement,
                                                p_retrieval):
        await update_job(job_id, {\"stage\": \"tier2_5\", \"progress\": 0.62})
        with timings.stage(\"tier2_5\"):
            try:
                image_url = public_url_for(job_id, image_path)
                reverse_out = await reverse_lookup(image_url, sha)
                if reverse_out.get(\"p_fake\") is not None:
                    signal_ins.append(SignalIn(
                        name=\"img.reverse\",
                        p_fake=float(reverse_out[\"p_fake\"]),
                        enabled=True, weight_hint=1.0,
                    ))
                    signals_raw[\"img.reverse\"] = {
                        \"p_fake\": float(reverse_out[\"p_fake\"]),
                        \"explanation\": f\"reverse search: {reverse_out['reason']}\",
                        \"artifacts\": {\"top_hits\": reverse_out[\"top_hits\"],
                                      \"cached\": reverse_out.get(\"cached\")},
                        \"elapsed_ms\": timings.data.get(\"tier2_5\", 0),
                        \"enabled\": True,
                    }
            except Exception as e:
                log.warning(\"tier2_5.fail\",
                            extra={\"error_code\": type(e).__name__})

    # ─── Stage 9: Tier 3 VLM (gated) ──────────────────────────────────────
    vlm_out: dict[str, Any] = {\"invoked\": False}
    if (not prov.hit and settings.has_llm and settings.enable_vlm
            and vlm_should_invoke(extremity, agreement)):
        await update_job(job_id, {\"stage\": \"tier3\", \"progress\": 0.72})
        with timings.stage(\"tier3\"):
            try:
                vlm_out = await vlm_judge(image_path)
                if not vlm_out.get(\"dropped\"):
                    signal_ins.append(SignalIn(
                        name=\"img.vlm\", p_fake=float(vlm_out[\"p_ai\"]),
                        enabled=True, weight_hint=1.0,
                    ))
                    signals_raw[\"img.vlm\"] = {
                        \"p_fake\": float(vlm_out[\"p_ai\"]),
                        \"explanation\": vlm_out.get(\"rationale\", \"\"),
                        \"artifacts\": {\"defects\": vlm_out.get(\"defects\", []),
                                      \"calls\": vlm_out.get(\"calls\", 1)},
                        \"elapsed_ms\": timings.data.get(\"tier3\", 0),
                        \"enabled\": True,
                    }
                vlm_out[\"invoked\"] = True
            except Exception as e:
                log.warning(\"tier3.fail\",
                            extra={\"error_code\": type(e).__name__})

    # ─── Stage 10: final fuse + cross-modal bonus ─────────────────────────
    await update_job(job_id, {\"stage\": \"fuse\", \"progress\": 0.82})
    with timings.stage(\"fuse\"):
        fr = fuse(signal_ins)
        fr = add_cross_modal(fr, signal_ins)

    # ─── Stage 10.5: manipulation cross-check ────────────────────────────
    manipulation_flag = _manipulation_check(signals_raw,
                                             provenance_hit=prov.hit)

    # ─── Stage 11: abstention / verdict ──────────────────────────────────
    await update_job(job_id, {\"stage\": \"abstain\", \"progress\": 0.87})
    with timings.stage(\"abstain\"):
        if prov.hit:
            # Tier 0 short-circuits everything
            label = \"AI-GENERATED\" if prov.p_ai > 0.5 else \"REAL\"
            verdict_out = type(\"V\", (), {\"label\": label,
                                          \"confidence\": 0.99,
                                          \"abstained\": False,
                                          \"rationale\":
                                          f\"Tier-0 provenance: {prov.source}\"})()
            fr_p_ai = prov.p_ai
            fr_conf = 0.99
        elif manipulation_flag:
            verdict_out = type(\"V\", (), {
                \"label\": \"MANIPULATED\",
                \"confidence\": min(0.95, max(0.55, fr.p_ai)),
                \"abstained\": False,
                \"rationale\": manipulation_flag[\"detail\"],
            })()
            fr_p_ai = fr.p_ai
            fr_conf = verdict_out.confidence
        else:
            verdict_out = decide(fr, signal_ins, sample.content_type,
                                  ood_flag=ood_flag)
            fr_p_ai = fr.p_ai
            fr_conf = verdict_out.confidence

    # ─── Stage 12: XAI assets ────────────────────────────────────────────
    await update_job(job_id, {\"stage\": \"xai\", \"progress\": 0.92})
    with timings.stage(\"xai\"):
        comp_art = signals_raw.get(\"img.compression\", {}).get(\"artifacts\", {})
        eye_art = signals_raw.get(\"img.eye_forensics\", {}).get(\"artifacts\", {})
        meta_sum = (signals_raw.get(\"img.meta\", {})
                     .get(\"artifacts\", {}).get(\"exif_summary\", {}))
        xai_payload = await build_artefacts(
            job_id=job_id,
            image_rgb=image_rgb,
            compression_artifacts=comp_art,
            eye_artifacts=eye_art,
            meta_summary=meta_sum,
        )

    # ─── Stage 13: narrative ─────────────────────────────────────────────
    await update_job(job_id, {\"stage\": \"narrate\", \"progress\": 0.97})
    with timings.stage(\"narrate\"):
        evidence_packet = build_evidence_packet({
            \"verdict\": verdict_out.label,
            \"p_ai_generated\": fr_p_ai,
            \"confidence\": fr_conf,
            \"agreement\": fr.agreement,
            \"content_type\": sample.content_type,
            \"abstained\": verdict_out.abstained,
            \"signals\": [
                {\"name\": n, \"p_fake\": s[\"p_fake\"],
                 \"weight\": fr.weights.get(n, 0.0),
                 \"explanation\": s[\"explanation\"]}
                for n, s in signals_raw.items()
            ],
            \"provenance\": {\"hit\": prov.hit, \"source\": prov.source,
                           \"details\": prov.details},
            \"reverse_search\": {\"reason\": reverse_out.get(\"reason\"),
                                \"top_hits\": reverse_out.get(\"top_hits\", [])},
            \"debug\": {\"vlm\": vlm_out, \"ood_flag\": ood_flag,
                      \"manipulation_flag\": manipulation_flag},
        })
        narrative, narr_source = await write_narrative(evidence_packet)

    xai_payload[\"narrative\"] = narrative
    xai_payload[\"narrative_source\"] = narr_source

    # ─── Stage 14: persist + finish ──────────────────────────────────────
    result_doc = _build_result_doc(
        job_id=job_id, modality=modality, profile=profile,
        content_type=sample.content_type,
        signals_raw=signals_raw,
        fusion_result=fr, verdict=verdict_out,
        provenance=prov, reverse_out=reverse_out,
        retrieval_meta=retrieval_meta,
        third_party_results=third_party_results,
        xai_payload=xai_payload,
        manipulation_flag=manipulation_flag,
        ood_flag=ood_flag, ood_scores=ood_scores,
        ct_scores=ct_scores,
        durations_ms=timings.data,
        input_dict=inp,
        vlm_out=vlm_out,
    )

    await save_result(result_doc)
    await update_job(job_id, {\"status\": \"done\", \"stage\": \"done\",
                              \"progress\": 1.0,
                              \"finished_at\": result_doc[\"finished_at\"]})
    log.info(\"runner.done\", extra={\"job_id\": job_id,
                                    \"status\": verdict_out.label})


# ─────────────────────── result builder ───────────────────────────────────
def _build_result_doc(*, job_id, modality, profile, content_type,
                      signals_raw, fusion_result, verdict,
                      provenance, reverse_out, retrieval_meta,
                      third_party_results, xai_payload,
                      manipulation_flag, ood_flag, ood_scores,
                      ct_scores, durations_ms, input_dict,
                      vlm_out) -> dict:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    # Build serialisable signal list
    sig_list = []
    for name, s in signals_raw.items():
        sig_list.append({
            \"name\": name,
            \"p_fake\": s[\"p_fake\"],
            \"weight\": fusion_result.weights.get(name, 0.0),
            \"explanation\": s[\"explanation\"],
        })
    sig_list.sort(key=lambda x: abs(x[\"p_fake\"] - 0.5), reverse=True)

    return {
        \"job_id\": job_id,
        \"modality\": modality,
        \"profile\": profile,
        \"calibration\": fusion_result.calibration,
        \"fusion_model\": fusion_result.fusion_model,
        \"content_type\": content_type,
        \"content_type_scores\": ct_scores,
        \"verdict\": verdict.label,
        \"p_ai_generated\": float(provenance.p_ai if provenance.hit
                                else fusion_result.p_ai),
        \"confidence\": float(verdict.confidence),
        \"agreement\": float(fusion_result.agreement),
        \"extremity\": float(fusion_result.extremity),
        \"cross_modal_bonus\": float(fusion_result.cross_modal_bonus),
        \"abstained\": bool(verdict.abstained),
        \"provenance\": {
            \"hit\": provenance.hit,
            \"source\": provenance.source,
            \"details\": provenance.details,
        },
        \"vlm_invoked\": bool(vlm_out.get(\"invoked\", False)),
        \"reverse_invoked\": bool(reverse_out.get(\"invoked\", False)),
        \"signals\": sig_list,
        \"retrieval\": retrieval_meta,
        \"reverse_search\": reverse_out,
        \"third_party\": [
            {\"provider\": pr.provider,
             \"p_fake\": pr.p_fake,
             \"invoked\": pr.invoked,
             \"explanation\": pr.explanation,
             \"elapsed_ms\": pr.elapsed_ms}
            for pr in third_party_results
        ],
        \"xai\": xai_payload,
        \"input\": input_dict,
        \"durations_ms\": durations_ms,
        \"finished_at\": now,
        \"debug\": {
            \"ood_flag\": ood_flag,
            \"ood_scores\": ood_scores,
            \"manipulation_flag\": manipulation_flag,
            \"vlm\": {k: v for k, v in vlm_out.items() if k != \"raw\"},
            \"verdict_rationale\": verdict.rationale,
        },
    }
```

> **Why one giant function instead of an OO state machine?** The stages are
> linear and read top-to-bottom in the same order as the Masterplan figure.
> A class would scatter the flow across methods; the long function keeps the
> 5-tier story visible in a single screen of code. AGENTS.md naming rule.

---

## 3. `backend/services/jobs.py` — fire-and-forget background queue

We don't need Celery for a single-user local app. `asyncio.create_task` is
enough. Concurrency cap = 2 to avoid two GPU jobs evicting each other's models.

```python
# file: /app/backend/services/jobs.py
\"\"\"In-process async job queue. Concurrency capped at 2 (configurable).

Caller does:
    job_id = await enqueue_image_job(input_dict)
    # returns immediately. Job processes in the background.

Status is polled via routes/jobs.py::GET /jobs/{id}.\"\"\"
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from backend.db.repos import create_job
from backend.services.device import detect_profile
from backend.services.runner import run_image_job

log = logging.getLogger(\"jobs\")

_SEM = asyncio.Semaphore(2)


async def _wrap(job_id: str) -> None:
    async with _SEM:
        log.info(\"job.start\", extra={\"job_id\": job_id})
        await run_image_job(job_id)
        log.info(\"job.done\", extra={\"job_id\": job_id})


async def enqueue_image_job(input_dict: dict[str, Any]) -> str:
    \"\"\"Insert the Job document, fire the background task, return job_id.\"\"\"
    job_id = str(uuid.uuid4())
    await create_job({
        \"_id\": job_id,
        \"status\": \"queued\",
        \"stage\": \"queued\",
        \"progress\": 0.0,
        \"modality\": \"image\",
        \"profile\": detect_profile(),
        \"input\": input_dict,
        \"error\": None,
    })
    # Schedule. We deliberately don't `await` the wrap.
    asyncio.create_task(_wrap(job_id))
    return job_id
```

---

## 4. `backend/utils/upload.py` — file persistence + public URLs

```python
# file: /app/backend/utils/upload.py
\"\"\"Save uploaded bytes to /app/backend/storage/jobs/{job_id}/upload.{ext},
return a dict ready to drop into `Job.input`.

Also exposes `public_url_for(job_id, local_path)` — builds the URL the runner
hands to SerpAPI / Hive so they can fetch the asset from the public preview
URL.\"\"\"
from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.utils.errors import too_large, unsupported_mime

log = logging.getLogger(\"upload\")

STORAGE_ROOT = Path(\"/app/backend/storage/jobs\")
ALLOWED_IMAGE_EXT = {\".jpg\", \".jpeg\", \".png\", \".webp\"}
EXT_FOR_MIME = {
    \"image/jpeg\": \".jpg\", \"image/png\": \".png\", \"image/webp\": \".webp\",
}


def _save_atomic(dst: Path, data: bytes) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + \".tmp\")
    tmp.write_bytes(data)
    os.replace(tmp, dst)


def store_upload(job_id: str, filename: str, data: bytes,
                 declared_mime: str | None = None) -> dict[str, Any]:
    \"\"\"Persist + compute sha + return input dict.\"\"\"
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise too_large(settings.max_upload_mb)

    sniffed_mime = declared_mime or mimetypes.guess_type(filename)[0] or \"\"
    if sniffed_mime not in EXT_FOR_MIME:
        # python-magic sniff as a backup
        try:
            import magic
            sniffed_mime = magic.from_buffer(data[:4096], mime=True)
        except Exception:
            pass
    if sniffed_mime not in EXT_FOR_MIME:
        raise unsupported_mime(f\"unsupported mime {sniffed_mime!r}\")

    ext = EXT_FOR_MIME[sniffed_mime]
    dst = STORAGE_ROOT / job_id / f\"upload{ext}\"
    _save_atomic(dst, data)
    sha = hashlib.sha256(data).hexdigest()

    return {
        \"filename\": filename,
        \"sha256\": sha,
        \"bytes\": len(data),
        \"mime\": sniffed_mime,
        \"path\": str(dst),
    }


def public_url_for(job_id: str, local_path: str) -> str:
    \"\"\"Build the absolute URL the runner can hand to SerpAPI / Hive.

    The Kubernetes ingress maps /api/jobs/{id}/upload to the file. We use the
    FRONTEND-side REACT_APP_BACKEND_URL... no, wait — backend writes its OWN
    canonical URL. For this app, callers use a `PUBLIC_BACKEND_URL` env var.
    Fallback to localhost (tests).\"\"\"
    import os
    base = os.environ.get(\"PUBLIC_BACKEND_URL\", \"\").rstrip(\"/\")
    if not base:
        # Final fallback — works for local curl tests
        base = \"http://localhost:8001\"
    # Map local path to its serving route
    p = Path(local_path)
    name = p.name
    return f\"{base}/api/jobs/{job_id}/upload/{name}\"
```

> Add to `01_setup.md §3.1` outline:
>
> ```
> # --- For Tier 2.5/1.5 callbacks that need a public URL for the upload ---
> PUBLIC_BACKEND_URL=
> ```
>
> On the Emergent preview this is the same as `REACT_APP_BACKEND_URL`. SerpAPI
> requires an externally reachable URL; localhost will not work in production.

---

## 5. `backend/routes/analyze.py` — POST /analyze (full body)

```python
# file: /app/backend/routes/analyze.py
\"\"\"POST /api/analyze — multipart upload kicks off a job.\"\"\"
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, File, UploadFile

from backend.services.jobs import enqueue_image_job
from backend.utils.errors import AppError
from backend.utils.upload import store_upload

log = logging.getLogger(\"route.analyze\")
router = APIRouter()


@router.post(\"/analyze\")
async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    # We do not know job_id until we call store_upload — but we want path
    # /jobs/{id}/upload.ext, so we first generate the id ourselves.
    import uuid
    job_id = str(uuid.uuid4())
    try:
        input_dict = store_upload(job_id, file.filename or \"upload.bin\",
                                   data, file.content_type)
    except AppError:
        raise
    # Pre-populate the job with the correct id (overrides default in jobs.py)
    from backend.db.repos import create_job
    from backend.services.device import detect_profile
    await create_job({
        \"_id\": job_id, \"status\": \"queued\", \"stage\": \"queued\",
        \"progress\": 0.0, \"modality\": \"image\",
        \"profile\": detect_profile(),
        \"input\": input_dict, \"error\": None,
    })
    # Fire the runner WITHOUT going through enqueue_image_job (which would
    # generate a fresh id)
    import asyncio
    from backend.services.runner import run_image_job
    asyncio.create_task(run_image_job(job_id))

    return {\"job_id\": job_id, \"status\": \"queued\",
            \"input\": {\"filename\": input_dict[\"filename\"],
                       \"sha256\": input_dict[\"sha256\"],
                       \"bytes\": input_dict[\"bytes\"],
                       \"mime\": input_dict[\"mime\"]}}
```

> The duplication with `services/jobs.py::enqueue_image_job` is deliberate —
> the upload route needs to *know* the job id *before* the file is saved (so
> the asset path is `/jobs/{id}/upload.ext`). `enqueue_image_job` exists for
> programmatic callers (cron, tests) that don't care about the id ordering.

---

## 6. `backend/routes/jobs.py` — full body

```python
# file: /app/backend/routes/jobs.py
\"\"\"Job lifecycle endpoints.\"\"\"
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from backend.db.repos import get_job, get_result

log = logging.getLogger(\"route.jobs\")
router = APIRouter()


ALLOWED_ASSETS = {\"heatmap.png\", \"frequency.png\",
                  \"compression.png\", \"eye_overlay.png\"}


@router.get(\"/jobs/{job_id}\")
async def job_status(job_id: str) -> dict:
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(404, \"job not found\")
    return {
        \"job_id\": job_id,
        \"modality\": job[\"modality\"],
        \"status\": job[\"status\"],
        \"stage\": job.get(\"stage\", \"\"),
        \"progress\": float(job.get(\"progress\", 0.0)),
        \"started_at\": job.get(\"created_at\"),
        \"finished_at\": job.get(\"finished_at\"),
        \"error\": job.get(\"error\"),
    }


@router.get(\"/jobs/{job_id}/result\")
async def job_result(job_id: str, debug: int = 0) -> dict:
    result = await get_result(job_id)
    if result is None:
        # If the job exists but is not done, return helpful 202
        job = await get_job(job_id)
        if job and job[\"status\"] != \"done\":
            raise HTTPException(202,
                                f\"job not finished (status={job['status']})\")
        raise HTTPException(404, \"result not found\")
    # Strip Mongo _id if present
    result.pop(\"_id\", None)
    if not debug:
        result.pop(\"debug\", None)
    return result


@router.get(\"/jobs/{job_id}/assets/{name}\")
async def job_asset(job_id: str, name: str):
    if name not in ALLOWED_ASSETS:
        raise HTTPException(404, \"asset not allowed\")
    if \"/\" in name or \"\\\" in name or \"..\" in name:
        raise HTTPException(400, \"bad name\")
    p = Path(\"/app/backend/storage/jobs\") / job_id / \"assets\" / name
    if not p.exists():
        raise HTTPException(404, \"asset not found\")
    return FileResponse(str(p), media_type=\"image/png\")


@router.get(\"/jobs/{job_id}/upload/{name}\")
async def job_upload(job_id: str, name: str):
    \"\"\"Serve the original upload (used by SerpAPI/Hive callbacks).\"\"\"
    if \"/\" in name or \"\\\" in name or \"..\" in name:
        raise HTTPException(400, \"bad name\")
    p = Path(\"/app/backend/storage/jobs\") / job_id / name
    if not p.exists():
        raise HTTPException(404, \"upload not found\")
    # Guess content type from extension
    suffix = p.suffix.lower()
    mt = {\".jpg\": \"image/jpeg\", \".jpeg\": \"image/jpeg\",
          \".png\": \"image/png\", \".webp\": \"image/webp\"}.get(suffix,
                                                            \"application/octet-stream\")
    return FileResponse(str(p), media_type=mt)


@router.get(\"/jobs/{job_id}/report.json\")
async def job_report(job_id: str) -> Response:
    \"\"\"Downloadable JSON report (same as /result?debug=1, content-disposition set).\"\"\"
    result = await get_result(job_id)
    if result is None:
        raise HTTPException(404, \"result not found\")
    result.pop(\"_id\", None)
    body = json.dumps(result, indent=2, default=str).encode()
    return Response(
        content=body, media_type=\"application/json\",
        headers={\"Content-Disposition\":
                 f'attachment; filename=\"report-{job_id[:8]}.json\"'},
    )
```

---

## 7. `backend/routes/history.py` — full body

```python
# file: /app/backend/routes/history.py
\"\"\"GET /api/history — recent jobs for the history side panel.\"\"\"
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from backend.db.repos import list_jobs, get_result

log = logging.getLogger(\"route.history\")
router = APIRouter()


@router.get(\"/history\")
async def history(limit: int = 20) -> dict[str, Any]:
    limit = max(1, min(100, int(limit)))
    jobs = await list_jobs(limit=limit)
    items = []
    for j in jobs:
        item = {
            \"job_id\": j[\"_id\"],
            \"modality\": j.get(\"modality\"),
            \"status\": j.get(\"status\"),
            \"created_at\": j.get(\"created_at\"),
            \"finished_at\": j.get(\"finished_at\"),
            \"filename\": j.get(\"input\", {}).get(\"filename\"),
            \"verdict\": None,
            \"p_ai_generated\": None,
            \"confidence\": None,
        }
        if j.get(\"status\") == \"done\":
            r = await get_result(j[\"_id\"])
            if r:
                item[\"verdict\"] = r.get(\"verdict\")
                item[\"p_ai_generated\"] = r.get(\"p_ai_generated\")
                item[\"confidence\"] = r.get(\"confidence\")
        items.append(item)
    return {\"items\": items, \"count\": len(items)}
```

---

## 8. `backend/routes/correct.py` — full body

```python
# file: /app/backend/routes/correct.py
\"\"\"POST /api/jobs/{id}/correct — user correction.

Saves the label, appends the upload's CLIP embedding to the hard-negative
partition (so retrieval improves immediately, no re-training), and schedules
a calibration refresh when the unconsumed-label count crosses 50/100/200.\"\"\"
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.db.repos import get_job, get_result, save_label, unconsumed_labels
from backend.detectors.image._io import load_rgb
from backend.retrieval.embedder import embed_image
from backend.retrieval.hard_negatives import append as hard_neg_append

log = logging.getLogger(\"route.correct\")
router = APIRouter()


class Correction(BaseModel):
    label: Literal[\"ai\", \"real\"]
    note: str = \"\"


@router.post(\"/jobs/{job_id}/correct\")
async def correct(job_id: str, body: Correction) -> dict:
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(404, \"job not found\")
    res = await get_result(job_id)
    if res is None:
        raise HTTPException(404, \"no result to correct\")

    # 1. Persist label
    label_id = f\"{job_id}-{body.label}\"
    await save_label({
        \"_id\": label_id,
        \"job_id\": job_id,
        \"label\": body.label,
        \"note\": body.note[:240],
        \"path\": job[\"input\"][\"path\"],
        \"predicted_verdict\": res.get(\"verdict\"),
        \"predicted_p_ai\": res.get(\"p_ai_generated\"),
    })

    # 2. Append to hard-negative bank (immediate retrieval boost)
    try:
        img = load_rgb(job[\"input\"][\"path\"])
        vec = await embed_image(img)
        new_size = hard_neg_append(\"image\", body.label, vec,
                                    source=f\"job:{job_id}\")
    except Exception as e:
        log.warning(\"correct.hardneg_fail\",
                    extra={\"error_code\": type(e).__name__})
        new_size = -1

    # 3. Check whether to suggest recalibration
    pending = await unconsumed_labels()
    should_recalibrate = len(pending) in (50, 100, 200)
    return {
        \"ok\": True,
        \"label_id\": label_id,
        \"hard_negative_bank_size\": new_size,
        \"pending_labels\": len(pending),
        \"should_recalibrate\": should_recalibrate,
    }
```

---

## 9. `backend/routes/refdb.py` — expanded (stats + thumb)

```python
# file: /app/backend/routes/refdb.py
\"\"\"Reference-DB endpoints. Stats + per-id thumbnail serving.\"\"\"
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.retrieval.index import refdb_stats

log = logging.getLogger(\"route.refdb\")
router = APIRouter()

THUMB_DIR = Path(\"/app/backend/storage/refdb/thumbs\")


@router.get(\"/refdb/stats\")
async def stats() -> dict:
    return refdb_stats()


@router.get(\"/refdb/thumb/{thumb_id}.jpg\")
async def thumb(thumb_id: str):
    # Path traversal guard
    if \"/\" in thumb_id or \"\\\" in thumb_id or \"..\" in thumb_id:
        raise HTTPException(400, \"bad id\")
    p = THUMB_DIR / f\"{thumb_id}.jpg\"
    if not p.exists():
        raise HTTPException(404, \"thumb not found\")
    return FileResponse(str(p), media_type=\"image/jpeg\")
```

---

## 10. `backend/routes/health.py` — final form

Replace the M0 stub from `02_backend_skeleton.md §8.1`:

```python
# file: /app/backend/routes/health.py
\"\"\"GET /api/health — comprehensive readiness probe.\"\"\"
from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import APIRouter

from backend.config import settings
from backend.db.mongo import db
from backend.detectors.registry import loaded_signals
from backend.retrieval.index import refdb_stats
from backend.services.device import detect_profile

log = logging.getLogger(\"route.health\")
router = APIRouter()
_START = time.time()


@router.get(\"/health\")
async def health() -> dict:
    db_ok = True
    try:
        await db().command(\"ping\")
    except Exception:
        db_ok = False

    stats = refdb_stats()
    refdb_loaded = stats.get(\"loaded\", False)

    # Calibration metrics (if file exists)
    calib_path = Path(\"/app/backend/storage/refdb/calibration.json\")
    ece = None
    auroc = None
    if calib_path.exists():
        import json
        try:
            data = json.loads(calib_path.read_text())
            eces = [v[\"ece\"] for v in data.values() if isinstance(v, dict)]
            aurs = [v[\"auroc\"] for v in data.values() if isinstance(v, dict)]
            if eces:
                ece = round(sum(eces) / len(eces), 4)
            if aurs:
                auroc = round(sum(aurs) / len(aurs), 4)
        except Exception:
            pass

    # Pending user labels
    n_user_labels = 0
    try:
        n_user_labels = await db().labels.count_documents({\"consumed\": False})
    except Exception:
        pass

    return {
        \"status\": \"ok\" if db_ok else \"degraded\",
        \"profile\": detect_profile(),
        \"signals_loaded\": loaded_signals(),
        \"db_ok\": db_ok,
        \"gemini_ok\": settings.has_llm,
        \"serpapi_ok\": bool(settings.serpapi_key),
        \"hive_ok\": bool(settings.hive_api_key),
        \"sightengine_ok\": bool(settings.sightengine_user
                                 and settings.sightengine_secret),
        \"refdb_loaded\": refdb_loaded,
        \"refdb_size\": stats.get(\"sizes\", {}),
        \"fusion_mode\": \"uniform\",            # set to \"lr_l2\" / \"gbdt\" by /fusion/__init__ at boot
        \"calibration\": \"platt_refdb\",
        \"ece_refdb_holdout\": ece,
        \"auroc_refdb_holdout\": auroc,
        \"n_user_labels\": n_user_labels,
        \"uptime_s\": int(time.time() - _START),
    }


@router.get(\"/profile\")
async def profile() -> dict:
    return {\"profile\": detect_profile(), \"device\": settings.torch_device}


@router.get(\"/modalities\")
async def modalities() -> dict:
    return {
        \"supported\": [\"image\"],
        \"enabled_signals\": {
            \"image\": [
                \"img.prithiv\", \"img.freq\", \"img.clip0\", \"img.meta\",
                \"img.compression\", \"img.ocr_gibberish\", \"img.eye_forensics\",
                \"img.retrieval\", \"img.npr\", \"img.ufd\", \"img.dire\",
                \"img.t15.hive\", \"img.t15.sightengine\",
                \"img.reverse\", \"img.vlm\",
                # v1.5+ additions documented in 05c_v15_addendum.md
                \"img.rup\", \"img.prism\", \"img.clip_centroid\",
                \"img.hf.organika\", \"img.hf.umm_maybe\", \"img.hf.wvolf\",
            ],
        },
    }
```

---

## 11. End-to-end integration test

```python
# file: /app/backend/tests/integration/test_runner_smoke.py
\"\"\"End-to-end smoke: upload → poll → fetch result.

This is the test the testing-agent runs at M1 exit. Requires Mongo + at least
one cloud_lite detector loadable. NO external API calls (no LLM, no SerpAPI,
no Hive/SightEngine) — those are tested separately.\"\"\"
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from httpx import AsyncClient
from PIL import Image

from backend.server import app


@pytest.mark.asyncio
async def test_full_image_job(tmp_path):
    # Build a random image upload
    arr = (np.random.rand(384, 384, 3) * 255).astype(\"uint8\")
    src = tmp_path / \"test.png\"
    Image.fromarray(arr).save(src)
    data = src.read_bytes()

    async with AsyncClient(app=app, base_url=\"http://test\") as ac:
        # POST /analyze
        r = await ac.post(
            \"/api/analyze\",
            files={\"file\": (\"test.png\", data, \"image/png\")},
        )
        assert r.status_code == 200
        job_id = r.json()[\"job_id\"]
        assert job_id

        # Poll until done
        for _ in range(60):                          # 60 × 1 s = 60 s max
            r = await ac.get(f\"/api/jobs/{job_id}\")
            j = r.json()
            if j[\"status\"] in (\"done\", \"failed\"):
                break
            await asyncio.sleep(1.0)

        assert j[\"status\"] == \"done\", f\"job ended {j['status']}: {j.get('error')}\"

        # /result?debug=0
        r = await ac.get(f\"/api/jobs/{job_id}/result\")
        assert r.status_code == 200
        res = r.json()
        assert res[\"job_id\"] == job_id
        assert res[\"verdict\"] in {\"AI-GENERATED\", \"REAL\",
                                    \"INCONCLUSIVE\", \"MANIPULATED\"}
        assert 0.0 <= res[\"p_ai_generated\"] <= 1.0
        assert 0.0 <= res[\"confidence\"] <= 1.0
        assert isinstance(res[\"signals\"], list) and len(res[\"signals\"]) >= 3
        assert \"debug\" not in res

        # /result?debug=1
        r = await ac.get(f\"/api/jobs/{job_id}/result?debug=1\")
        assert r.status_code == 200
        assert \"debug\" in r.json()

        # /assets/frequency.png
        r = await ac.get(f\"/api/jobs/{job_id}/assets/frequency.png\")
        assert r.status_code == 200
        assert r.content[:8] == b\"\x89PNG
\x1a
\"
```

---

## 12. AGENTS.md compliance check

| Standard | Where honoured in this file |
|---|---|
| Modular design, SRP | Each tier in its own coroutine block; `runner._run` is a top-to-bottom flow, no nested side effects |
| Comprehensive error handling | Every external boundary wrapped in try/except + `log.warning`; outer `_run` wrapped in `run_image_job` |
| Performance budgets | `PER_DETECTOR_TIMEOUT_S=25s`, third-party 8s (inherited from §5b), VLM 30s (inherited from §7), retrieval ~5 ms |
| Async/await throughout | All I/O paths are awaitable; CPU work goes through `asyncio.to_thread` inside each detector |
| Structured logging | Every stage logs `event/dur_ms/job_id`; logs.py redacts secrets |
| API design | RESTful: POST /analyze creates, GET /jobs/{id} reads, GET /jobs/{id}/result reads, POST /jobs/{id}/correct mutates, downloadable report.json |
| Data validation | Pydantic models in `schemas/jobs.py` and `schemas/results.py`; `Correction` Pydantic model on the input boundary |

---

## 13. Section exit criteria

```bash
# Unit
pytest backend/tests/unit -q
mypy backend/services/runner.py backend/routes/

# Integration (requires Mongo + at least one image detector)
pytest backend/tests/integration/test_runner_smoke.py -q

# Manual smoke (1-liner curl)
API=\"$(grep REACT_APP_BACKEND_URL /app/frontend/.env | cut -d= -f2)\"
JOB=$(curl -s -F \"file=@/tmp/sample.jpg\" \"$API/api/analyze\" | python3 -c \"import sys,json;print(json.load(sys.stdin)['job_id'])\")
sleep 25
curl -s \"$API/api/jobs/$JOB/result\" | python3 -m json.tool | head -60
```

Expected: a `done`-status job with a `verdict` in the 4-class enum, a non-empty
`signals` list (≥3 entries on cloud_lite), and a `frequency.png` asset of
non-zero size at `/api/jobs/{id}/assets/frequency.png`.

When this passes, **M1 is complete**. Proceed to `11_frontend.md`.

---

## 14. Implementation deltas to record in `/app/memory/PRD.md`

When implementation reaches M1, append:

```text
## Implementation Notes — M1

1. Verdict literal expanded from 3 to 4 classes (added MANIPULATED). All
   schemas/results.py + fusion/types.py updated.
2. `routes/analyze.py` reserves job_id BEFORE store_upload so the upload
   path embeds the same id (avoids a directory rename later).
3. `services/runner.py` is one long function by design — see §2 docstring.
4. `_manipulation_check` is a runner-stage cross-check, not a detector.
   Logic intentionally conservative (meta<0.30 AND freq>0.65 AND comp>0.65).
5. Eye detector's `_eye_score()` extended to return five overlay fields
   (cx, cy, radius_px, highlight_x, highlight_y) — see 09_xai_and_narrator.md §5.
```

Next: `11_frontend.md` — the Control-Room UI: dropzone, result page, history,
correction flow. React 19 + TypeScript strict + Tailwind + shadcn/ui + Recharts
+ Phosphor + Vitest + Playwright + axe-core.
"