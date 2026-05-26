"# 13 — Milestones & Definition of Done

> Goal: a deterministic, step-by-step path from empty repo → first finish.
> Each milestone has **exit criteria** that MUST pass before advancing.
> No advancement on hope; only on green tests.
>
> **First-finish scope:** image modality, all 5 tiers, `cloud_lite` end-to-end,
> v1.4 plan (no new accuracy boosters).

---

## 0. Milestone graph

```
M0  Scaffold + device-detect + skeleton
        │
        ▼
M1  Tier-1 image (cloud_lite) + cold-start calibration
        │
        ▼
M2  Tier-0 Provenance Gate
        │
        ▼
M3  Tier-2 Retrieval + Tier-2.5 Reverse Search + Tier-3 VLM
    + Tier-1.5 third-party APIs + Developer mode + Correction endpoint
        │
        ▼
    testing_agent_v3 E2E suite  →  FIRST FINISH
        │
        ▼
M4  Audio modality            (Phase 1 follow-up)
M5  Video modality            (Phase 1 follow-up)
M6  mac_full / cuda_full validation
M7  Active learning + adaptive fusion auto-promote (LR/GBDT)
M8  Phase 1.5 — text detection (Binoculars)
M9  Polish + docker compose + README
```

---

## 1. M0 — Scaffold + device-detect + skeleton

### 1.1 Deliverables (in order)

1. Dependencies installed per `01_setup.md §2`. `pip freeze > requirements.txt`.
2. Folder tree created per `01_setup.md §4`.
3. Frontend TypeScript migration per `11_frontend.md §2`. `tsc --noEmit` green.
4. `backend/server.py`, `config.py`, `deps.py` per `02_backend_skeleton.md §1–3`.
5. `backend/db/mongo.py` + `repos.py` per `02_backend_skeleton.md §4–5`.
6. `backend/utils/logs.py`, `errors.py`, `retry.py`, `timing.py` per `02_backend_skeleton.md §6`.
7. `backend/schemas/jobs.py`, `results.py` per `02_backend_skeleton.md §7`.
8. `backend/services/device.py` (auto-detect profile) per `03_detector_framework.md §6`.
9. `backend/services/storage.py` + `router.py` (MIME sniff) per `02_backend_skeleton.md §8`.
10. Empty route stubs returning correct envelopes — `/api/health`, `/api/profile`, `/api/modalities`, `/api/analyze`, `/api/jobs/{id}`, `/api/jobs/{id}/result`, `/api/history`, `/api/refdb/stats`, `/api/jobs/{id}/correct` (returns `501 NOT_IMPLEMENTED` until M3).
11. Frontend shells per `11_frontend.md §11–15` — `App.tsx`, three pages, empty result view.
12. `pyproject.toml` per `12_scripts_and_testing.md §8` with ruff + mypy + pytest-cov.
13. Pre-commit hook: `ruff`, `mypy`, `pytest -x`, `cd frontend && yarn lint && yarn tsc --noEmit`.
14. `tests/unit/test_skeleton.py` — one passing test (e.g. `health` endpoint returns 200).

### 1.2 Exit criteria (M0)

| Check | Command | Pass |
|---|---|---|
| Backend boots | `sudo supervisorctl status backend` | RUNNING |
| Frontend boots | `sudo supervisorctl status frontend` | RUNNING |
| Health endpoint | `curl ${URL}/api/health` | `200`, body contains `status:ok`, `profile`, `db_ok` |
| Profile endpoint | `curl ${URL}/api/profile` | `200`, body contains `profile` matching env |
| Lint | `ruff check backend/` | 0 errors |
| Types | `mypy backend/` | 0 errors |
| Frontend types | `cd frontend && yarn tsc --noEmit` | 0 errors |
| Frontend lint | `cd frontend && yarn lint` | 0 errors |
| Unit smoke | `pytest backend/tests/unit -x` | green |
| Frontend smoke | `cd frontend && yarn vitest run` | green |
| Dev page loads | screenshot `/` via `mcp_screenshot_tool` | renders header + empty drop zone |

### 1.3 What is deliberately NOT in M0

- Any detector (M1)
- Any provenance check (M2)
- Retrieval / reverse search / VLM (M3)
- Reference DB (M3)
- Fusion (stub uniform with single signal until M1)
- Correction endpoint (M3)

---

## 2. M1 — Tier-1 image (`cloud_lite`) + cold-start calibration

### 2.1 Deliverables

1. `backend/detectors/base.py` — abstract `Detector` per `03_detector_framework.md §1`.
2. `backend/detectors/registry.py` — `MODELS` table + LRU evict per `03_detector_framework.md §2`.
3. `backend/detectors/content_type.py` — CLIP zero-shot 6-way per `03_detector_framework.md §3`.
4. `backend/detectors/tta.py` — 3-view TTA per `03_detector_framework.md §4`.
5. `backend/detectors/image/prithiv.py` per `05_tier1_detectors.md §2`.
6. `backend/detectors/image/frequency.py` per `05_tier1_detectors.md §3`.
7. `backend/detectors/image/clip0.py` per `05_tier1_detectors.md §4`.
8. `backend/detectors/image/meta.py` per `05_tier1_detectors.md §5`.
9. `backend/detectors/image/compression.py` per `05_tier1_detectors.md §6`.
10. `backend/detectors/image/ocr_gibberish.py` per `05_tier1_detectors.md §7` (v1.4).
11. `backend/detectors/image/eye_forensics.py` per `05_tier1_detectors.md §8` (v1.4, mediapipe guarded).
12. `backend/fusion/fuse.py` — uniform mode + mean imputation per `08_fusion_calibration_abstention.md §2`.
13. `backend/fusion/selector.py` per `08_fusion_calibration_abstention.md §3`.
14. `backend/fusion/calibrate.py` — Platt scaling per `08_fusion_calibration_abstention.md §4`.
15. `backend/abstention/gate.py` — content-type-aware thresholds per `08_fusion_calibration_abstention.md §5`.
16. `backend/xai/heatmap.py` — GradCAM (NPR backbone unavailable on cloud_lite → CLIP attention fallback) per `09_xai_and_narrator.md §1`.
17. `backend/xai/plots.py` — FFT radial PNG per `09_xai_and_narrator.md §2`.
18. `backend/xai/narrator.py` — Gemini + rule-based fallback per `09_xai_and_narrator.md §3`.
19. `backend/services/runner.py` — pipeline up to Tier-1 + cold fusion per `10_runner_orchestrator.md §3`.
20. `/api/analyze` + `/api/jobs/*` wired end-to-end (no Tier-0/2/2.5/3 yet — stubbed as no-ops).
21. Frontend Wave-1 components per `11_frontend.md §21 M1`: `DropZone`, `ProgressSteps`, `VerdictCard`, `ConfidenceAgreementBars`, `NarrativePanel`, `SignalBarChart`, `HeatmapPanel`, `FrequencyPanel`, `MetadataTable`, `HistoryList`, `UploadPage`, `JobPage`, basic `AboutPage`.
22. Cold-start Platt scaling **pre-computed at build time** on a bundled 100-sample mini-eval set under `backend/calibration/samples/`. Generates `backend/fusion/platt.json`. Replaced by real refDB Platt in M3.

### 2.2 Exit criteria (M1)

| Check | Command | Pass |
|---|---|---|
| Real photo fixture → REAL | `curl POST /api/analyze` with `tests/fixtures/img/real_photo.jpg` | verdict `REAL`, `confidence > 0.5` within 30 s on cloud_lite |
| SDXL fixture → AI | same with `tests/fixtures/img/sdxl_*.png` | verdict `AI-GENERATED`, `confidence > 0.5` |
| Result schema | `GET /api/jobs/{id}/result` | matches Pydantic; `signals` ≥ 5; `xai.narrative` non-empty |
| Heatmap asset | `GET /api/jobs/{id}/assets/heatmap.png` | `200`, `image/png` |
| FFT asset | `GET /api/jobs/{id}/assets/fft.png` | `200`, `image/png` |
| Coverage on detectors+fusion | `pytest backend/tests/unit --cov-fail-under=80` | green |
| Frontend renders Wave-1 components | Vitest tests for each | green |
| Axe-core on `/` and `/about` | Playwright | 0 serious/critical |
| ECE in /health | `GET /api/health` | `ece_refdb_holdout` present (initially from mini-eval) |
| Narrator fallback | unset `GEMINI_API_KEY`, re-run | `narrative_source = \"fallback_template\"`, pipeline completes |

### 2.3 Decisions locked at M1

- Detector ordering: `meta → compression` (cheap, deterministic) before `prithiv → freq → clip0` (model loads). Saves ~1 s per job when meta/compression already decisive.
- Content-type classifier runs **after** preprocess but **before** any detector — selfie-gating of `eye_forensics` requires it.
- TTA always 3 views on cloud_lite (original + h-flip + jpeg-recompress q=85). Heavier patch voting is `mac_full`/`cuda_full` only.

---

## 3. M2 — Tier-0 Provenance Gate

### 3.1 Deliverables

1. `backend/provenance/c2pa_check.py` per `04_tier0_provenance.md §1`.
2. `backend/provenance/sd_watermark.py` per `04_tier0_provenance.md §2`.
3. `backend/provenance/synthid_check.py` (guarded import) per `04_tier0_provenance.md §3`.
4. `backend/provenance/meta_watermark.py` (guarded) per `04_tier0_provenance.md §4`.
5. `services/runner.py` patched: provenance gate runs **first**, short-circuits when hit.
6. `schemas/results.py` extended with `provenance` block already declared in M1.
7. Frontend `ProvenanceBadge.tsx` per `11_frontend.md §14.4`.
8. Bundled fixtures:
   - `tests/fixtures/img/c2pa_signed.jpg` (re-signed via `c2patool`)
   - `tests/fixtures/img/sd_watermarked.png` (Stable Diffusion default watermark intact)
   - `tests/fixtures/img/sd_watermark_stripped.png` (same image, watermark scrubbed)

### 3.2 Exit criteria (M2)

| Check | Pass |
|---|---|
| C2PA signed fixture → `verdict=REAL`, `provenance.hit=true`, `provenance.source=\"c2pa\"`, `p_ai=0.99` low | yes |
| SD watermarked fixture → `verdict=AI-GENERATED`, `provenance.source=\"sd_wm\"` | yes |
| Watermark-stripped fixture → falls through to ensemble (no provenance hit) | yes |
| Frontend renders `ProvenanceBadge` when `provenance.hit=true` | yes |
| Abstention is **bypassed** on provenance hit (verdict shown even if low agreement) | yes |
| Ensemble still runs in background for telemetry; `signals[]` populated | yes |
| Coverage on `provenance/` ≥ 80 % | yes |
| `synthid-text` absent does not crash (guarded import returns `hit=false`) | yes |

---

## 4. M3 — Tier-2 retrieval + Tier-2.5 reverse search + Tier-3 VLM + Tier-1.5 third-party + Dev mode + Correction

This is the largest milestone. Split into 4 sub-steps that can be tested
independently.

### 4.1 M3.1 — Reference DB + Tier-2 retrieval

1. `backend/retrieval/embedder.py` per `06_tier2_retrieval.md §2`.
2. `backend/retrieval/index.py` (FAISS load/query/dedup) per `06_tier2_retrieval.md §3`.
3. `backend/retrieval/hard_negatives.py` per `06_tier2_retrieval.md §4`.
4. `backend/retrieval/sources/*.py` per `06_tier2_retrieval.md §5` + `12_scripts_and_testing.md §3.3–3.6`.
5. `backend/scripts/build_reference_db.py` per `12_scripts_and_testing.md §3.2`.
6. `backend/scripts/run_calibration.py` per `12_scripts_and_testing.md §4`.
7. **Build the refDB once locally**: `python -m backend.scripts.build_reference_db --target-real 5000 --target-ai 5000` (~10 hours). Saves `image_real.npy`, `image_ai.npy`, `image_real.index`, `image_ai.index`, `image_*_sources.json`.
8. Re-run `run_calibration.py` to regenerate `platt.json` on refDB (replaces M1 mini-eval).
9. Patch `services/runner.py` to call retrieval after Tier-1.
10. Frontend `RetrievalNeighborsPanel.tsx` per `11_frontend.md §14.12`.
11. `/api/refdb/stats` returns real counts; `/api/refdb/thumb/{id}.jpg` serves thumbnails.

**Exit criteria M3.1:**
- `image_real.index` and `image_ai.index` exist; size ≥ 4900 each (allow 2 % dedup loss).
- `GET /api/refdb/stats` returns counts and `auroc_retrieval_alone` ≥ 0.75 on internal holdout.
- Pipeline result includes `retrieval.neighbors[]` (5 entries with thumbs).
- SHA self-leak guard tested: upload one image already in refDB → `retrieval` signal excluded, debug note shown.
- `ece_refdb_holdout` < 0.10 in `/api/health`.

### 4.2 M3.2 — Tier-2.5 reverse search (SerpAPI)

1. `backend/reverse_search/serpapi_client.py` per `07_tier2_5_and_tier3.md §1`.
2. `backend/reverse_search/interpreter.py` per `07_tier2_5_and_tier3.md §2` (matches Appendix D in Masterplan).
3. `backend/reverse_search/cache.py` (24 h Mongo TTL) per `07_tier2_5_and_tier3.md §3`.
4. `services/runner.py`: invoke reverse search **only when** `extremity < 0.30 OR agreement < 0.70`.
5. Frontend `ReverseSearchBadge.tsx` + `ReverseSearchPanel.tsx`.

**Exit criteria M3.2:**
- Real Reuters fixture (camera EXIF, pre-2022 indexed online) → reverse hit with `pre_ai_era_news` reason → `p_fake ≈ 0.07` → boosts REAL verdict.
- Civitai-hosted fixture → `ai_gallery_hit` reason → `p_fake ≈ 0.93`.
- Cache HIT on re-upload returns identical reverse output with zero SerpAPI calls (verified via mocked API call counter).
- Missing `SERPAPI_KEY` → signal silently dropped; pipeline completes.
- Gate respected: high-extremity uploads do NOT invoke SerpAPI (verified by log assertion).

### 4.3 M3.3 — Tier-1.5 third-party APIs (Hive / SightEngine / AI-or-Not)

> Optional but ships with M3 per `05b_tier1_5_third_party.md`. Each absent
> key = signal absent (mean-imputed). Quota-gated by `extremity < 0.30`.

1. `backend/third_party/hive.py` per `05b_tier1_5_third_party.md §2`.
2. `backend/third_party/sightengine.py` per `05b_tier1_5_third_party.md §3`.
3. `backend/third_party/aiornot.py` per `05b_tier1_5_third_party.md §4`.
4. `services/runner.py`: invokes all three **in parallel** with 8 s per-call timeout, after Tier-2 fusion, before Tier-2.5.
5. Each result appended as its own row in `signals[]` (`tp.hive`, `tp.sightengine`, `tp.aiornot`).

**Exit criteria M3.3:**
- Each provider mocked with respx → integration test green.
- All three keys absent → signals silently absent; pipeline completes.
- Per-call timeout enforced (8 s).

### 4.4 M3.4 — Tier-3 VLM + Developer mode + Correction + OOD novel-generator

1. `backend/vlm/judge.py` per `07_tier2_5_and_tier3.md §4` — Gemini 3 Flash vision via `emergentintegrations`.
2. `backend/vlm/prompts.py` per `07_tier2_5_and_tier3.md §5`.
3. **Counter-prompt second opinion** when `ENABLE_VLM_SECOND_OPINION=true` per `07_tier2_5_and_tier3.md §6` (v1.3.1 — already in v1.4 plan).
4. `services/runner.py`: invoke VLM **only when** `extremity < 0.25 OR agreement < 0.63`. Caps Gemini calls.
5. `backend/retrieval/ood_isolation.py` per `08_fusion_calibration_abstention.md §7` (v1.4) — IsolationForest at refDB build time saved to `backend/retrieval/ood_real.pkl` and `ood_ai.pkl`; loaded at runtime.
6. Patch `services/runner.py` to set `novel_generator_suspected=true` and force `verdict=INCONCLUSIVE` when OOD on both clusters.
7. `backend/fusion/crossmodal_bonus.py` per `08_fusion_calibration_abstention.md §6`.
8. `backend/routes/correct.py` — `POST /api/jobs/{id}/correct` per `02_backend_skeleton.md §10`.
9. `backend/retrieval/hard_negatives.py`: on correction, append upload's embedding to hard partition + hot-reload FAISS.
10. Frontend Wave-3 components per `11_frontend.md §21 M3`: `VLMBadge`, `VLMRationalePanel`, `ContentTypeBadge`, `CompressionFingerprintPanel`, `CorrectVerdictBar`, `DeveloperPanel`. Final `AboutPage` health surface.
11. Backend `debug=1` query param surfaces `DebugBlock` with raw signals, fusion vector, gate states, full retrieval list.

**Exit criteria M3.4:**
- VLM-uncertain fixture (deliberately ambiguous SD-realistic photo) triggers VLM → rationale rendered + `vlm_invoked=true`.
- Counter-prompt: same fixture invoked with `ENABLE_VLM_SECOND_OPINION=true` → if rationales disagree, signal weight halved or dropped (verified via debug panel weight).
- Confident verdict (provenance hit OR `extremity > 0.80`) → VLM NOT invoked (gate test).
- Missing `GEMINI_API_KEY` → VLM signal absent; pipeline completes.
- OOD fixture (an upload from a deliberately-novel generator) → `novel_generator_suspected=true` AND verdict forced to `INCONCLUSIVE`.
- `POST /api/jobs/{id}/correct` increments `refdb_size.image_real_hard` (or `_ai_hard`) by 1; next identical-ish upload's retrieval shows the new neighbor (within 1 s — verified via integration test).
- Frontend dev mode hotkey toggles panel; threshold sliders re-render verdict client-side without network call.

---

## 5. M3 final gate — testing_agent_v3

After M3.1–M3.4 are individually green, run `testing_agent_v3` once with
the JSON payload from `12_scripts_and_testing.md §11`.

### 5.1 Pass criteria

- Zero **high-severity** blockers in the report.
- All listed `features_or_bugs_to_test` items pass.
- Axe-core: zero serious/critical violations on `/` and `/about`.
- Mobile Safari project: navigable, dropzone tappable, no overflow.

### 5.2 If blockers

For each high-severity issue:
1. Read `/app/test_reports/iteration_{n}.json` carefully.
2. Reproduce locally with the same fixture.
3. Fix at the root cause (no patches around the symptom).
4. Add a regression test.
5. Re-run the relevant unit/integration test → green.
6. Re-run `testing_agent_v3`.

Do not advance to \"first finish\" until the report is clean.

---

## 6. Definition of Done — Phase 1 First-Finish

Copied from `Masterplan.md §22` and extended with v1.4 additions. Every
box must be ticked.

**Pipeline correctness**
- [ ] All `/api/*` endpoints in `Masterplan.md §12` work.
- [ ] Image modality produces verdicts + XAI in `cloud_lite` on bundled fixtures within 30 s.
- [ ] Tier-0 short-circuit verified on C2PA + SD-watermark fixtures.
- [ ] Tier-1.5 third-party signals appear when keys set; absent gracefully when not.
- [ ] Tier-2 reference DB built at 5000+5000; AUROC-alone ≥ 0.78 on holdout.
- [ ] Tier-2 patch-level retrieval contributes; composite-fake fixture catches.
- [ ] Tier-2.5 reverse search gated + cached + falls back when key absent.
- [ ] Tier-3 VLM gated + counter-prompt mode tested; absent gracefully when no key.
- [ ] OCR-gibberish + eye-forensics signals contribute and are unit-tested.
- [ ] Compression-forensics scores match expected ranges on PNG/JPEG fixtures.
- [ ] Content-type router: 6 types classified; type-specific thresholds applied.
- [ ] Cold-start Platt scaling: loaded from refDB; `ece_refdb_holdout < 0.10`.
- [ ] OOD novel-generator detector: forces INCONCLUSIVE on cluster-anomalous uploads.
- [ ] Cross-modal multiplicative bonus applied when ≥3 tiers agree; capped at +0.10.
- [ ] Hard-negative append + reindex on `POST /jobs/{id}/correct` works and survives restart.
- [ ] Adaptive fusion: uniform mode active at n<100; LR mode unit-tested on synthetic data.

**Frontend**
- [ ] All routes (`/`, `/job/:id`, `/about`) render.
- [ ] All `data-testid` from `11_frontend.md §16` present in DOM.
- [ ] Developer panel reveals raw signals + working threshold sliders.
- [ ] Three verdict states visually distinct; cyan brand; no purple gradients; no emoji.
- [ ] Phosphor icons rendering; IBM Plex Sans / Inter / JetBrains Mono loaded from Google Fonts.
- [ ] Mobile Safari project navigable; touch targets ≥ 44 px.
- [ ] Reduce-motion preference respected.

**Quality gates**
- [ ] `testing_agent_v3` full image E2E run passes.
- [ ] `ruff check backend/` clean.
- [ ] `mypy backend/` clean.
- [ ] `pytest backend/tests` coverage ≥ 80 % on `detectors/`, `fusion/`, `retrieval/`, `provenance/`, `reverse_search/`, `abstention/`, `third_party/`.
- [ ] `cd frontend && yarn tsc --noEmit` clean.
- [ ] `cd frontend && yarn lint` clean.
- [ ] `cd frontend && yarn vitest run --coverage` ≥ 80 % lines / statements / functions.
- [ ] `cd frontend && yarn playwright test` green incl. mobile Safari project.
- [ ] `axe-core` zero serious/critical on `/` and `/about`.
- [ ] `python -m backend.scripts.verify_registry --profile cloud_lite` green.
- [ ] `python -m backend.scripts.license_audit` green.

**AGENTS.md compliance**
- [ ] AGENTS.md naming: every file short + professional; verified by grep against banned patterns.
- [ ] Type hints on all public signatures; mypy strict on guarded packages.
- [ ] Structured JSON logging one-line-per-event from `utils/logs.py`; redacts secrets.
- [ ] Every external call has explicit `asyncio.wait_for` timeout per `Masterplan.md §11.2`.
- [ ] Single error envelope shape across all endpoints.
- [ ] No hardcoded secrets; `.env.example` documented.
- [ ] Graceful degradation: removing each of GEMINI/SERPAPI/HIVE/SE/AIORNOT keys never crashes the pipeline.
- [ ] Retry with exponential backoff on Gemini/SerpAPI/HF/third-party APIs.
- [ ] Idempotency: SHA-keyed SerpAPI cache; identical re-uploads return identical results.

**AGENTS_FRONTEND.md compliance (P0)**
- [ ] TypeScript strict; zero `any` without inline justification.
- [ ] Initial bundle < 200 KB gz; route-split confirmed.
- [ ] LCP < 2.5 s on `/` (Lighthouse-CI report committed).
- [ ] WCAG 2.1 AA: keyboard nav, ARIA labels, ≥ 4.5:1 contrast, visible focus ring.
- [ ] Every component has loading, error, empty states (verified by Vitest).

**Documentation**
- [ ] README with setup, profile-switching, refDB build instructions, key acquisition URLs.
- [ ] `licenses.txt` committed (output of `license_audit.py`).
- [ ] `calibration/report.md` committed.
- [ ] `/app/memory/PRD.md` updated with implementation notes and any documented deviations.

---

## 7. Phase 1 follow-up (after first-finish, before \"v1 done\")

Same plan, applied to audio and video:

### 7.1 M4 — Audio modality

1. `detectors/audio/w2v2df.py` (wav2vec2-deepfake) per `Masterplan §11 registry`.
2. `detectors/audio/spectral.py` — log-mel + bicoherence forensics.
3. `detectors/audio/prosody.py` — pitch range, jitter, shimmer (training-free).
4. `detectors/audio/aasist3.py` — AASIST3 anti-spoofing.
5. `retrieval/embedder.py` extended for `embed.wavlm` (WavLM-base-plus, MIT).
6. XAI: spectrogram PNG, prosody table.
7. Audio fixtures: real voice clip, ElevenLabs clone, Tortoise TTS, generic AI music.

### 7.2 M5 — Video modality

1. Frame sampler (8 frames cap on cloud_lite, 32 on mac/cuda).
2. `detectors/video/img_ens.py` — per-frame image ensemble.
3. `detectors/video/flicker.py` — temporal high-freq.
4. `detectors/video/syncnet.py` — lip-sync mismatch.
5. `detectors/video/blink.py` — blink rate / asymmetry.
6. `detectors/video/identity.py` — ArcFace identity drift.
7. Audio track of video routed through M4 audio pipeline.
8. XAI: frame timeline strip; per-frame heatmap subset.

### 7.3 M6 — `mac_full` + `cuda_full` validation

1. `detectors/image/npr.py`, `ufd.py`, `dire.py` per `05_tier1_detectors.md §9–11`.
2. `detectors/image/dire.py` on CPU on Mac (MPS fallback unstable) — toggle via `ENABLE_DIRE_MPS`.
3. Patch-voting on >512² images.
4. RetinaFace face detector (insightface, guarded).
5. VRAM benchmarks on RTX 3050 4 GB — stage table in `Masterplan.md §11.1` validated by `pytest -m gpu`.

### 7.4 M7 — Active learning + adaptive fusion auto-promote

1. `tune_thresholds.py` scheduled hourly (cron in supervisor).
2. `selector.py` auto-promotes uniform → `lr_l2` at `n_user_labels ≥ 100`, → `gbdt` at `≥ 500`.
3. LR/GBDT weight files hot-reloaded on mtime change (no restart).
4. Isotonic per-signal calibration at `n ≥ 500`.

### 7.5 M8 — Phase 1.5

1. Text modality: Binoculars perplexity-ratio + GLTR (no API cost).
2. `PARTIALLY-MODIFIED` verdict once enough INCONCLUSIVE samples accumulate to train a 3-class fusion head.
3. PDF report export.
4. Active-learning UI (\"was this real or AI?\" non-blocking sidebar).

### 7.6 M9 — Polish + docker compose + README

1. `docker-compose.yml` runs cloud_lite profile on a fresh machine.
2. README rewritten for first-time install: 3 commands to running app.
3. Optional CI workflow on GitHub Actions (lint + typecheck + unit + integration on cloud_lite).
4. Demo screenshot pack committed.

---

## 8. Risk register (M0→M3 scope only)

| Risk | M | Mitigation |
|---|---|---|
| Civitai / Lexica rate-limit during refDB build | M | Per-source `asyncio.sleep(0.2)`; resumable build with SHA dedup |
| 5000+5000 build takes >10 h on slow connection | M | Build accepts `--target-real`/`--target-ai`; can ship at 2000+2000 and grow |
| Mediapipe wheel unavailable on the runtime Python | L | `eye_forensics.py` already guarded — silently skips signal |
| Gemini API rate-limit | M | Cap via `vlm_invoked` gate; narrator falls back to rule-based templates |
| SerpAPI 100-call free quota burned fast | M | 24 h SHA cache + extremity/agreement gate; signal drops on `SERPAPI_QUOTA` |
| Hive / SightEngine / AI-or-Not provider takes >8 s | L | Per-call timeout 8 s; signal dropped on timeout |
| FAISS index size grows beyond cloud_lite RAM | L | `IndexFlatIP` at 10 K × 512-d = 20 MB — trivially small |
| Frontend TypeScript migration breaks craco | M | `tsconfig.json` tested in M0 exit; rollback path = revert renames |
| Coverage gate (80 %) blocks merge on a slow Tuesday | L | Coverage is on guarded packages only; non-critical scripts excluded in `pyproject.toml` |
| Reference DB AUROC < 0.78 holdout | M | Expand to 7500+7500 via `--target` flags; verify after each +500 batch |

---

## 9. Sequencing summary (one-line plan)

```
M0 (1 day) → M1 (3 days) → M2 (1 day) → M3.1 refDB build (overnight)
   → M3.2 reverse (1 day) → M3.3 third-party (1 day)
   → M3.4 VLM+dev+correction (2 days) → testing_agent_v3 → first finish.
```

Total: ~10 working days assuming steady pace and no integration surprises.

---

## 10. AGENTS.md mapping for this file

| Standard | Where honored |
|---|---|
| Documentation: README + setup (§9) | §6 DoD entry; covered by `01_setup.md` |
| Test-driven development (§5) | Every milestone has explicit exit criteria + pytest gate |
| Coverage > 80 % (§5) | §6 DoD entry + `12_scripts_and_testing.md §8` |
| Modular design / SRP (§2) | One milestone per concern; one detector per file |
| Idempotency (§11) | §6 DoD entry; SHA cache + resumable refDB |
| Retry / backoff (§11) | §6 DoD entry; `utils/retry.py` |
| Performance budgets (AGENTS_FRONTEND §9) | §6 DoD frontend block |
| Accessibility WCAG 2.1 AA (AGENTS_FRONTEND §10) | §6 DoD frontend block + Playwright axe-core gate |
| ADRs (§9) | Each \"(v1.3)\" / \"(v1.4)\" addition in `Masterplan.md` and these docs is the ADR |

---

End of `13_milestones_and_dod.md`. Source of truth for the milestone path
and Phase 1 first-finish exit conditions. Once every DoD checkbox in §6 is
ticked, declare first finish in `/app/memory/PRD.md` and hand back to the
user for review.
"