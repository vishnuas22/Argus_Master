"# Implementation Documentation — Multimodal Deepfake Detection (v1.4)

> **Status:** Implementation-ready. Each file in this folder is a copy-paste blueprint for one slice of the system.
> **Source of truth for *what* to build:** `Masterplan.md` (v1.3, approved).
> **Source of truth for *how* to build it:** this `/app/docs/` folder (v1.4).
> **Last updated:** 2026-02

---

## 0. Read order (do not skip)

| # | File | What it gives you |
|---|---|---|
| 00 | `00_README.md` (this) | Index, deltas vs Masterplan, AI-implementer prompt |
| 01 | `01_setup.md` | Tech stack, deps install order, `.env` outline (no secrets), supervisor |
| 02 | `02_backend_skeleton.md` | `server.py`, `config.py`, `deps.py`, `db/`, `utils/`, Pydantic schemas, basic routes |
| 03 | `03_detector_framework.md` | `detectors/base.py`, `registry.py`, `content_type.py`, `tta.py`, `services/device.py` |
| 04 | `04_tier0_provenance.md` | C2PA + SD-watermark + SynthID + Meta watermark |
| 05 | `05_tier1_detectors.md` | `prithiv`, `frequency`, `clip0`, `meta`, `compression`, `npr`, `ufd`, `dire`, **`ocr_gibberish`** (NEW v1.4), **`eye_forensics`** (NEW v1.4) |
| 05b | `05b_tier1_5_third_party.md` | **NEW v1.4 — Hive / SightEngine / AI-or-Not ensemble members** |
| 06 | `06_tier2_retrieval.md` | CLIP embedder, FAISS index, hard-negative memory, hybrid scraper + curated dataset builder, **patch-level retrieval** (NEW v1.4) |
| 07 | `07_tier2_5_and_tier3.md` | SerpAPI reverse search + Gemini VLM with v1.3.1 second-opinion |
| 08 | `08_fusion_calibration_abstention.md` | Platt-on-refDB, adaptive fusion, cross-modal bonus, content-type gate, **OOD novel-generator override** (NEW v1.4) |
| 09 | `09_xai_and_narrator.md` | GradCAM, FFT plot, narrator (Gemini + rule-based fallback) |
| 10 | `10_runner_orchestrator.md` | `services/runner.py` — the 5-tier orchestrator (heart of the system) |
| 11 | `11_frontend.md` | All React 19 + TypeScript components, Control Room theme, no AI slop |
| 12 | `12_scripts_and_testing.md` | `build_reference_db`, `run_calibration`, `tune_thresholds`, pytest, testing_agent_v3 plan |
| 13 | `13_milestones_and_dod.md` | M0→M3 step-by-step with checkpoints + Definition of Done |


 **Implementer note (2026-02 patch):** the cross-document schema deltas that
> previously lived in `10_runner_orchestrator.md §1` have been **consolidated
> into `02_backend_skeleton.md §7` and `08_fusion_calibration_abstention.md §1`**.
> You can now implement each file top-to-bottom in the order below without
> holding deltas in memory. The `MANIPULATED` verdict, the `third_party` field,
> and the `Verdict.label` literal are already final in those two files.

---

## 1. What v1.4 adds on top of v1.3.1

Five free, training-free accuracy boosters, plus a frontend stack lock-in.

### 1.1 Tier-1 additions (image)

#### A. `img.ocr_gibberish` (NEW)
Tesseract reads any text on the image; words are dictionary-checked. Diffusion models still produce gibberish on signs, plates, watermarks, captions. When ≥40 % of detected tokens are non-dictionary AND total token count ≥ 4, `p_fake = 0.85`. Otherwise neutral. Very high *precision*, low recall — perfect ensemble citizen.

#### B. `img.eye_forensics` (NEW, content-type-gated)
Only runs when `content_type == \"selfie_portrait\"`. Three measurements:
1. Pupil circularity (real eyes are near-circular when frontal-facing)
2. Iris-boundary regularity (real irises have monotone-radius boundaries)
3. Left-vs-right eye highlight asymmetry (real photos have a single dominant light source → highlights align)

Uses `mediapipe` FaceMesh landmarks. Training-free. Outputs `p_fake` + per-eye debug overlay.

### 1.2 Tier-1.5: third-party detector APIs (NEW)

A new tier between Tier 1 and Tier 2. Each provider is **another orthogonal ensemble member** that has seen training data we cannot afford to gather:

| Provider | Free quota | Direction signal |
|---|---|---|
| Hive Moderation | 1k calls/mo | `p_ai` 0–1 + per-class breakdown |
| SightEngine | 2k calls/mo | `p_ai_generated` 0–1 |
| AI-or-Not | 100 calls/mo | binary `ai|real` + confidence |

All three calls run **in parallel**, each with 8 s timeout. Each is its own signal in the fusion vector; any combination of them may be absent (mean-imputed). Gated by `extremity < 0.30` to conserve quota.

Cost: $0/month within free tiers. Documented in `05b_tier1_5_third_party.md`.

### 1.3 Tier-2 enhancement: patch-level retrieval (NEW)

The existing full-image retrieval misses composite fakes (\"AI background, real face\"). We index **4 patches per refDB image** (4 corners) alongside the full-image index. Query time: full-image k=15 + per-patch k=8 → combined with a max-similarity-rule. Adds <50MB total index size, <20 ms query latency.

Implementation appended to `06_tier2_retrieval.md` §9.

### 1.4 Tier-3 / fusion override: OOD novel-generator detector (NEW)

A 2-class **Isolation Forest** trained on CLIP embeddings of the refDB at build time (one IF per cluster: `real`, `ai`). On every upload, compute:

```
ood_real = isolation_forest_real.score(vec)
ood_ai   = isolation_forest_ai.score(vec)
```

When the upload is anomalous to **both** clusters (`ood_real > τ AND ood_ai > τ`) → set `novel_generator_suspected = true`, force `verdict = INCONCLUSIVE` with narrative *\"This image looks unlike anything in our reference DB — it may come from a new generator we haven't catalogued. Manual review recommended.\"*

This converts confident-wrong on novel generators into honest-abstention. Documented in `08_fusion_calibration_abstention.md` §7.

### 1.5 Frontend stack lock-in (resolves Masterplan §14 vs AGENTS_FRONTEND.md conflict)

**Locked:** React 19 + **TypeScript (strict)** + Tailwind CSS + shadcn/ui + Recharts + Phosphor icons + Vitest + React Testing Library + Playwright + axe-core.

Honors AGENTS_FRONTEND.md P0 rules (TypeScript strict, ≥80 % test coverage, axe-core a11y, no `any` types) without the Next.js migration cost. Tooling enforcement table is in `11_frontend.md` §3.

---

## 2. Updated KPI math with v1.4 additions

| Tier / signal | `cloud_lite` AUROC lift | Notes |
|---|---|---|
| Tier 0 baseline | +5–8 % on edge slice | deterministic |
| Tier 1 (5 base) | 75–80 % baseline | the core |
| **Tier 1 + OCR + eyes** | +1–2 % | high-precision adds |
| **Tier 1.5 third-party** | +2–3 % | private ensembles |
| Tier 2 retrieval | +6–10 % on novel-generator slice | similarity |
| **Tier 2 + patch retrieval** | +1 % overall, +5 % on composites | composite catch |
| Tier 2.5 reverse search | +10–15 % on \"is this real?\" slice | web priors |
| Tier 3 VLM (+ 2nd opinion) | +4–8 % on uncertain slice | semantic |
| Cross-modal bonus | +3 % confidence on consensus | super-additive |
| **OOD warning** | converts ~3 % errors → INCONCLUSIVE | honest abstention |

**Updated end-state targets:**
- `cloud_lite` raw AUROC: **89–93 %** (was 86–91 %)
- `mac_full` raw AUROC: **94–97 %** (was 92–96 %)
- **≥97 % accuracy on the non-abstained 75–82 % of uploads**

---

## 3. AI-implementer prompt (copy-paste this when handing off)

```text
You are implementing a multimodal AI/deepfake detection system. The plan is in
/app/Masterplan.md (v1.3 — what). The implementation specs are in /app/docs/
00_README.md through /app/docs/13_milestones_and_dod.md (v1.4 — how).

Strict rules:
1. Implement in milestone order (M0 → M1 → M2 → M3). Each milestone has exit
   criteria in /app/docs/13_milestones_and_dod.md. Do not advance until they
   pass.
2. Every file/folder follows /app/docs/ exactly. Short, professional names
   (AGENTS.md naming rule).
3. Code is copy-paste from /app/docs/. When you must deviate, log it in
   /app/memory/PRD.md under \"Implementation Notes\".
4. Run `pytest backend/tests/unit` after every file added under
   detectors/, fusion/, retrieval/, provenance/, reverse_search/, abstention/,
   third_party/. Coverage must stay ≥80% on those packages.
5. Before declaring M3 done, run testing_agent_v3 with the test plan in
   /app/docs/12_scripts_and_testing.md §6.
6. After M3, write a 5-line summary to /app/memory/PRD.md and stop. User reviews.

Begin with M0.
```

---

## 4. Conventions used across these docs

- **Code block headers:** `# file: path/from/app/root.py` — the exact file path the block belongs to. If absent, the block is illustrative.
- **`# TODO(M*):`** comments mark deliberate gaps for later milestones. They are *not* bugs.
- **`# NOTE:`** comments explain a non-obvious design choice. Keep them; they replace docstrings on tricky logic.
- **Imports** at top of each block are exhaustive — paste once, no second guessing.
- **Type hints** everywhere (mypy strict on `detectors/`, `fusion/`, `retrieval/`, `provenance/`, `reverse_search/`, `abstention/`, `third_party/`).
- **No emoji, no purple gradients, no decorative names.** AGENTS.md compliance.

---

## 5. What is deliberately *not* in these docs

- `.env` actual values — user supplies `GEMINI_API_KEY`, `SERPAPI_KEY`, optional `HIVE_API_KEY`, `SIGHTENGINE_USER` + `SIGHTENGINE_SECRET`, `AIORNOT_API_KEY`; structure documented in `01_setup.md`.
- Audio/video detectors — Phase 1 follow-up after first-finish. Skeleton placeholders only.
- Authentication — single-user local app, explicit non-goal.
- Docker compose — M9 polish stage.
- Boilerplate (`__init__.py`, `pyproject.toml` sections that are standard) — implementer adds.

---

## 6. End-state file count target (after M3, v1.4)

- Backend: ~58 Python files (+6 from v1.3.1 for OCR, eyes, 3 third-party clients, OOD)
- Frontend: ~24 TSX files
- Scripts: 6 Python files
- Tests: ~34 test files
- Config: 4 files

Each is documented either fully or as a clear stub in the 14 docs that follow.
"