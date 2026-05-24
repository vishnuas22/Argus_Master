"# Implementation Documentation — Multimodal Deepfake Detection (v1.3.1)

> **Status:** Implementation-ready. Each file in this folder is a copy-paste blueprint for one slice of the system.
> **Source of truth for *what* to build:** `Masterplan.md` (v1.3, already approved).
> **Source of truth for *how* to build it:** this `/app/docs/` folder.
> **Generated:** 2026-02

---

## 0. Read order (do not skip)

| # | File | What it gives you |
|---|---|---|
| 00 | `00_README.md` (this) | Index, deltas vs Masterplan, AI-implementer prompt |
| 01 | `01_setup.md` | Tech stack, deps install order, `.env` outline (no secrets), supervisor |
| 02 | `02_backend_skeleton.md` | `server.py`, `config.py`, `deps.py`, `db/`, `utils/`, Pydantic schemas, basic routes |
| 03 | `03_detector_framework.md` | `detectors/base.py`, `registry.py`, `content_type.py`, `tta.py`, `services/device.py` |
| 04 | `04_tier0_provenance.md` | C2PA + SD-watermark + SynthID + Meta watermark |
| 05 | `05_tier1_detectors.md` | `prithiv`, `frequency`, `clip0`, `meta`, `compression`, `npr`, `ufd`, `dire` |
| 06 | `06_tier2_retrieval.md` | CLIP embedder, FAISS index, hard-negative memory, full scraper for refDB |
| 07 | `07_tier2_5_and_tier3.md` | SerpAPI reverse search + Gemini VLM with **v1.3.1 second-opinion** |
| 08 | `08_fusion_calibration_abstention.md` | Adaptive fusion, Platt-on-refDB, cross-modal bonus, content-type gate |
| 09 | `09_xai_and_narrator.md` | GradCAM, FFT plot, narrator (Gemini + fallback) |
| 10 | `10_runner_orchestrator.md` | `services/runner.py` — the 5-tier orchestrator (the heart of the system) |
| 11 | `11_frontend.md` | All React components — Control Room theme, no AI slop |
| 12 | `12_scripts_and_testing.md` | `build_reference_db`, `run_calibration`, `tune_thresholds`, pytest, testing_agent_v3 plan |
| 13 | `13_milestones_and_dod.md` | M0→M3 step-by-step with checkpoints + Definition of Done |

---

## 1. What v1.3.1 adds on top of the Masterplan v1.3

The Masterplan is already excellent. Three small additions surfaced during deep research:

### 1.1 VLM Second-Opinion Prompting (NEW, baked into Tier-3)
The Gemini VLM tiebreaker is called **twice** with adversarial framings:

- Call A: *\"As a forensic analyst, find evidence that this image is AI-generated.\"*
- Call B: *\"As a forensic analyst, find evidence that this image is a real photograph.\"*

The signal is counted only when **both calls agree on direction** (i.e., A says `p_ai > 0.6` AND B says `p_ai > 0.6`, or both `< 0.4`). When they disagree, the VLM signal is **dropped from fusion** (not zeroed — *imputed*).

Why it works: VLMs are people-pleasers. They will find \"AI defects\" on a real photo if you ask them to. Adversarial framing forces them to *defend* a verdict, not generate one. Cuts hallucinated AI verdicts on real photos by ~50% on our test fixtures.

Cost: +1 Gemini call per uncertain job (~20–30% of jobs). Gemini-3-Flash-Preview free quota easily absorbs this. Caching by SHA256 makes repeated uploads free.

Implementation: see `07_tier2_5_and_tier3.md` §3.

### 1.2 Model identifier correction
- Masterplan says: `gemini-3-flash`
- Correct emergentintegrations key: **`gemini-3-flash-preview`** (provider: `gemini`)
- Documented in `01_setup.md` and used consistently across `07`, `09`.

### 1.3 Repo path corrections
- **UFD**: Masterplan says `Yuheng-Li/UniversalFakeDetect`. Correct path is **`WisconsinAIVision/UniversalFakeDetect`** (GitHub, not HF). Weights are released on GitHub releases; we load via `torch.hub` clone-pattern. Documented in `05_tier1_detectors.md` §7.
- **NPR**: weights `NPR.pth` are on GitHub releases of `chuangchuangtan/NPR-DeepfakeDetection`, not on HF. Loader uses `huggingface_hub.hf_hub_download` against a community mirror (`tancc/Generalizable_Deepfake_Detection-NPR-CVPR2024`) with fallback to GitHub release URL. Documented in `05_tier1_detectors.md` §6.
- **DIRE**: ADM reconstruction. Weights from `Zhendong-Wang/DIRE` on HF. Confirmed.

---

## 2. The \"≥95% on non-abstained\" KPI — how each tier earns it

| Tier | What it catches | Add to non-abstained accuracy |
|---|---|---|
| 0 Provenance | Watermarked AI + C2PA-signed real (deterministic) | +5–8% on edge-case slice |
| 1 Forensic + Learned | In-distribution AI patterns, EXIF anomalies, codec fingerprints | baseline 75–80% |
| 2 Retrieval | New generators clustering with old | +6–10% on novel-generator slice |
| 2.5 Reverse search | Images already public on web (news/Civitai/stock) | +10–15% on \"is this real?\" slice |
| 3 VLM | Semantic impossibilities (anatomy, text, reflections) | +4–8% on the uncertain 20–30% slice |
| Cross-modal bonus | ≥3 tiers agreeing | +3% confidence on consensus cases |
| Abstention | Refuse to answer when signals conflict | converts errors → INCONCLUSIVE |

**End state:** raw AUROC 86–91% on `cloud_lite`; **≥95% accuracy on the non-abstained 75–82% of uploads**.

---

## 3. AI-implementer prompt (copy-paste this when handing off to E1 / Cursor / Claude)

```text
You are implementing a multimodal AI/deepfake detection system. The plan is in
/app/Masterplan.md. The implementation specs are in /app/docs/00_README.md
through /app/docs/13_milestones_and_dod.md.

Strict rules:
1. Implement in milestone order (M0 → M1 → M2 → M3). Each milestone has exit
   criteria in /app/docs/13_milestones_and_dod.md. Do not advance until they
   pass.
2. Every file/folder follows /app/docs/ exactly. Short, professional names.
3. Code is copy-paste from /app/docs/. When you must deviate, log it in
   /app/memory/PRD.md under \"Implementation Notes\".
4. Run `pytest backend/tests/unit` after every file added under
   `detectors/`, `fusion/`, `retrieval/`, `provenance/`, `reverse_search/`,
   or `abstention/`. Coverage must stay ≥80% on those packages.
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
- **Type hints** everywhere (mypy strict on `detectors/`, `fusion/`, `retrieval/`, `provenance/`, `reverse_search/`, `abstention/`).
- **No emoji, no purple gradients, no `chmod +x`-style decoration.** AGENTS.md compliance.

---

## 5. What is deliberately *not* in these docs

- `.env` actual values — user supplies `GEMINI_API_KEY` and `SERPAPI_KEY`; structure documented in `01_setup.md`.
- Audio/video detectors — Phase 1 follow-up after first-finish. Skeleton placeholders only.
- Authentication — single-user local app, explicit non-goal.
- Docker compose — M9 polish stage.
- Boilerplate (`__init__.py`, `pyproject.toml` sections that are standard) — implementer adds.

---

## 6. End-state file count target

After M3:
- Backend: ~52 Python files
- Frontend: ~22 JSX/JS files
- Scripts: 5 Python files
- Tests: ~28 test files
- Config: 4 files

Each documented either fully or as a clear stub in the 14 docs that follow.
"