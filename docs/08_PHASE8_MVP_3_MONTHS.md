# PHASE 8 — MVP: The Smallest ARGUS Buildable in 3 Months

> Scope discipline: one engineer-equivalent ×12 weeks, commodity hardware (8-core CPU, 32 GB RAM, no GPU required), open-source only, no proprietary data, no foundation-model fine-tuning. Everything below is buildable today with named, existing artifacts.

---

## 8.1 MVP Scope Cuts (decided up front, not discovered in week 10)

| In MVP | Deferred (post-MVP) |
|---|---|
| Tier 0 provenance (C2PA verify) | Trust-list governance UI |
| Tier 1 triage: JPEG quality + recompression count + resize + screenshot heuristic | Denoise estimation; AVIF/HEIC pipelines |
| Modules A, B, C, D, E, F | Module I (retrieval — needs index infra) |
| Module G *narrow*: shadow-line light-direction consistency only | Reflection geometry; reflectance models |
| Module H *narrow*: typography (OCR + lexicon) only | Hand topology, object-logic |
| Full fusion stack (gate → LightGBM → isotonic → conformal) | Online drift dashboard (telemetry logged from day 1, dashboard later) |
| Verdict JSON + analyst web UI + overlays | Journalist/layperson renderers; LLM polish |
| Eval harness: LOGO × laundering ladder | Continuous benchmark service |

## 8.2 Exact Modules, Models, Libraries

### Stack baseline
`Python 3.11 · PyTorch (CPU) · timm · OpenCV · Pillow · numpy/scipy · scikit-learn · LightGBM · MAPIE (conformal) · shap · pyexiftool (+ exiftool) · c2pa-python · jpegio (DCT access) · Tesseract/pytesseract (OCR) · FastAPI backend · React frontend`

### Tier 0 — Provenance
- `c2pa-python` (official CAI SDK bindings): manifest parse, signature verify, chain report. Output: valid / invalid / absent + signer identity. **~1 day of work.**

### Tier 1 — Degradation triage
- **JPEG quality:** read quant tables (`jpegio` / Pillow `quantization`); map to libjpeg-quality scale; compare against camera-vendor table corpus (public corpora exist).
- **Recompression generations:** DCT-coefficient first-digit/histogram periodicity (double-quantization) — classical algorithms, pure numpy, well-documented in the forensics literature.
- **Resize detection:** Gallagher-style resampling peak detection on the residual spectrum.
- **Screenshot heuristic:** aspect-ratio table + uniform-border detection + absence-of-camera-metadata + quant-table fingerprints of OS screenshot encoders; (small CNN upgrade post-MVP).
- Output `d` + uncertainty bands. **~1.5 weeks.**

### Evidence panel
| Module | Exact implementation | Weights/training needed |
|---|---|---|
| A. Metadata | `pyexiftool` full dump → rule pack (≈40 hand-written checks: software strings, table-vs-claimed-camera match, thumbnail consistency, date logic) | none |
| B. Compression history | ELA (recompress@q90, amplify diff), JPEG-ghost sweep (q 50–100), DQ localization map — all OpenCV/numpy | none |
| C. Spectral probe | FFT → radial power spectrum → peak detection + spectral-slope features, calibrated against real-corpus envelope (SpAN-style power calibration); resize disambiguation via `d` | envelope fit on real corpus only |
| D. Noise residual | **TruFor** (GRIP-UNINA, public weights) run as packaged module: anomaly map + its native confidence map + global score. *License check week 1: research-only → MVP flag as "research mode"; commercial path = retrain Noiseprint-style net on open data (documented contingency, ~3 GPU-days on rented hardware, post-MVP)* | pretrained |
| E. Real-distribution probe | **DINOv2 ViT-B/14** via `timm`, frozen, CPU. Reference index: FAISS over ~1.2 M real-image embeddings. Score: kNN distance percentile vs real-corpus calibration; report 5 nearest real exemplars | embedding pass over real corpus (CPU-days, one-off, parallelizable) |
| F. Perturbation-sensitivity probe | RIGID-style: cosine similarity drop between DINOv2 embeddings of x vs x+structured-noise (and blur variant); shares E's backbone — marginal cost = 1–2 extra forward passes | threshold calibration on real corpus only |
| G. Physics (narrow) | Segment-anything-lite or threshold-based shadow extraction → object-shadow line construction → light-azimuth consistency test (classical Kee/Farid-style projective constraints, OpenCV). *Fires only when ≥2 clear cast shadows detected; else `unavailable`* | none |
| H. Typography (narrow) | Tesseract OCR → confidence-weighted tokens → lexicon/character-validity check (garbled-glyph score) | none |

### Fusion + verdict
- Gate (`r_m(d)` lookup tables from §8.4 calibration) → **LightGBM** 3-class (≤200 trees, depth 4, monotonic constraints) → isotonic (sklearn) → **MAPIE** split-conformal (α user-set) → verdict builder (pydantic schema of Phase 7) + SHAP ranking.
- Analyst UI: React; image canvas with toggleable overlay layers; evidence cards; JSON download.

## 8.3 Dataset Strategy (open only)

| Role | Sources | Notes |
|---|---|---|
| Real corpus (module E index + envelopes + reliability curves) | **COCO** (web-realistic), **RAISE** (RAW/high-quality), **Dresden** (camera diversity), **OpenImages subset**, FFHQ subset (faces) | target ~1.2 M images; diversity over volume — phones, DSLRs, scans, screenshots-of-real |
| Fake corpus (fusion training + eval ONLY — never trains evidence modules E/F/C) | **GenImage** (8 generators), **Synthbuster** (incl. recent commercial generators), **DF40** subset; self-generated **SDXL + FLUX.1-dev** locally (license-checked) for fresh in-house fakes | ~50–100 k; *de-confound first* (next row) |
| **De-confounding pass (critical)** | Re-encode ALL training/eval images through identical pipeline mixes (JPEG q-mix, resize-mix) so format/quality never predicts label | This single step invalidates the GenImage JPEG/PNG confound that inflates most published numbers (Phase 10.2) |
| Manipulated class | **CASIA v2**, **Columbia**, **IMD2020**, DEFACTO subset; self-made inpainting set (SD-inpaint on COCO, masks logged) | for the 3rd hypothesis + module B validation |
| Laundering ladder | Deterministic simulator: JPEG(q∈{30..95}) × resize(0.25–1.0) × screenshot-sim × measured platform presets (round-trip a probe set through WhatsApp/Telegram/X manually once, fit presets) | versioned artifact `launder-v1`; used for reliability curves AND eval |

## 8.4 Calibration & Training Procedure (order matters)

1. Build real corpus index + spectral envelopes + probe thresholds (modules C/E/F need only real data).
2. Run full panel over (real ∪ fake ∪ manipulated) × laundering ladder → ~1–2 M module-output records (CPU-weeks; parallelize on 2–3 rented VMs ≈ $100–200 — the single biggest compute line).
3. Fit reliability curves `r_m(d)` (isotonic/binned per module per degradation axis).
4. Train fusion LightGBM on a *generator-stratified* split; tune leave-one-generator-out.
5. Fit isotonic calibration on held-out; fit conformal calibration stratified by degradation bucket.
6. Freeze and version everything: `judge-v1`, `lc-v1`, `launder-v1`, `index-v1`.

## 8.5 Evaluation Methodology (the contract with reality)

**Primary protocol — the LOGO × Ladder matrix:** for each held-out generator g (never in fusion training) × each laundering rung ℓ: report AUROC, precision@1%-base-rate, ECE, risk–coverage curve, abstention rate. *No single-number headline allowed; the matrix is the result.*

| Metric | Target (MVP exit) |
|---|---|
| AUROC, unseen generator, pristine | ≥ 0.90 |
| AUROC, unseen generator, heavy laundering (q50 + 0.5× resize) | ≥ 0.75 **with** abstention ≤ 35% |
| Conformal coverage validity (per stratum) | within ±2% of nominal 1−α |
| ECE (fused probabilities) | ≤ 0.05 |
| Confident-wrong rate (decisive verdict, wrong, trust>0.7) | ≤ 2% — **the metric that matters most** |
| Evidence-attribution audit (human check: do cited artifacts exist?) | ≥ 95% pass on 200-sample audit |
| Latency, full panel, CPU | ≤ 8 s p95 |

**Baselines to beat:** (1) reliability-weighted voting (own fallback); (2) a fine-tuned EfficientNet/CLIP-probe classifier (the "standard approach" strawman-that-isn't); (3) best available open training-free detector run standalone. ARGUS must beat (2) on unseen-generator and laundered strata specifically — that is the thesis being tested.

## 8.6 Twelve-Week Plan

| Weeks | Deliverable |
|---|---|
| 1 | Repo/CI/schema scaffolding; license audit (TruFor, FLUX outputs, datasets); dataset downloads start; **de-confounding pipeline written first** |
| 2–3 | Tier 0 + Tier 1 complete with unit tests against synthetic laundering cases; modules A, B |
| 4–5 | Modules C, E, F (one DINOv2 service, FAISS index build running in background); module D wrapped |
| 6 | Modules G (narrow), H (narrow); panel integration; module-contract conformance tests |
| 7 | Laundering simulator finalized + platform presets measured; calibration sweep launched (rented CPUs) |
| 8 | Reliability curves fit; fusion judge + isotonic + conformal trained; weighted-voting fallback |
| 9 | Verdict builder + explanation templates; analyst UI (canvas, overlays, evidence cards) |
| 10 | Full LOGO × Ladder evaluation run; baseline comparisons; confident-wrong autopsy |
| 11 | Fixes from autopsy (typically: gate floors, conflict features, template wording); re-run |
| 12 | Freeze v1; evaluation report; demo set; backlog grooming for post-MVP (retrieval module first) |

**Risk buffer:** modules G/H are pre-scoped narrow precisely because they are the likely overruns; if week 6 slips, G ships as "lighting-direction sanity check" only and H as OCR-gibberish flag only — both still contribute panel diversity, which is what the architecture needs from them at MVP.
