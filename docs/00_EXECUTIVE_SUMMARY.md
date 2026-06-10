# ARGUS — Authenticity Assessment Platform
## Executive Summary & Document Map

**Version:** 1.0 — June 2026
**Status:** Design specification, research-grounded (see `REFERENCES.md`)

---

## 1. The One-Sentence Thesis

> **ARGUS is not a deepfake detector. It is a degradation-aware evidence court: it first measures how much forensic evidence an image can still carry, then convenes a panel of heterogeneous, generator-agnostic evidence modules, and finally renders a structured verdict — with calibrated uncertainty and the formal right to say "I don't know."**

Every architectural decision in this document flows from three observations that the 2024–2026 research record now firmly supports:

1. **Classifiers trained on fakes are structurally doomed.** A classifier's decision boundary is defined by the fakes it has seen. The set of fakes is open and adversarially expanding; the set of real images is comparatively stable. Therefore ARGUS models *realness*, not *fakeness*, wherever possible (one-class / training-free methods: ZED, SpAN, RIGID-family — see Phase 1, 6).

2. **Social-media laundering is the main event, not an edge case.** A 2025 benchmark showed 15 state-of-the-art forgery-localization methods collapsing under platform compression/resizing. Most forensic evidence lives in high frequencies; laundering is a high-frequency eraser. ARGUS therefore estimates the image's **degradation state first**, and conditions the reliability of every downstream module on it. This "triage-before-judgment" loop is ARGUS's core novelty (Phase 4, 9).

3. **A bare score is a liability.** Production users (journalists, trust & safety, courts) need to know *what evidence supports the conclusion*, *what contradicts it*, and *when the system is guessing*. ARGUS outputs a structured verdict — evidence ranking, contradictions, human-readable reasoning — wrapped in **conformal prediction** for distribution-free abstention guarantees (Phase 5, 7).

---

## 2. Architecture at a Glance

```
                        ┌──────────────────────────────────────────────┐
 INPUT IMAGE ──────────►│  TIER 0 · PROVENANCE FAST-PATH               │
                        │  C2PA / Content Credentials verification     │──► signed & valid → verdict
                        └──────────────────┬───────────────────────────┘     (with caveats)
                                           │ absent / stripped / invalid
                                           ▼
                        ┌──────────────────────────────────────────────┐
                        │  TIER 1 · DEGRADATION TRIAGE                 │
                        │  JPEG quality & generations · resize factor  │
                        │  screenshot detection · denoise/filter est.  │
                        │  → DEGRADATION STATE VECTOR  d ∈ R^k         │
                        └──────────────────┬───────────────────────────┘
                                           ▼
                        ┌──────────────────────────────────────────────┐
                        │  TIER 2 · EVIDENCE PANEL  (parallel modules) │
                        │                                              │
                        │  A. Metadata & container forensics           │
                        │  B. Compression-history (ELA, JPEG ghosts)   │
                        │  C. Spectral probe (FFT upsampling traces)   │
                        │  D. Learned noise residual (TruFor-style)    │
                        │     + reliability map                        │
                        │  E. Real-distribution probe                  │
                        │     (frozen DINOv2 + one-class / kNN)        │
                        │  F. Perturbation-sensitivity probe           │
                        │     (training-free, RIGID-style)             │
                        │  G. Physics & geometry (shadows/reflections/ │
                        │     projective consistency)                  │
                        │  H. Semantic plausibility (hands, text,      │
                        │     typography, object logic)                │
                        │  I. Retrieval & context (reverse search,     │
                        │     near-duplicate provenance)   [optional]  │
                        │                                              │
                        │  Each module outputs:                        │
                        │  { evidence_score, reliability_score,        │
                        │    confidence_score, artifacts[] }           │
                        │  reliability = f(module, d)  ◄── conditioned │
                        │                on degradation state          │
                        └──────────────────┬───────────────────────────┘
                                           ▼
                        ┌──────────────────────────────────────────────┐
                        │  TIER 3 · THE COURT (fusion & verdict)       │
                        │  1. Reliability gate (drop/duck unreliable)  │
                        │  2. Calibrated stacking (LightGBM over       │
                        │     evidence × reliability features)         │
                        │  3. Isotonic calibration                     │
                        │  4. Conformal wrapper → {authentic, AI-gen,  │
                        │     manipulated, ABSTAIN} with guaranteed    │
                        │     error rate                               │
                        │  5. Verdict builder: scores + evidence       │
                        │     ranking + contradictions + explanation   │
                        └──────────────────────────────────────────────┘
```

---

## 3. Key Design Decisions (and where they are defended)

| # | Decision | Rationale | Phase |
|---|----------|-----------|-------|
| 1 | Model **realness**, not fakeness | Fake class is open-set; real class is (more) stationary | 1, 6 |
| 2 | **DINOv2 frozen** backbone, not CLIP, not fine-tuned CNN | Under transformations CLIP drops to ~42% acc; DINOv2 holds ~92% | 2, 8 |
| 3 | **Degradation triage before judgment**; reliability conditioned on degradation state | Evidence quality varies per image; modules must know when they are blind | 4 |
| 4 | Training-free probes (spectral, perturbation-sensitivity) as first-class citizens | Zero generator-specific training → cannot overfit to known generators | 3, 6 |
| 5 | PRNU **demoted**, latent inversion (DIRE-style) **demoted to optional plugin** | PRNU needs reference cameras & dies under compression; inversion is generator-family-specific and expensive | 3 |
| 6 | Fusion = reliability-gated stacking + **conformal abstention**, not GNN/pure Bayes | Best calibration-per-compute; formal coverage guarantees; per-image evidence weighting | 5 |
| 7 | C2PA = **fast-path tier**, never trust anchor | Strippable, revocation/expiry gaps (2026 analyses); absence of credentials is weak evidence | 3 |
| 8 | XAI = **template-grounded** explanations from evidence artifacts, LLM only as optional rephraser | No hallucination in a forensic product | 7 |
| 9 | Evaluation = **leave-one-generator-out × laundering ladder**, never i.i.d. test split | Measures the two failure modes that kill production detectors | 8 |

---

## 4. What Is Genuinely New Here

1. **Degradation-conditioned reliability calibration**: per-module reliability curves `r_m(d)` learned offline by passing labeled data through a simulated laundering ladder, then queried at inference using the estimated degradation state. No published system conditions *every* evidence stream on a measured laundering state (Phase 9).
2. **Conformal authenticity verdicts**: distribution-free abstention with user-settable risk level — converting "detector" into "assessor" (Phase 5).
3. **The evidence-court output contract**: a machine-readable verdict schema (scores, ranked evidence, contradictions, reliability disclosure) designed for downstream policy engines and human review (Phase 7).
4. **A laundering-aware open benchmark** falls out of the evaluation methodology almost for free and is itself publishable (Phase 9).

---

## 5. Constraint Compliance

| Constraint | How ARGUS satisfies it |
|---|---|
| Images only | All modules are image-native; no video/audio machinery |
| Open-source stack | PyTorch, timm (DINOv2), TruFor (research weights — license caveat in Phase 10), OpenCV, scikit-learn, LightGBM, MAPIE, exiftool, c2pa-python |
| Limited budget / commodity hardware | Frozen ViT-B/14 runs on CPU (~1–3 s/image); spectral/JPEG/metadata modules are milliseconds; no training of large models |
| No proprietary datasets | COCO/RAISE/Dresden/OpenImages (real), GenImage/Synthbuster/DF40 (fake), self-generated SDXL/Flux locally |
| No foundation-model fine-tuning | Backbones frozen; only lightweight heads (linear/kNN/GBM) are trained |
| Explainable | Every score traces to named artifacts; verdict schema in Phase 7 |
| Generalizes to unseen generators | Real-distribution modeling + training-free probes + physics evidence + abstention (Phase 6) |

---

## 6. Document Map

| File | Content |
|---|---|
| `01_PHASE1_PROBLEM_DECONSTRUCTION.md` | Why detectors fail; limits of classification; why accuracy lies; future-proof evidence |
| `02_PHASE2_FIRST_PRINCIPLES_DESIGN.md` | Axioms; the three-tier architecture; per-component justification |
| `03_PHASE3_EVIDENCE_TAXONOMY.md` | 22 evidence sources scored on robustness, laundering, explainability, cost |
| `04_PHASE4_RELIABILITY_ARCHITECTURE.md` | The module contract; degradation-conditioned reliability; contradiction handling |
| `05_PHASE5_EVIDENCE_FUSION.md` | Seven fusion strategies compared; the recommended hybrid |
| `06_PHASE6_UNKNOWN_GENERATOR_RESISTANCE.md` | What survives, what dies, how to know you don't know |
| `07_PHASE7_EXPLAINABILITY.md` | Verdict schema; trust/risk scores; grounded explanation generation |
| `08_PHASE8_MVP_3_MONTHS.md` | Exact modules, models, libraries, datasets, eval protocol, week-by-week plan |
| `09_PHASE9_RESEARCH_CONTRIBUTION.md` | Novelty, publishable units, competitor gaps, IP |
| `10_PHASE10_HOSTILE_REVIEW_AND_REDESIGN.md` | Adversarial self-review; redesign; final score /100 |
| `REFERENCES.md` | Sources underpinning every major claim |
