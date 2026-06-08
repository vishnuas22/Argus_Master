# PHASE 2 — First-Principles Design

> *We discard the assumption that ARGUS is a detector. We rebuild from the question: "What system would a forensic scientist design to reason about authenticity under uncertainty?"*

## 2.1 First principles

We start from axioms, not architectures:

- **A1 — Authenticity is a hypothesis, not a label.** The correct output is a *posterior belief* over a hypothesis, with a stated uncertainty, not a class.
- **A2 — No single trace is dispositive.** Forensic conclusions come from *convergence of independent evidence*. Therefore the system is a fusion engine, not a classifier.
- **A3 — Evidence has variable quality.** Every cue depends on physical preconditions in the image; when those are destroyed, the cue must be *silenced*, not guessed. Therefore reliability is a first-class, per-module quantity.
- **A4 — The generator is unknown.** Design for the open set. Prefer generator-agnostic evidence; treat generator-specific evidence as fast-decaying.
- **A5 — A conclusion that cannot be explained cannot be trusted.** Explanation is part of the output contract, not a post-hoc add-on.
- **A6 — The adversary includes the platform.** Robustness to laundering is a design constraint equal in weight to accuracy.
- **A7 — Absence of evidence ≠ evidence of absence.** "No camera fingerprint" is weak on a laundered image and strong on a pristine one — the *same* observation means different things depending on reliability.

From these, the architecture is forced: **parallel orthogonal evidence extractors → per-module reliability estimation → calibrated uncertainty-aware fusion → open-set gate → tri-axial scoring → explanation generator.**

## 2.2 The three orthogonal outputs (and why one number is wrong)

A single "authenticity score" conflates three independent questions. ARGUS separates them:

1. **Authenticity score** *(P(authentic capture) )*: the calibrated posterior that the image is a real photographic capture of a real scene, given the evidence.
2. **Trust score** *(epistemic quality)*: how much evidence the image actually afforded, and how reliable/non-contradictory it was. A pristine RAW yields high trust; a 5th-gen screenshot yields low trust **regardless of the authenticity reading.** This is the "should you act on this?" axis.
3. **Risk score** *(decision-facing)*: the probability-weighted *harm-relevant* likelihood of manipulation, combining authenticity, trust, contextual prior, and the *severity* of contradictions. Risk is what a moderator or analyst actually triages on.

> Why this matters: an image can be **probably authentic (0.8) but low trust (0.3)** — "looks real but we can barely tell." Incumbent systems cannot express this. It is the single most useful thing ARGUS can say to a human.

## 2.3 System architecture (component by component)

```
                          ┌──────────────────────────────────────────┐
   image ──▶ INGEST &     │           EVIDENCE EXTRACTION LAYER        │
            QUALITY  ──▶  │  (parallel, orthogonal, each emits a       │
            PROFILER      │   {evidence, reliability, confidence})     │
                │         │                                            │
                │         │  P  Provenance/C2PA   ┐                    │
                │         │  M  Metadata/EXIF      │ Tier A            │
                │         │  C  Camera pipeline    │ (generator-       │
                │         │  N  Sensor/PRNU        │  agnostic)        │
                │         │  J  JPEG/compression   ┘                   │
                │         │  F  Frequency/wavelet  ┐                   │
                │         │  R  Residual/CNN       │ Tier C            │
                │         │  D  Diffusion/inversion┘ (fingerprints)    │
                │         │  H  Physics/lighting   ┐                   │
                │         │  S  Semantic/anatomy   │ Tier A/B          │
                │         │  T  Typography/texture │                   │
                │         │  X  Retrieval/embedding┘                   │
                └────────▶│                                            │
                          └─────────────────┬──────────────────────────┘
                                            │ N×{e,r,c} tuples + global quality profile
                                            ▼
                          ┌──────────────────────────────────────────┐
                          │   RELIABILITY & CONTRADICTION LAYER       │
                          │  • per-module reliability calibration     │
                          │  • cross-module contradiction graph       │
                          │  • OOD / open-set gate (is evidence       │
                          │    itself out-of-distribution?)           │
                          └─────────────────┬──────────────────────────┘
                                            ▼
                          ┌──────────────────────────────────────────┐
                          │   CALIBRATED FUSION CORE                  │
                          │  Bayesian belief combination  +          │
                          │  learned correction head (LightGBM) +    │
                          │  conformal/abstention layer              │
                          └─────────────────┬──────────────────────────┘
                                            ▼
                  ┌─────────────┬────────────┬─────────────┐
                  ▼             ▼            ▼             ▼
            Authenticity     Trust        Risk      Explanation (XAI):
              (posterior)  (epistemic)  (decision)  ranked evidence,
                                                     contradictions, NL verdict
```

### Component-by-component justification

For each: **why it exists · problem solved · failure modes · compute cost · long-term robustness.**

**(0) Ingest & Quality Profiler.**
- *Why:* every downstream reliability estimate needs a quantitative description of how degraded the image is.
- *Solves:* the static-weighting failure (§1.1.7). Produces a *quality vector*: resolution, estimated JPEG quality & generations, blockiness, blur/Laplacian energy, noise floor, screenshot likelihood, double-compression indicators, EXIF presence.
- *Failure modes:* mis-estimating compression history on exotic codecs; adversarial quality spoofing.
- *Cost:* negligible (classical DSP, <50 ms CPU).
- *Robustness:* very high — quality estimation is generator-independent and only gets more useful over time.

**(P/M) Provenance & Metadata.**
- *Why:* the only cryptographically grounded, generator-independent positive evidence.
- *Solves:* the "durable evidence" mandate (§1.4 Tier A). Validates C2PA manifests, signature chains, soft-binding watermarks; parses EXIF/XMP for internal consistency (camera model ↔ lens ↔ resolution ↔ timestamps ↔ thumbnail).
- *Failure modes:* metadata is trivially strippable (laundering → low reliability, not false negative) and forgeable (so *presence* is weighted by signature validity). Absence is near-uninformative.
- *Cost:* negligible.
- *Robustness:* C2PA *presence* is rising over time and is forgery-resistant; this module's value grows.

**(C/N/J) Camera-pipeline / Sensor-PRNU / JPEG.**
- *Why:* a real capture has a coherent physical imaging chain; synthesis must fabricate one.
- *Solves:* generator-agnostic positive authenticity + tamper localization. CFA/demosaicing correlation maps, PRNU consistency across the frame, JPEG ghost / double-quantization / grid-discontinuity maps (also localize splices).
- *Failure modes:* destroyed by heavy recompression/downscale (→ reliability collapses, correctly); PRNU needs reference or self-consistency only.
- *Cost:* low–moderate (classical, 0.1–1 s CPU).
- *Robustness:* high on pristine images, gracefully self-silencing on laundered ones — exactly the desired behavior.

**(F/R/D) Frequency-Wavelet / Residual-CNN / Diffusion-inversion.**
- *Why:* fast, cheap, strong on *known* generators.
- *Solves:* current-generation synthetic detection (the Tier-C bucket).
- *Failure modes:* the core arms-race trap — fingerprints move with each new generator; the OOD gate must heavily discount these on unknown inputs.
- *Cost:* low (FFT) to moderate (a small CNN/CLIP probe, ~20–100 ms GPU).
- *Robustness:* *low and declining by design* — used as corroboration, never as backbone.

**(H/S/T) Physics-lighting / Semantic-anatomy / Typography-texture.**
- *Why:* generators improve at texture far faster than at globally consistent optics, geometry, and world knowledge.
- *Solves:* durable, generator-agnostic anomaly detection (lighting-direction inconsistency, shadow/perspective geometry, anatomy/count errors, garbled text, texture-stationarity anomalies).
- *Failure modes:* hard, sometimes ambiguous; needs careful calibration to avoid false alarms on legitimately unusual scenes.
- *Cost:* moderate–high (a VLM or geometric estimators, 0.2–2 s).
- *Robustness:* medium–high and slowly decaying — a long-term moat.

**(X) Retrieval / Embedding.**
- *Why:* authenticity can be established *externally* without trusting a single pixel.
- *Solves:* the strongest open-set signal — provenance-by-corroboration. Reverse-image / near-duplicate search over an indexed corpus + perceptual-hash + CLIP-embedding neighborhood; surfaces earlier originals, known-fake matches, and stock/AI-gallery matches.
- *Failure modes:* corpus coverage gaps; novel images with no neighbors yield abstention, not error.
- *Cost:* moderate (embedding + ANN search).
- *Robustness:* very high and *increasing* with corpus growth — generators cannot rewrite history.

**Reliability & Contradiction Layer / Fusion Core / Open-set Gate / XAI** — designed in Phases 4–7.

## 2.4 Challenged assumptions (what we explicitly reject)

- **Rejected:** "A bigger/better classifier solves it." — No; convergence (§1.2) caps pure-pixel detection.
- **Rejected:** "Frequency artifacts are a stable tell." — They are architecture-specific and laundering-fragile.
- **Rejected:** "More training data on more generators generalizes." — It interpolates among *seen* generators; the failure is on *unseen* ones.
- **Rejected:** "One authenticity number is the deliverable." — Conflates three orthogonal quantities and discards uncertainty.
- **Adopted instead:** orthogonal evidence + reliability-awareness + calibrated open-set fusion + tri-axial explained output.

\newpage
