# ARGUS — FORMAL PROBLEM & LABEL SPECIFICATION
**Document #2 of the validation set · Status: REQUIRED BEFORE ANY CALIBRATOR IS FIT**
**Version:** 0.1

> Why this exists: every calibrator and every metric in ARGUS consumes a binary `authentic / synthetic` label. But authenticity lives on a **continuous edit spectrum** with no clean boundary (RAW → Lightroom → AI-denoise → 200 px generative fill → full synthesis). If the label is undefined, every ECE/Brier/AUROC number is undefined too. This document fixes the target, the decision boundary, the abstention semantics, and the unit of analysis — *before* fitting anything.

---

## 1. The quantity ARGUS estimates (definition)
ARGUS estimates, for a single still image `x`:

> **A(x) = P( x is a substantially-unmanipulated photographic capture of a real physical scene | evidence )**

This is deliberately **not** "was any neural network involved." Modern capture pipelines use learned denoise/HDR/super-res; those remain *authentic captures*. The target is **scene authenticity + capture authenticity**, not "pixel purity."

**Primary product output:** the **likelihood ratio** `LR(x) = P(e|authentic)/P(e|synthetic)`, which is **prior-independent**. The posterior `A(x)` is only computed when a deployer supplies an explicit `context_prior` (see §6). *Rationale: the base rate of fakes is set adversarially and is non-stationary; shipping a posterior calibrated to last month's prior is wrong on the highest-stakes day.*

## 2. The label taxonomy (the edit spectrum, made discrete)
We define **6 ordinal classes** and a **binary projection** used for training/metrics:

| Class | Description | Binary label | Notes |
|------:|-------------|:------------:|-------|
| **C0** | RAW / in-camera JPEG, no edits | **authentic** | gold authentic |
| **C1** | Global photographic edits (exposure, WB, crop, learned denoise/HDR) | **authentic** | *includes computational photography* — explicitly authentic |
| **C2** | Local non-generative retouch (clone/heal, dodge/burn) | **authentic** | bounded; see §3 threshold |
| **C3** | Generative *inpainting/fill* of a sub-region | **synthetic (tamper)** | the "partial synthesis" case |
| **C4** | Generative *outpainting* / major composite | **synthetic (tamper)** | |
| **C5** | Fully synthetic (text-to-image / full face-swap) | **synthetic** | gold synthetic |

**Binary projection rule:** `{C0,C1,C2} → authentic`, `{C3,C4,C5} → synthetic`. **C3 is the boundary class** and is treated specially in evaluation (§4).

## 3. The decision boundary (the hard part, made explicit)
The boundary between **C2 (authentic)** and **C3 (synthetic tamper)** is defined by an **area + semantic-change threshold**, fixed here so labeling is reproducible:

> An edit is **C3 (synthetic)** iff it (a) **introduces or removes semantic content** (an object, a person, text meaning, a body part) using a **generative model**, regardless of area; **OR** (b) generatively alters a contiguous region **> 1.0% of image area**. Otherwise it is **C2 (authentic)**.

- Pure *removal* via generative fill counts as C3 (content was fabricated to cover the hole).
- Photographic-only local edits (no generative model) are always C2, at any area.
- **Ambiguous cases** (e.g., an edit that is borderline on both criteria) are routed to **double annotation + adjudication** (§5) and, if still tied, are **excluded from the calibration set** and **retained only in a separate "boundary" eval bucket**.

## 4. Evaluation treatment of the boundary class C3
- C3 is **never** used to fit per-module isotonic calibrators (its label is the least stable).
- C3 is reported as its **own stratum** in all results, so a system that does well on C5/full-synthesis but fails on C3/partial-synthesis cannot hide behind an average.
- Localization metrics (IoU of predicted tamper heatmap vs annotated edit mask) are reported on C3/C4 only.

## 5. Annotation protocol (for C3/C4 boundary and suspicious-but-real)
- **Two independent annotators** per ambiguous image; **Cohen's κ** reported; κ < 0.6 on any batch triggers guideline revision.
- **Adjudication:** disagreements resolved by a third senior annotator; if unresolved → "boundary, excluded from calibration."
- **Provenance-anchored gold:** wherever possible, authentic labels come from **provenance** (RAW capture, C2PA, pre-2021 timestamp), *not* annotator judgment, to avoid the "real-looking ⇒ labeled real" circularity.
- **Synthetic gold:** comes from **known generation provenance** (we generated it, or it is from a labeled corpus), not from "looks fake."

## 6. The `context_prior` problem (explicit resolution)
The MVP has **no mechanism** to obtain a per-image prior. Resolution, fixed here:
- **Default product output is `LR`, not posterior `A(x)`.** No prior is invented.
- When a posterior is required for a metric (ECE on `A(x)`), use a **declared, fixed evaluation prior** (e.g., the empirical class balance of the *labeled* test set), recorded in the run config, and report **prior-sensitivity bands** (recompute at prior ∈ {0.1, 0.3, 0.5, 0.7, 0.9}).
- Calibration claims are made on the **declared-prior** basis only and labeled as such.

## 7. Abstention semantics (what "abstain" means and when it is allowed)
- **Abstain** = ARGUS declines to emit a confident Authenticity verdict; it still emits `LR`, Trust, Risk, evidence ranking, and a "what we could/could not check" report.
- **Trigger (frozen for the study):** `evidence_coverage < τ_cov` **OR** `mean_reliability < τ_rel`. Thresholds τ are tuned on a **calibration split only**, recorded, and frozen before test.
- **Abstention is scored honestly:** on the hard subset, abstention rate is a **reported failure metric**, and a mandatory **forced-decision (100% coverage) accuracy** is also reported, so the system cannot "win" selective-risk by abstaining on everything hard.

## 8. Unit of analysis & scope boundaries
- **Unit:** a single still image. **Out of scope (declared):** video, audio, multi-frame, and "intent" (a real photo used in a misleading context is *authentic* under A(x); context/harm is the Risk axis's job, not Authenticity's).
- **One image = one row** in all statistics. Laundered variants of the same source are **grouped** for bootstrap resampling (cluster by source image to avoid leakage; see Stat Plan §4).

## 9. Known label limitations (stated, not hidden)
- C3 boundary is a human convention, not a law of nature; κ quantifies its softness.
- Provenance-anchored authentic labels skew toward devices/regions that emit provenance (bias; see Datasheet Doc #4).
- "Substantially unmanipulated" inherits the 1.0% area threshold's arbitrariness; sensitivity to this threshold (0.5% / 2%) is reported as a robustness check.

---
*Sign-off gate: no isotonic calibrator may be fit until §2–§3 are frozen and the calibration set is filtered to exclude C3-boundary and adjudication-tied images.*
