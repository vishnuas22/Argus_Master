# ARGUS — DATASET DATASHEETS & DATA-INTEGRITY CONTROLS
**Document #4 of the validation set · Status: REQUIRED BEFORE DATA IS USED**
**Version:** 0.1 · Format follows Gebru et al., "Datasheets for Datasets," adapted.

> Why this exists: three of ARGUS's deployment-blocking risks are data risks — (1) **real/fake confounding** (FFHQ *is* the GAN source → the classifier learns alignment, not fakeness), (2) **web contamination** (post-2022 synthetic images mislabeled as "real"), (3) **demographic/device skew** (Western, high-end, English) → biased reliability and biased false positives. This document forces each to be measured, not assumed.

---

## PART I — POOL-BY-POOL DATASHEETS

### Pool 1 — Authentic-pristine (RAISE + Dresden)
- **Purpose:** transfer-function fitting; pristine-image ECE; camera-pipeline/noise calibration.
- **Composition:** RAW + in-camera JPEG; multiple camera bodies/sensors.
- **Label basis:** **provenance-anchored** (native RAW capture) — not annotator judgment.
- **Known skew:** high-end cameras, controlled scenes, Western-photographer origin.
- **Mandatory guard:** enforce a **device-diverse subset** (≥ N distinct sensors); record per-device counts.
- **Excluded:** any file lacking RAW provenance; any post-2021 ingestion of uncertain origin.

### Pool 2 — Authentic-wild (Flickr-CC, pre-2021)
- **Purpose:** laundered-real ECE; the "real but degraded" distribution.
- **Label basis:** **temporal anchoring** — hard cutoff **< 2021** (pre-diffusion-saturation), so a "real" label is credible.
- **Contamination control:** see Part II §C1. Spot-audit a random 200 for synthetic contamination.
- **Known skew:** Flickr-user demographics (Western/hobbyist/English skew) — flagged, quantified in Part III.

### Pool 3 — Synthetic (GenImage + DiffusionForensics subsets) — **CONTRAST & LOGO ONLY**
- **Purpose:** the demoted CLIP probe; the LOGO arms. **Never enters the analytic fusion pool.**
- **Label basis:** known generation provenance.
- **Partition:** by **generator family**; ≥ 2 families held out entirely for LOGO.
- **Critical guard (confound):** authentic counterpart images for any seen-generator comparison must be **source-matched** (Part II §C2), or results are reported as confounded.

### Pool 4 — Suspicious-but-real (~500, **to be built**)
- **Purpose:** the **false-positive measurement set the original docs omit** — the liability-critical metric.
- **Required composition (quota-sampled, recorded):**
  - low-end / budget phones (≥ 100);
  - non-Western faces & dress (≥ 100);
  - unusual optics (macro, fisheye, astrophotography, medical/endoscopic) (≥ 100);
  - non-Latin script in-scene (≥ 80);
  - **flagship-phone computational-photography** images (deep-fusion/HDR/night-mode) (≥ 120) — the specific case that breaks the "too-clean" module.
- **Label basis:** provenance-anchored authentic (capture by known person/device).
- **Use:** FP rate overall + **per-subgroup**; never used for fitting.

### Pool 5 — Physics-aware generator holdout (for KT-2 / H2) — **to be sourced**
- **Purpose:** falsify the "durable physics backbone."
- **Composition:** outputs of **one** recent (2025–26) physics/geometry-aware generator (e.g., a 3D-/lighting-conditioned diffusion model).
- **Use:** physics/shadow/lighting module AUROC only. Held out from everything else.

### Transform set — Laundering grid
- JPEG-Q ∈ {95,85,75,60,40}; downscale ∈ {1.5,2,3}×; screenshot-sim; double-compress.
- **V2 (not this study):** real platform round-trips (upload→download). A 200-image **real-vs-sim spot check** (KT/A2) is run this study to bound the sim-to-real gap.

---

## PART II — INTEGRITY CONTROLS (each produces a number that gates the study)

### C1 — Web-contamination audit (authentic pools)
- **Procedure:** run the demoted synthetic-probe + a published AI-image detector over Pools 1–2; manually review the top-scoring 200.
- **Gate:** estimated synthetic-contamination rate of the "authentic" pool **must be < 2%**; if higher, tighten temporal cutoff / provenance requirement before use.
- **Reported:** contamination rate ± CI in the final report.

### C2 — Real/fake confound control (the FFHQ trap)
- **Procedure:** for every seen-generator comparison, construct **source-matched** authentic negatives (same alignment, resolution, and capture pipeline as the generator's training source). Train the probe; evaluate on a held-out **confound-controlled** pair set.
- **Gate (KT-6):** if probe accuracy **collapses toward chance** on the controlled set, **all seen-generator accuracy numbers are reported as confounded** and de-emphasized.

### C3 — Leakage control (LOGO + laundering)
- No generator family appears in both train and test.
- All laundered children of a source image stay in the **same split** (no source spans train/test).
- Calibrators and the CLIP probe are fit **only** on seen families + the calibration split.

### C4 — Lockbox & manifest
- Pools 3-holdout, 4, 5 and the laundered test images go in a **write-once** directory.
- A **SHA-256 manifest** of every test file is recorded at freeze; the analysis verifies hashes at runtime (detects accidental mutation).

---

## PART III — BIAS & REPRESENTATION ACCOUNTING (release gate)

### Subgroup axes (defined now, measured later)
- **Device tier:** flagship / mid / low-end / unknown.
- **Region/appearance proxy:** Western / non-Western (annotated on faces/dress where applicable).
- **Script:** Latin / non-Latin in-scene text.
- **Capture condition:** well-lit / low-light / computational-photography.

### Required reporting
- Per-subgroup **counts** in every pool (exposes skew quantitatively).
- Per-subgroup **false-positive ("authentic flagged manipulated") rate** with CIs.
- **Release gate (mirrors Pre-reg liability gate):** per-subgroup FP disparity **< 2×**; overall suspicious-but-real FP **< 10%**. Failing either = **deployment-blocking**, reported as such.

### Honesty clause
- Where a subgroup is under-represented, the correct behavior is **lower Trust / abstain**, not a confident guess; this is verified by checking that abstention (not error) rises on under-represented inputs.

---

## PART IV — LICENSING & PROVENANCE OF THE DATASETS THEMSELVES
- Record license + redistribution terms for each pool (RAISE, Dresden, Flickr-CC, GenImage, DiffusionForensics) before use.
- Pool 4 (suspicious-but-real) and Pool 5 sourcing: record consent/licensing for any human-subject imagery; prefer Creative-Commons / self-captured.
- Released artifact = **split manifests + SHA-256 + per-image labels**, not necessarily the pixels (respect upstream licenses); enough for exact reproduction by a third party with dataset access.

---
*Data-readiness checklist before Day 4: C1 contamination < 2% ☐ · C2 confound-controlled pairs built ☐ · Pool 4 quotas met + subgroup-labeled ☐ · Pool 5 sourced ☐ · lockbox SHA-256 recorded ☐ · per-pool licenses logged ☐.*
