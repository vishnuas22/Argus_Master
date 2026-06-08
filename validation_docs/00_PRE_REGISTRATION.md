# ARGUS — PRE-REGISTRATION OF THE VALIDATION STUDY
**Document #1 of the validation set · Status: TEMPLATE TO FREEZE BEFORE DATA IS TOUCHED**
**Version:** 0.1 (draft) → freeze as 1.0 with a content hash committed to git before Day 4.
**Binding rule:** Everything in §3–§7 is fixed *before* any test-set result is observed. Any post-hoc deviation must be logged in §10 (Amendments) with timestamp + justification, or the corresponding result is reported as *exploratory*, not *confirmatory*.

> Purpose: convert ARGUS's prose claims into a small set of confirmatory tests with **decision rules written in advance**, so that a positive result is publishable and a negative result is a legitimate, fundable outcome (success-as-falsification). This is the single cheapest artifact that separates a design document from science.

---

## 1. Background and motivation (one paragraph)
ARGUS asserts that decoupling per-module **reliability** from **confidence**, and tempering forensic evidence by an empirically-calibrated reliability term, yields better calibration and selective risk under social-media laundering than (a) the same fusion without reliability and (b) a temperature-scaled single classifier. It further asserts that generator-agnostic "Tier-A" evidence (physics/provenance/retrieval) survives unseen generators where fingerprint evidence collapses. None of these claims has been measured. This study tests the claims that are simultaneously **central** and **unvalidated**, on consumer hardware, with open data, in 30 days.

## 2. Confirmatory hypotheses (and the prose claim each operationalizes)

| ID | Confirmatory hypothesis (one-sided unless stated) | Prose claim it tests |
|----|---------------------------------------------------|----------------------|
| **H1a** | `A-full` has **lower** laundered-ECE than `A-null` | Reliability layer improves calibration |
| **H1b** | `A-full` has **lower** hard-subset selective risk than `A-null` | Reliability layer improves selective risk |
| **H1c** | `A-full` has **lower** laundered-ECE than `CLIP-TS` | ARGUS beats a calibrated single classifier |
| **H1d** | `A-full` has **lower** hard-subset selective risk than `CLIP-TS` | (same) |
| **H2** | Physics/shadow/lighting module AUROC on a held-out **physics-aware** generator is **> 0.60** | "Durable Tier-A backbone" |
| **H3** | On answered cases, error rate is **non-increasing** as laundering severity rises while abstention rises | "Graceful, honest degradation" |
| **H4** | Effective number of independent evidence channels (participation ratio) is **> 4** | "20+ orthogonal modules" |
| **H5 (falsification)** | Pre-isotonic ECE of `r·LLR` tempering is **not** statistically better than proper reliability marginalization | "Calibrated by construction" (we expect to *reject*) |
| **H6** | Removing each module's top-ranked cue moves the verdict in the predicted direction by **> 0** (correlation-controlled) | "Explanation faithfulness" |

**Primary endpoints:** H1a–H1d. **Secondary:** H2, H3, H4. **Falsification/exploratory:** H5, H6.

## 3. Experimental arms (frozen)
- **A-full** — 7 classical modules → per-module isotonic calibration → reliability-tempered log-odds pool → threshold abstention.
- **A-null** — *identical* modules, calibrators, pool, and abstention, but **all reliability_score forced to 1.0**. The *only* difference from A-full is the reliability layer.
- **CLIP-TS** — frozen `open_clip` ViT-B/32 (`laion2b_s34b_b79k`) → logistic-regression probe → temperature scaling. Single-classifier baseline.
- **(Secondary baselines, reported but not gating):** deep-ensemble of 3 probes; one recent open-set/CLIP-based detector with post-hoc calibration.

Arms receive **byte-identical inputs**. No arm sees the test generators or the suspicious-but-real set during any fitting step.

## 4. Datasets and splits (frozen; full detail in Datasheet, Doc #4)
- **Authentic-pristine:** RAISE + Dresden (device-diverse subset enforced).
- **Authentic-wild:** Flickr-CC **pre-2021** (hard date cutoff to avoid diffusion-era contamination).
- **Synthetic (contrast & LOGO only, never in the analytic pool):** GenImage + DiffusionForensics subsets, partitioned by **generator family**.
- **Suspicious-but-real (~500):** low-end phones, non-Western faces/dress, unusual optics, non-Latin script, flagship-phone computational-photography images. *False-positive measurement only.*
- **LOGO split:** hold out **≥2 entire generator families**; all calibration/probe fitting uses only the *seen* families.
- **Laundering grid (applied identically to all arms):** JPEG-Q ∈ {95,85,75,60,40}; downscale ∈ {1.5,2,3}×; screenshot-sim; double-compress.
- **Hard subset (pre-defined predicate):** {no valid C2PA} ∧ {no retrieval hit} ∧ {JPEG-Q ≤ 60 OR screenshot flag}.

**Test-set lockbox:** the LOGO held-out families, the laundered test images, and the suspicious-but-real set are placed in a write-once directory with a recorded SHA-256 manifest and are **not opened** until §3 and §6 are frozen.

## 5. Metrics (exact definitions; formulas in Stat Plan, Doc #3)
- **ECE** — 15-bin *adaptive* (equal-mass) binning, reported **separately on pristine vs laundered**.
- **Brier score** — primary proper scoring rule (ECE is binning-biased at small N; Brier is the tie-breaker).
- **Selective risk / risk–coverage** — on the **hard subset**, PLUS a mandatory **forced-decision accuracy at 100% coverage** (abstention rate on the hard subset is itself reported as a *failure* metric, not a feature).
- **LOGO AUROC** — with 1000× bootstrap CIs.
- **Robustness curves** — AUROC vs JPEG-Q and vs downscale.
- **Effective channel count** — participation ratio of the module-correlation eigenspectrum.

## 6. Decision rules (FROZEN — written before any test result)

### Primary go/no-go (Day 17 checkpoint)
> **The reliability thesis (H1) SURVIVES iff `A-full` beats BOTH `A-null` AND `CLIP-TS` on BOTH laundered-ECE AND hard-subset selective risk, each with non-overlapping 95% bootstrap CIs (1000 resamples, BCa).**

- If `A-full` and `A-null` CIs **overlap** on either primary metric → the reliability layer (the sole novelty) is **empirically null** → **STOP. Publish the negative result.** ARGUS reduces to a forensic ensemble.
- If `CLIP-TS` matches `A-full` on calibration within CIs → the differentiator is gone → STOP/relabel.
- **Effect-size floor:** a "win" also requires the point estimate to clear a pre-registered **minimum meaningful difference** of **ΔECE ≥ 0.02** and **Δselective-risk ≥ 0.03** (a statistically significant but trivially small gain does not count as survival).

### Secondary rules
- **H2:** physics-module AUROC ≤ 0.60 (upper CI bound < 0.65) on the physics-aware generator → the durable-backbone claim is **falsified**; re-score the architecture and demote physics to decaying Tier-B in all downstream claims.
- **H4:** effective channel count ≤ 4 → report "orthogonality" claim as **not supported**; the Naïve-Bayes independence assumption is flagged as violated in all writeups.
- **H3:** if error-on-answered *increases* with laundering severity (positive slope, CI excludes 0) → "graceful degradation" **not supported**.
- **Liability gate (KT-7):** if suspicious-but-real false-"manipulated" rate ≥ 10% overall, OR ≥ 2× on any pre-defined subgroup → flag as **deployment-blocking** regardless of all other results.

## 7. Statistical analysis summary (full plan in Doc #3)
- **CIs:** BCa bootstrap, 1000 resamples, paired across arms (same images).
- **Significance:** paired bootstrap difference test; report the full CI, not just p.
- **Multiple comparisons:** primary family = {H1a,H1b,H1c,H1d}; control family-wise error with **Holm–Bonferroni** at α=0.05.
- **Power:** target ability to detect ΔECE=0.02 at 80% power (sample-size justification in Doc #3 → drives the ≥1000-per-class requirement).

## 8. What would make us *abandon* vs *amend*
- **Abandon (publish negative):** H1 primary rule fails. This is an acceptable, pre-committed outcome.
- **Amend (log + continue):** a tooling failure (e.g., FAISS/CLIP version mismatch) corrupts an arm → fix, re-freeze, re-run, log in §10.
- **Never:** silently swap metrics, datasets, or thresholds after seeing results. That converts the study from confirmatory to exploratory and must be labeled as such.

## 9. Roles, timeline, and artifacts
- **Timeline:** the 30-day roadmap (Doc set intro). Day-17 = primary checkpoint.
- **Artifacts on completion:** (1) the frozen pre-registration hash; (2) `results/` with per-arm metric tables + CIs; (3) reliability diagrams; (4) the negative-or-positive report; (5) the released open-data split manifest.
- **Reproducibility:** every number traceable to a seed + commit (see Doc #5).

## 10. Amendments log (append-only)
| Date | Section | Change | Justification | Confirmatory→Exploratory? |
|------|---------|--------|---------------|---------------------------|
| _(empty at freeze)_ | | | | |

---
*Freeze checklist before Day 4: §2 hypotheses fixed ☐ · §3 arms implemented to spec ☐ · §4 lockbox sealed + SHA-256 recorded ☐ · §6 decision rules committed to git ☐ · document hash recorded ☐.*
