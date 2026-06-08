# ARGUS — STATISTICAL ANALYSIS PLAN (SAP)
**Document #3 of the validation set · Status: FREEZE WITH THE PRE-REGISTRATION**
**Version:** 0.1

> Why this exists: the document's named evaluation weakness is "no significance testing / CIs; ECE binning-biased at small N." With only a few held-out generators, every AUROC/ECE is high-variance; single point numbers are not publishable. This SAP fixes the estimators, the resampling scheme, the power/sample-size justification, and the multiple-comparison control **before** data is seen.

---

## 1. Estimands (what each number means)
| Symbol | Estimand | Estimator |
|--------|----------|-----------|
| `ECE_L` | expected calibration error on laundered images | 15-bin **adaptive (equal-mass)** ECE |
| `Brier` | mean squared error of probabilistic prediction | `mean((p − y)²)` |
| `SR_hard(c)` | selective risk on hard subset at coverage `c` | error among the `c`-fraction most-confident answered cases |
| `FD_acc` | forced-decision accuracy at 100% coverage (hard subset) | accuracy with abstention disabled |
| `AUROC_LOGO` | discrimination on held-out generator families | rank-based AUROC |
| `D_eff` | effective number of independent channels | participation ratio `(Σλ)²/Σλ²` of the module-correlation eigenspectrum |

## 2. Why ECE alone is insufficient → the metric stack
- **ECE is binning-biased and low-power at small N.** It is reported, but the **confirmatory tie-breaker is Brier** (a proper scoring rule, unbiased).
- Reliability **diagrams** (with per-bin Wilson CIs) accompany every ECE number; a single scalar is never reported alone.
- For discrimination, **AUROC** is reported with bootstrap CIs; for the hard subset, **risk–coverage curves** + the single mandatory **FD_acc** point.

## 3. Resampling & uncertainty: BCa bootstrap, paired, source-clustered
- **Method:** bias-corrected-and-accelerated (BCa) bootstrap, **B = 1000** resamples.
- **Pairing:** arms are compared on the **same images** → resample the *difference* `metric(A-full) − metric(A-null)` per bootstrap replicate (paired).
- **Clustering (critical):** the unit of resampling is the **source image**, not the laundered variant. All laundered children of a source are resampled together to prevent leakage-inflated significance. (Tied to Label Spec §8.)
- **Reported quantity:** the full 95% BCa CI of every difference, not just a p-value.

## 4. Sample-size / power justification (drives the ≥1000-per-class rule)
- **Target effect:** minimum meaningful difference **ΔECE = 0.02**, **Δselective-risk = 0.03** (from Pre-reg §6 effect-size floor).
- **Power target:** 80% to detect ΔECE = 0.02 at α = 0.05 (two-sided, paired bootstrap).
- **Variance assumption:** per-image squared-error SD ≈ 0.25 (conservative for `p∈[0,1]`), correlation across arms ρ ≈ 0.7 (paired design reduces variance by `1−ρ`).
- **Result:** required **n ≈ 900–1100 per class per condition** → the pre-registered **≥1000 authentic + ≥1000 synthetic per held-out family** satisfies 80% power with margin. A formal recompute is run on the *calibration* split before the lockbox opens; if power < 80%, sample sizes are increased *before* test (logged as a pre-data amendment, not a post-hoc one).

## 5. Multiple-comparison control
- **Primary family** = {H1a, H1b, H1c, H1d} (4 tests). Control family-wise error with **Holm–Bonferroni** at α = 0.05.
- **Secondary family** = {H2, H3, H4} reported with CIs; Holm within the secondary family.
- **Exploratory** (H5, H6, per-module ablations, robustness sweeps): reported with CIs and explicitly labeled **exploratory — not error-controlled**. No exploratory result is described as "significant."

## 6. The ablation grid analysis (correlation-aware)
- Each ablation arm's effect = paired difference vs Full, with BCa CI.
- **Backup-masking control:** because correlated modules mask each other under leave-one-out, the **primary per-group attribution is grouped knockout** (remove a correlated cluster identified by §1 `D_eff` clustering) and **Shapley values** over modules (Monte-Carlo, 200 permutations) — Shapley is robust to the correlation confound that breaks naïve leave-one-out.
- The single most important ablation row is **−reliability (all r=1)**; it must show the largest laundered-metric degradation, else H1 is dead (this is KT-1 embedded in the grid).

## 7. Calibration-specific analyses
- ECE computed **separately** on pristine vs laundered (never pooled — pooling hides the entire claim).
- **Prior-sensitivity:** every posterior-based metric recomputed at prior ∈ {0.1,0.3,0.5,0.7,0.9} (Label Spec §6); report the band.
- **Pre-vs-post isotonic:** report ECE before and after per-module isotonic, so the H5 ("calibrated by construction") falsification is visible: if pre-isotonic ECE is poor, "by construction" is false.

## 8. Robustness & stratified reporting (anti-gaming)
Mandatory stratified tables — **no headline number without these strata**:
- by **laundering severity** (each JPEG-Q, each downscale, screenshot);
- by **hard vs easy subset**;
- by **edit class** C0–C5 (boundary class C3 isolated);
- by **subgroup** (device tier, region, script) for false-positive rates.
- **FD_acc at 100% coverage on the hard subset is mandatory** alongside any risk–coverage curve (prevents the "abstain on everything hard" trick).

## 9. Decision arithmetic (mirrors Pre-reg §6, made computational)
```
SURVIVE_H1 = (CI95(ECE_L[A-null] − ECE_L[A-full]).low  > 0) AND
             (CI95(ECE_L[CLIP-TS] − ECE_L[A-full]).low > 0) AND
             (CI95(SR_hard[A-null] − SR_hard[A-full]).low  > 0) AND
             (CI95(SR_hard[CLIP-TS] − SR_hard[A-full]).low > 0) AND
             (point(ECE_L[A-null] − ECE_L[A-full]) >= 0.02) AND
             (point(SR_hard[A-null] − SR_hard[A-full]) >= 0.03)
             # after Holm–Bonferroni across the 4 CI tests
FALSIFY_H2 = CI95(AUROC_phys).high < 0.65
SUPPORT_H4 = D_eff > 4
```

## 10. Reproducibility of the statistics
- Single seed for the bootstrap RNG, recorded in the run config (Doc #5).
- The analysis script consumes only the frozen `predictions.parquet` (one row per image per arm) and emits all tables/figures deterministically.
- Raw per-image predictions are released with the paper so any reviewer can recompute every CI.

---
*Anti-pattern checklist (all must be FALSE at submission): pooled pristine+laundered ECE ☐ · point estimate without CI ☐ · leave-one-out used as the sole attribution ☐ · risk–coverage shown without FD_acc ☐ · exploratory result called "significant" ☐.*
