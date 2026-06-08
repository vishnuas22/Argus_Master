# PHASE 5 — Evidence Fusion

> *We need to combine N reliability-weighted evidence tuples into calibrated posteriors. We evaluate seven candidate strategies and recommend a hybrid.*

## 5.1 Requirements the fusion layer must satisfy

1. **Reliability-aware:** the weight of each cue must scale with its reliability *at inference time*, not a fixed learned weight.
2. **Calibrated:** output a probability whose value means what it says (low ECE), *including under distribution shift*.
3. **Open-set safe:** must not become overconfident when all evidence is OOD; must support abstention.
4. **Explainable:** must expose each cue's contribution (for Phase 7).
5. **Data-efficient:** trainable without proprietary or massive datasets.
6. **Contradiction-aware:** able to *raise uncertainty* on conflict rather than average it away.
7. **Cheap & deployable** on commodity hardware.

## 5.2 Candidate comparison

| Strategy | Calibration | Reliability-weighting | Explainability | Data need | OOD safety | Verdict |
|---|---|---|---|---|---|---|
| **Weighted voting** | poor | manual/static | high | none | poor (fixed weights) | Too rigid; can't adapt weights to image quality |
| **Bayesian fusion (log-likelihood-ratio combination)** | **excellent** | **native** (LLRs scaled by reliability) | **high** (per-cue LLR contribution) | low (per-module calibration only) | **good** (uninformative cue → LLR 0) | **Core — chosen** |
| **LightGBM** | good (w/ calibration) | learned (reliability as features) | medium-high (SHAP) | medium | medium (extrapolation risk) | **Correction head — chosen** |
| **XGBoost** | good | learned | medium-high (SHAP) | medium | medium | Equivalent to LightGBM; LightGBM lighter |
| **Graph neural network** | medium | learned | low | **high** | low (overfits) | Overkill; data-hungry; opaque |
| **Probabilistic graphical model (Bayesian network)** | excellent | native | **very high** | low–medium (structure design) | good | Strong; subsumed into the Bayesian core |
| **Mixture of experts** | medium | gating ≈ reliability | medium | high | medium | Conceptually right; the reliability layer *is* a hand-designed, calibrated MoE gate |

### Why each is or isn't chosen

- **Weighted voting:** the right *intuition* (combine opinions) but static weights violate Requirement 1. It is the degenerate special case of Bayesian fusion with fixed LLRs; we keep the intuition, drop the rigidity.
- **Bayesian fusion (log-likelihood-ratio / Naïve-Bayes-style belief combination):** Each module's calibrated output is converted to a **log-likelihood ratio** `LLR_i = log[ P(evidence_i | authentic) / P(evidence_i | synthetic) ]`. Reliability scales it: `LLR_i' = reliability_i · LLR_i` — an uninformative channel contributes *exactly zero*, which is the mathematically correct behavior. The posterior log-odds is `logit(prior) + Σ LLR_i'`. **This satisfies calibration, native reliability-weighting, explainability (each LLR is a contribution bar), OOD-safety (OOD → reliability→0 → LLR→0), and data-efficiency (only per-module calibration needed).** Its only weakness is the naïve conditional-independence assumption between modules.
- **The independence problem & the correction head:** modules are *not* independent (frequency, residual, and diffusion modules share information). Pure Naïve Bayes double-counts correlated evidence and becomes overconfident. **Fix:** a **LightGBM correction head** takes the per-module `(evidence, reliability, confidence)` tuples + the quality vector + the contradiction-graph features and learns the *residual* correction to the Bayesian log-odds. It models inter-module dependence and interaction effects that the independence assumption misses, while the Bayesian layer provides the calibrated, interpretable backbone. LightGBM is chosen over XGBoost for lower footprint and faster CPU inference (commodity-hardware constraint); they are otherwise interchangeable. SHAP values give per-feature attributions for the XAI layer.
- **GNN / full PGM / MoE:** each is theoretically attractive but either data-hungry (GNN), heavier to maintain (full PGM), or already *implicitly realized*: the reliability layer is a **calibrated, hand-designed mixture-of-experts gate**, and the contradiction graph captures the dependency structure a PGM would encode — without the data cost.

## 5.3 Recommended fusion architecture for ARGUS

A **three-stage cascade**:

```
Stage 1 — BAYESIAN BELIEF CORE (calibrated, interpretable backbone)
   per-module evidence → per-module calibrated LLR (Platt/isotonic, fit per module)
   reliability-scaled:  LLR_i' = reliability_i · LLR_i
   posterior log-odds  = logit(context_prior) + Σ_i LLR_i'
   → Authenticity_bayes,  plus per-cue LLR contributions (for XAI)

Stage 2 — LEARNED CORRECTION HEAD (LightGBM)
   features = [ all (e_i, r_i, c_i), quality vector, contradiction features,
               Authenticity_bayes, OOD score, evidence-coverage ]
   target   = ground-truth authentic/synthetic (LOGO-split training)
   → Authenticity_corrected  (models inter-module dependence + interactions)
   → SHAP attributions

Stage 3 — UNCERTAINTY & ABSTENTION (conformal / selective prediction)
   epistemic uncertainty from: evidence coverage, mean reliability,
        contradiction severity, OOD score, head-vs-core disagreement
   conformal prediction set → if set is non-singleton OR coverage<τ:  ABSTAIN
   → calibrated Authenticity, Trust (=epistemic quality), abstain flag
```

**Outputs of fusion:**
- **Authenticity** = the conformalized, corrected posterior `P(authentic)`.
- **Trust** = `f(mean reliability, evidence coverage, 1−contradiction_severity, 1−OOD)` — the epistemic-quality axis.
- **Risk** = `g(1−Authenticity, contradiction_severity, context_prior, harm_weight) · (clamp on low Trust → "uncertain-high-risk")`.

## 5.4 Why this is the strongest approach

- It is **calibrated by construction** (Bayesian LLRs + isotonic per module + conformal wrapper) — the metric that actually matters off-distribution.
- It is **natively reliability-weighted** — the single capability incumbents lack.
- It is **explainable at two levels** — Bayesian LLR contributions (causal, monotone) *and* SHAP (interaction-aware).
- It is **open-set safe** — OOD pushes reliability→0, evidence drops out, the system abstains rather than hallucinating confidence.
- It is **data-light** — the Bayesian core needs only per-module calibration; only the small correction head needs labels, and it is regularized and LOGO-validated.
- It runs in **milliseconds on CPU.**

**One-line summary:** *Bayesian belief combination for calibration and interpretability, a small gradient-boosted correction head for inter-module dependence, and a conformal/abstention wrapper for honest open-set uncertainty.*

\newpage
