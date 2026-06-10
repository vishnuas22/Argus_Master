# PHASE 5 — Evidence Fusion

> Given nine module outputs `{e_m, r_m, c_m}` + degradation state `d` + conflict features, produce a calibrated, abstention-capable verdict. This phase compares the seven candidate strategies honestly and derives the recommendation.

---

## 5.1 What the Fusion Layer Must Do (requirements derived from Phases 1–4)

- **R1.** Use reliability: weight evidence per-image by `r_m(d)` — including modules being entirely absent.
- **R2.** Capture interactions: "spectral peak matters only if no resize detected" is a *conditional* rule; linear weighting cannot express it.
- **R3.** Output calibrated probabilities over multiple hypotheses: {camera-original, AI-generated, manipulated} — not binary (Phase 2 axiom A1).
- **R4.** Support formal abstention (axiom A5).
- **R5.** Train from *small* data (a few thousand fused examples) — the fusion layer is the one fake-supervised component, so it must be tiny and cheap to retrain quarterly (axiom A2: perishability is managed, not denied).
- **R6.** Be auditable: a forensic product cannot have an unexplainable judge (axiom A6).
- **R7.** Run in milliseconds on CPU.

---

## 5.2 Candidate-by-Candidate Analysis

### 5.2.1 Weighted voting / weighted averaging
`score = Σ w_m·r_m·e_m / Σ w_m·r_m`.
- **For:** trivially auditable, zero training, naturally reliability-aware, no overfitting possible.
- **Against:** fails **R2** (no interactions) and destroys conflict information — {+0.9, −0.9} and {0.0, 0.0} both average to 0, though they are opposite epistemic states (Phase 4.4). Fails R3 (single scalar).
- **Verdict:** rejected as the judge; **retained as the mandatory degraded-mode fallback** (when fusion-model trust is itself in doubt, e.g., severe drift detected) and as the baseline every learned fusion must beat in evaluation.

### 5.2.2 Bayesian fusion (naive Bayes / explicit likelihood models)
`P(H|E) ∝ P(H)·Π P(e_m|H)`.
- **For:** principled, transparent, handles missing evidence natively (drop the factor), priors make base rates explicit (Phase 1.2 L5).
- **Against:** conditional-independence assumption is **badly violated** — modules C, E, F share the frequency domain and even a backbone; ELA and DQ share JPEG mathematics. Correlated evidence double-counts, producing exactly the overconfidence ARGUS exists to eliminate. Modeling the full joint `P(e_1..e_9|H, d)` honestly requires the data we don't have (R5).
- **Verdict:** rejected as implemented mechanism; its *structure* survives — explicit priors and likelihood-ratio reporting appear in the verdict layer (Phase 7 reports per-evidence likelihood ratios for auditability, R6).

### 5.2.3 / 5.2.4 LightGBM vs XGBoost (gradient-boosted trees over the evidence feature vector)
Features: `[e_m, r_m, c_m] × 9 ⊕ d ⊕ conflict features ⊕ availability mask` (~40 dims).
- **For:** the workhorse for small-N tabular learning — exactly this regime (R5); trees natively express conditional logic (R2: "if s_resize>0.3 then ignore e_spectral"); handles missing values natively (R1); milliseconds inference (R7); SHAP gives exact per-feature attributions (R6); monotonic constraints (e.g., fused authenticity non-decreasing in provenance validity) inject domain priors and curb overfitting.
- **Against:** needs labeled fused examples — the one place fake-data bias enters (mitigations: tiny model [≤200 trees, depth ≤4], monotonic constraints, leave-one-generator-out *fusion* validation, quarterly retrain budget ~minutes of CPU).
- **LightGBM vs XGBoost:** functionally equivalent at this scale. LightGBM chosen: faster training, leaf-wise growth slightly better on small wide data, native categorical support, lighter dependency. A two-line swap if ever needed — a non-decision.

### 5.2.5 Graph Neural Networks
Evidence-as-graph, message passing between module nodes.
- **Against (decisive):** there is **no natural graph** — nine modules with known, fixed relationships are a feature vector, not a topology; GNNs are data-hungry (violates R5) and weakly auditable (violates R6); everything a GNN could learn here, trees over ~40 dims learn with 100× less data.
- **For (narrow future case):** if ARGUS later fuses *spatially-localized* evidence across image regions (region-graph of per-patch findings), a GNN becomes defensible. Post-MVP research track, not the judge.

### 5.2.6 Probabilistic graphical models (hand-built Bayes net over evidence + latent process variables)
- **For:** maximum auditability (the graph *is* the explanation); encodes expert knowledge ("resize causes spectral peaks") explicitly; principled missing-data handling.
- **Against:** structure specification is brittle expert labor; parameter estimation still needs the joint data; inference machinery is heavier than trees; in practice dominated by GBM+SHAP on every requirement except philosophical purity.
- **Verdict:** rejected for v1; the *causal structure documentation* it would require is written anyway (module dependency notes feeding monotonic constraints + conflict patterns).

### 5.2.7 Mixture of Experts
Learned gating network routes each input to relevant experts; the 2026 trend (FRAME's adaptive forensic-path routing) validates the *concept*.
- **Key insight:** **ARGUS already is a mixture of experts — with a physically-grounded, auditable gate.** The reliability gate `r_m(d)` *is* the routing function; unlike a learned gate it requires no training data, extrapolates to unseen degradation states by monotonicity, and is fully explainable ("Noiseprint excluded: 3× recompression detected"). A *learned* gate would re-introduce exactly the small-data, black-box problems R5/R6 forbid.
- **Verdict:** adopted **structurally** (gating by reliability), rejected as an additional learned component.

---

## 5.3 Comparison Matrix

| Strategy | R1 reliab. | R2 interact. | R3 multi-hyp | R4 abstain | R5 small-data | R6 audit | R7 speed | Verdict |
|---|---|---|---|---|---|---|---|---|
| Weighted voting | ✔ | ✘ | ✘ | ✘ | ✔✔ | ✔✔ | ✔✔ | fallback + baseline |
| Bayesian (naive) | ✔ | ✘ | ✔ | ~ | ✔ | ✔✔ | ✔✔ | structure reused in verdict |
| **LightGBM** | ✔ | ✔✔ | ✔ | via wrapper | ✔✔ | ✔ (SHAP+monotonic) | ✔✔ | **core judge** |
| XGBoost | ✔ | ✔✔ | ✔ | via wrapper | ✔✔ | ✔ | ✔✔ | equivalent alternate |
| GNN | ~ | ✔✔ | ✔ | ~ | ✘✘ | ✘ | ✔ | rejected (no graph, no data) |
| PGM | ✔ | ✔ | ✔✔ | ~ | ✘ | ✔✔ | ✔ | rejected (brittle to build) |
| MoE (learned gate) | ✔✔ | ✔✔ | ✔ | ~ | ✘ | ✘ | ✔ | adopted structurally, gate = r_m(d) |

---

## 5.4 The Recommended Architecture: Reliability-Gated Conformal Stacking

```
 {e_m, r_m, c_m} ×9,  d,  conflict K, patterns
        │
        ▼
 ① RELIABILITY GATE (the "MoE gate", physically grounded)
    drop modules with r_m < 0.25 → availability mask
    (excluded modules listed in verdict with reasons)
        │
        ▼
 ② STACKED JUDGE — LightGBM, 3-class softmax
    {camera-original, AI-generated, manipulated}
    ≤200 trees, depth ≤4, monotonic constraints,
    trained leave-one-generator-out, retrained quarterly
        │
        ▼
 ③ ISOTONIC CALIBRATION (per class, on held-out calibration split)
    → honest probabilities (ECE-validated)
        │
        ▼
 ④ CONFORMAL WRAPPER (split conformal / RAPS, via MAPIE)
    nonconformity = 1 − p_true  (+ conflict-inflation term)
    calibration stratified by degradation bucket
    user sets risk α (e.g., 0.05 or 0.10)
    → prediction SET with guaranteed ≥(1−α) coverage
       |set| = 1 → decisive verdict
       |set| > 1 → ABSTAIN-with-content ("AI-generated or
                   manipulated; camera-original excluded at 95%")
        │
        ▼
 ⑤ VERDICT BUILDER (Phase 7)
    scores + SHAP-derived evidence ranking + contradictions
    + per-evidence likelihood ratios + unavailable-evidence list
```

**Why the conformal wrapper is non-negotiable.** It is the only component that converts "detector" into "assessor" with a mathematical guarantee: under exchangeability with the calibration set, the true hypothesis lies in the emitted set with probability ≥ 1−α — *regardless of how wrong the judge model is*. Partial abstentions ("not camera-original, at 95%") are often exactly the actionable answer a trust & safety analyst needs even when full attribution is impossible. The known weakness — exchangeability breaks under drift — is mitigated by degradation-stratified calibration and treated as a monitored, periodically re-calibrated assumption (Phases 6, 10), not ignored.

**Failure-containment property.** Worst case (judge badly mis-trained on a biased fusion set): ① still gates impossible evidence out, ④ still guarantees coverage *on calibration-like data* and widens sets under nonconformity, ⑤ still discloses raw per-module evidence — so a human can overrule the judge by reading the witnesses directly. No single component's failure produces a silently confident wrong verdict, which was failure F1's lesson.
