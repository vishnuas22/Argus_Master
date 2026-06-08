# FINAL ARCHITECTURE SCORE

> *A single honest number, with every lost point itemized and justified. Scored against the ideal: a system that reliably, honestly, explainably assesses image authenticity in the open set under laundering, on commodity hardware, with open-source tools.*

## Final score: **84 / 100**

### Score breakdown by dimension

| Dimension | Weight | Score | Notes |
|---|:--:|:--:|---|
| Open-set / unseen-generator robustness | 20 | 16 | Durable-evidence backbone + abstention; −4 for residual decay of physics over time |
| Calibration & uncertainty honesty | 15 | 13 | Bayesian+conformal+decomposed uncertainty; −2 for void-under-shift caveat |
| Laundering robustness | 15 | 12 | Self-silencing reliability layer; −3 because median laundered traffic forces abstention |
| Explainability | 12 | 11 | Evidence ranking + contradictions + grounded NL + localization; −1 VLM-narrative risk |
| Reliability-awareness (core novelty) | 12 | 11 | Strong, novel; −1 sim-vs-real calibration gap |
| Buildability (3 mo, commodity HW, OSS) | 10 | 8 | Achievable but operationally heavy; −2 complexity |
| Cost / scalability | 8 | 6 | Cascade fixes most; −2 still pricier than one forward pass |
| Bias / fairness | 8 | 6 | Now measured & surfaced; −2 inherits open-data demographic skew |
| **Total** | **100** | **84** | |

## Justification of every point lost (16 points)

- **−4 — Physics/semantic durability is finite, not infinite.** Generators are improving at 3-D and semantic consistency. ARGUS slows and *honestly signals* this decay (modules down-weight themselves as calibrated accuracy falls) and is built to absorb new evidence classes — but it cannot claim a permanent moat. Points lost for the real, ongoing decay of even the durable backbone.
- **−3 — Abstention on median traffic.** On the most common real input (low-res, laundered, no provenance, no web match) ARGUS often cannot reach a confident Authenticity verdict and falls back to "high-risk-if-contradicted / low-trust / abstain + request better evidence." This is the *honest and correct* behavior and is valuable to high-stakes users, but it is a genuine capability ceiling, not a full solution, for mass-market always-answer use cases.
- **−2 — Conformal guarantees weaken under distribution shift.** The uncertainty layer is *empirically validated*, not *theoretically guaranteed*, in exactly the open-set regime where guarantees would be most desirable. We claim only what we can defend.
- **−2 — Operational complexity / cost above a single classifier.** Even with the cheap-to-expensive cascade, ARGUS is materially more complex to build, calibrate, and maintain (per-module calibration, retrieval corpus, OOD models, drift monitoring) and costlier per ambiguous image than a single forward pass.
- **−2 — Demographic/device dataset bias.** Reliability and semantic calibration inherit the Western, high-end-device, English-centric skew of open authentic datasets. The redesign *measures and surfaces* this and lowers Trust on under-represented inputs (rather than confidently erring), but does not eliminate the underlying data gap.
- **−1 — Reliability calibration sim-vs-real gap.** Degradation transfer functions begin on simulated laundering; closing the gap requires continuous production-feedback recalibration that is not free.
- **−1 — VLM hallucination residual risk.** Geometric verification and grounded-only narration sharply reduce, but do not fully eliminate, the risk of a persuasively-wrong VLM-sourced cue; reliance on it is capped but nonzero.
- **−1 — Provenance/retrieval coverage is exogenous.** ARGUS's two strongest durable signals depend on C2PA adoption and corpus coverage that ARGUS does not control; their value is rising but currently partial.

## Why not higher, why not lower

- **Why not 95+:** because no image-only system can honestly claim to detect a *perfect* future generator's *novel, laundered, provenance-less* output — the ideal is partly unattainable in principle, and any system scoring itself 95+ on this problem is overclaiming. ARGUS's refusal to overclaim is a feature.
- **Why not <70:** because ARGUS structurally fixes the seven named failures of the incumbent paradigm — single classifier (→ orthogonal fusion), generator overfitting (→ durable-evidence backbone + OOD gate), benchmark-vs-production gap (→ laundering-aware reliability), poor explainability (→ evidence chains + contradictions), no uncertainty (→ calibrated tri-axial + abstention), no adaptation to evidence quality (→ the reliability layer). It is a different *category* of system, not a better classifier.

## One-paragraph executive verdict

ARGUS reframes image authenticity from a brittle, accuracy-chasing, single-classifier detection problem into a **reliability-aware, calibrated, explainable evidence-fusion problem that is honest about what it cannot know.** Its central innovations — per-module reliability decoupled from confidence via calibrated degradation transfer functions, reliability-scaled Bayesian fusion with an open-set OOD gate and conformal abstention, and a tri-axial Authenticity/Trust/Risk output with an explicit contradiction graph — directly target the structural reasons production detectors fail. It is buildable in three months on commodity hardware with open-source tools, generalizes to unseen generators by leaning on physics/provenance/retrieval rather than fingerprints, and degrades *gracefully and honestly* rather than failing silently and overconfidently. Its limitations — finite durability of even the best evidence, abstention on the hardest laundered traffic, data-driven bias, and exogenous provenance coverage — are real, **measured, and surfaced rather than hidden**, which is itself the design's most defensible property. **Final score: 84/100.**

---

## Appendix A — The module output contract (reference schema)

```json
{
  "module": "string",
  "tier": "A | B | C",
  "evidence_score": 0.0,        // [0,1] P(authentic) implied by this cue; 0.5 neutral
  "reliability_score": 0.0,     // [0,1] from quality profile × self-consistency × (1-OOD)
  "confidence_score": 0.0,      // [0,1] internal certainty given a readable channel
  "direction": "authentic | synthetic | tamper | neutral",
  "llr": 0.0,                   // calibrated log-likelihood ratio (pre-reliability)
  "llr_weighted": 0.0,          // reliability_score * llr  (actual fusion contribution)
  "ood_score": 0.0,             // [0,1] distance from this module's training manifold
  "localization": null,         // optional heatmap / bbox / overlay
  "rationale": "string",        // plain-language, grounded
  "features": {}                // raw measurements for audit/reproducibility
}
```

## Appendix B — Fusion pseudocode (reference)

```python
# Stage 1 — Bayesian belief core
log_odds = logit(context_prior)
for m in modules:
    llr = calibrate(m.evidence_score, m.calibrator)      # isotonic/Platt, per module
    contribution = m.reliability_score * llr             # OOD/laundering -> reliability~0 -> ~0
    log_odds += contribution
    m.llr_weighted = contribution                        # store for XAI ranking
p_bayes = sigmoid(log_odds)

# Stage 2 — LightGBM correction head (durable-channel features only; monotone, regularized)
features = assemble(modules, quality_vector, contradiction_features, p_bayes, ood_summary)
p_corrected = lgbm.predict(features)
shap_values = lgbm.shap(features)

# Stage 3 — uncertainty, abstention, tri-axial output
epistemic = combine(evidence_coverage, mean_reliability,
                    contradiction_severity, ood_summary, abs(p_corrected - p_bayes))
pred_set = conformal_set(p_corrected, calib_scores)      # Mondrian / shift-weighted
abstain  = (len(pred_set) != 1) or (evidence_coverage < TAU)

authenticity = p_corrected
trust        = f_trust(mean_reliability, evidence_coverage,
                       1 - contradiction_severity, 1 - ood_summary)
risk         = f_risk(1 - authenticity, contradiction_severity,
                      context_prior, harm_weight, low_trust_flag=trust < TRUST_MIN)

return Verdict(authenticity, trust, risk, abstain,
               ranking=sorted(modules, key=lambda m: abs(m.llr_weighted), reverse=True),
               contradictions=contradiction_graph.report(),
               silenced=[m for m in modules if m.reliability_score < 0.1],
               narrative=grounded_llm_or_template(...))
```

*End of ARGUS design document.*
