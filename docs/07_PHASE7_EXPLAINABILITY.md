# PHASE 7 — Explainability (XAI) System

> Requirement: production-grade explainability for three distinct audiences — machines (policy engines), analysts (T&S / journalists), and laypeople — without ever generating an unverifiable claim. Explainability here is an *architectural property* (named evidence, Phase 2 axiom A6), not a post-hoc visualization.

---

## 7.1 The Verdict Schema (machine-readable contract)

Every assessment emits one `verdict.json`:

```json
{
  "verdict_id": "argus-2026-06-14-000042",
  "schema_version": "1.0",
  "input": {
    "sha256": "…", "dimensions": [1024, 768], "format": "jpeg",
    "degradation_state": {
      "jpeg_quality_est": 71, "recompression_generations": 2,
      "resize_factor_est": 0.5, "screenshot_probability": 0.08,
      "effective_resolution": 512, "evidence_capacity": "MODERATE"
    }
  },
  "verdict": {
    "hypothesis_set": ["ai_generated"],
    "abstained": false,
    "probabilities": {
      "camera_original": 0.04, "ai_generated": 0.87, "manipulated": 0.09
    },
    "authenticity_score": 0.04,
    "trust_score": 0.71,
    "risk_score": 0.18,
    "conformal": { "alpha": 0.05, "set": ["ai_generated"], "calibration_stratum": "moderate_degradation" }
  },
  "evidence_ranking": [
    {
      "rank": 1, "module": "real_distribution_probe", "direction": "synthetic",
      "evidence_score": -0.78, "reliability": 0.74, "confidence": 0.81,
      "shap_contribution": -0.31, "likelihood_ratio": 8.2,
      "artifacts": [{ "type": "embedding_outlier",
        "checkable_claim": "Image embedding lies outside the 99.2th percentile of distances to its 50 nearest real-image neighbors (reference set v3, n=1.2M)",
        "visual": "artifacts/nn_exemplars_42.png" }]
    },
    {
      "rank": 2, "module": "physics_geometry", "direction": "synthetic",
      "evidence_score": -0.55, "reliability": 0.88, "confidence": 0.62,
      "shap_contribution": -0.19, "likelihood_ratio": 4.1,
      "artifacts": [{ "type": "shadow_inconsistency",
        "checkable_claim": "Cast-shadow lines from the two foreground figures intersect at incompatible light-source positions (angular disagreement 38°, threshold 12°)",
        "visual": "artifacts/shadow_lines_42.png",
        "location": { "regions": [[120,340,260,520],[610,300,720,480]] } }]
    }
  ],
  "contradictions": [
    {
      "modules": ["semantic_plausibility", "real_distribution_probe"],
      "description": "Semantic analysis found no world-logic anomalies (hands, text, object interactions all plausible) while statistical evidence indicates synthesis",
      "interpretation": "Consistent with a high-quality full generation; semantic cleanliness does not outweigh statistical evidence",
      "conflict_contribution": 0.12
    }
  ],
  "unavailable_evidence": [
    { "module": "noise_residual", "reason": "reliability 0.14 below floor: ≥2 recompression generations suppress residual traces" },
    { "module": "provenance_c2pa", "reason": "no manifest present (absent in ~97% of 2026 web images; weak evidence either way)" }
  ],
  "explanation": { "summary": "…", "detail": "…", "audience": "analyst" },
  "meta": { "module_versions": {"…": "…"}, "fusion_model": "judge-2026Q2", "reliability_curves": "lc-v7", "total_compute_ms": 4180 }
}
```

Design rules embedded in the schema: every score traces to named modules; every artifact carries a `checkable_claim` a human can independently verify; missing evidence is disclosed with reasons; contradictions are mandatory, never averaged away; all model/calibration versions are pinned for reproducibility (forensic chain-of-custody discipline).

---

## 7.2 The Three Top-Level Scores — Exact Semantics

Three scores because users conflate three different questions; separating them is the UX core.

- **`authenticity_score` ∈ [0,1] — "What does the evidence say?"**
  Calibrated `P(camera_original)` from the fusion stack. 0.04 = evidence points strongly away from camera-original.

- **`trust_score` ∈ [0,1] — "How much should you trust this assessment?"**
  *Meta-*evidence quality: monotone combination of (i) evidence capacity from `d`, (ii) panel coverage (Σ reliabilities of surviving modules / total), (iii) conformal set size, (iv) novelty score (Phase 6.3.2), (v) unresolved conflict mass. A pristine RAW judged by nine witnesses → 0.95; a thumbnail judged by three → 0.4 *regardless of which way the verdict points*. This is the number that prevents the classic failure of confident garbage on degraded inputs (F4/F7).

- **`risk_score` ∈ [0,1] — "How likely is it that someone is actively gaming this assessment?"**
  Adversarial-posture indicators: metadata-forensics mismatches (claimed camera vs pipeline traces), implausible cleanliness, laundering severity beyond organic norms, evidence-pattern novelty, provenance-validation *failures* (vs mere absence). Low trust + high risk reads "we can't see much, *and the blindness looks deliberate*" — for an analyst, often the most important sentence in the report.

Worked triples: `(0.9, 0.9, 0.05)` = authentic, well-evidenced, calm. `(0.04, 0.71, 0.18)` = AI-generated, decent evidence, no gaming. `(0.5, 0.2, 0.7)` = can't tell, evidence destroyed, destruction looks intentional → escalate to human regardless of the 0.5.

---

## 7.3 Explanation Generation — Grounded, Not Generated

**Hard rule: no free-running LLM in the explanation path.** A forensic explanation that hallucinates one artifact is a product-ending liability (and inadmissible). The pipeline:

1. **Template skeletons per verdict pattern** (decisive-synthetic, decisive-authentic, abstain-degraded, abstain-conflict, manipulated-localized, provenance-resolved, …) — written once by humans, legally reviewable.
2. **Slot filling exclusively from the verdict JSON** — every sentence maps 1:1 to a schema field; the `checkable_claim` strings (authored per artifact-type by module developers) are the only factual content.
3. **Audience renderers:** *analyst* (full detail + likelihood ratios + visuals), *journalist* (plain language + what-to-verify-yourself checklist), *API consumer* (JSON only).
4. **Optional LLM polish, constrained:** an LLM may *rephrase* the filled template for fluency under a verifier that checks the output is entailed by the JSON (claim-matching against schema fields); on any mismatch, fall back to the raw template. Off by default.

Example analyst summary (auto-generated from the JSON above):

> **Assessment: AI-generated (87% · conformal 95% set: {AI-generated}) · Trust: moderate-high (0.71) · Gaming risk: low (0.18).**
> The image has been recompressed twice and downscaled ~2×, leaving moderate evidence capacity; sensor-level analysis was therefore excluded (reliability 0.14). Of the seven available evidence streams, five indicate synthesis. Strongest: the image's deep-feature signature falls outside the 99th percentile of real-photograph neighborhoods (LR ≈ 8.2), and cast-shadow geometry implies two incompatible light sources (38° disagreement — see overlay, independently verifiable). Contradiction noted: no semantic anomalies were found, consistent with a high-quality generator rather than with authenticity. No provenance manifest is present (uninformative in 2026). **Verify yourself:** shadow-line overlay `shadow_lines_42.png`; nearest-real-neighbor panel `nn_exemplars_42.png`.

## 7.4 Visual Evidence Layer

Per-module overlay artifacts, each tied to a `checkable_claim`: compression-history heatmaps (ELA/ghost/DQ) with localization boxes; residual anomaly + confidence maps (module D); polar spectrum plots with the calibrated real-image envelope drawn (module C); shadow/reflection construction lines drawn on the image (module G — the single most persuasive artifact for humans); nearest-real-neighbor exemplar grids (module E); OCR crops of garbled text (module H). UI principle: the image is the canvas; evidence toggles as layers; every layer's claim is one click from its numeric basis.

## 7.5 Why This Is Implementable (not XAI vaporware)

Every element above is assembled from things the architecture already produces: SHAP values are exact for tree models (milliseconds); reliability curves and conformal sets are intrinsically interpretable objects; artifacts/claims are authored by module developers as part of the module contract (Phase 4.1) — the explanation layer *composes* existing structure rather than reverse-engineering a black box. That is the payoff of choosing named-evidence architecture in Phase 2: post-hoc XAI tries to extract explanations that were never there; ARGUS's explanations are the system's actual working notes.
