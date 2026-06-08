# B. RESEARCH ROADMAP

## V1 — 3 months · 1.5 FTE · consumer GPU optional
- **Wk 1–3:** modules 0–6 + frozen contract (with `reliability_var`) + audit store. **Gate G1:** all 7 emit valid tuples on 1k images.
- **Wk 4–6:** degradation sweep → fit transfer functions (reliability = fitted accuracy-vs-quality curve per module). **Gate G2:** monotone, cross-validated transfer functions with held-out fit error reported.
- **Wk 7–9:** reliability-aware mixture pool + isotonic per-module calibration + threshold abstention (coverage τ + mean-reliability τ). **Gate G3:** pristine ECE < 0.05.
- **Wk 10–12:** XAI (LLR ranking + heatmaps + **templated** narrative, no LLM in v1) + React console + **run Kill Test (C)**. **Decision gate G4 (go/no-go for V2):** reliability ablation shows statistically significant calibration gain on the hard/laundered subset. *Fail → publish the negative result and stop.*

## V2 — months 4–9 · 3–4 FTE
- Domain-scoped **CLIP+FAISS retrieval** with **OpenTimestamps corroboration + pHash-collision monitoring**; OOD/Mahalanobis gate live.
- **VLM non-scoring advisory** only + image-borne prompt-injection defense.
- **Per-subgroup error reporting as a hard release gate**; real-platform recalibration pipeline.
- **Frozen / versioned forensic mode** (pinned models, snapshotted corpus, deterministic re-run).
- **Gate:** per-subgroup FP disparity < 2× before any external pilot.

## V3 — months 10–21 · 5–6 FTE + governance
- Reposition as a **layer**: hardware-attestation hooks, network/propagation signals, multi-frame & cross-source corroboration, **video**, prospective per-generator testing, Daubert error-rate publication, independent bias audit, liar's-dividend safeguards (asymmetric thresholds against false "fake").

\newpage


# C. KILL TEST — the minimum experiment that proves ARGUS fundamentally wrong

## Primary — the reliability ablation (cheap, decisive; runnable in V1)
1. Fix all 7 modules + isotonic calibrators.
2. Test set: ≥3 held-out generator families × ≥1,000 synthetic + ≥1,000 authentic-wild, each through the full laundering grid. **Hard subset** = {no C2PA, no retrieval hit, JPEG-Q ≤ 60 or screenshot}.
3. Three arms, identical inputs:
   - **A-full:** reliability-aware pool.
   - **A-null:** all reliabilities forced to 1 (plain pooled evidence, no self-silencing).
   - **CLIP-TS:** temperature-scaled CLIP-ViT-B/32 probe.
4. Compute laundered-ECE, Brier, hard-subset selective risk; 1000× bootstrap CIs.
5. **Pre-registered decision rule:** the thesis **survives** only if **A-full beats both A-null and CLIP-TS** on laundered-ECE **and** hard-subset selective risk with **non-overlapping 95% CIs**.
   - A-full ≈ A-null ⇒ the reliability layer (the sole novelty) is **empirically null** ⇒ ARGUS = a forensic ensemble; **thesis dead.**
   - CLIP-TS ≈ A-full on calibration ⇒ the differentiator is gone.

## Secondary — durability falsification (one afternoon)
Take one recent geometry-/physics-aware generator; measure physics/shadow-module AUROC on its outputs. **AUROC ≤ 0.60 ⇒ the "durable Tier-A backbone" is already chance-level ⇒ open-set value proposition falsified**, independent of everything else.

Both run on a single consumer GPU in days. **Running C before building V2 is the single highest-leverage action in the program.**

\newpage
