# PHASE 9 — Research Contribution Analysis

> Sober assessment: what in ARGUS is genuinely novel vs. well-executed synthesis, what is publishable where, what competitors miss, and what defensible IP could emerge.

---

## 9.1 Novelty Audit (honest version)

| Component | Novel? | Prior art / delta |
|---|---|---|
| Multi-cue forensic fusion | **No** — Fontani et al. (2013, DS-theory), FRAME (2026, adaptive path routing) | ARGUS's delta is *what gates the fusion* (measured degradation), not fusion itself |
| Per-module confidence maps | **No** — TruFor ships one | TruFor's confidence is per-pixel, self-estimated, single-module |
| **Degradation-conditioned reliability calibration `r_m(d)`** | **Yes — the core claim.** Estimating a per-image laundering state and conditioning *every* module's reliability on it via offline laundering-ladder calibration has no published equivalent found in the 2024–2026 record | Closest: augmentation-robust training (implicit, unreported); TruFor (single-module, unconditioned); quality-aware face recognition (different problem, scalar quality) |
| **Conformal authenticity verdicts** | **Mostly yes.** Conformal prediction exists; deepfake-detection papers using it exist (2024). Applying it *over a reliability-gated evidence panel with degradation-stratified calibration and abstention-with-content semantics* is new as a system | The "partial verdict" output ({not camera-original} at guaranteed coverage) appears genuinely unpublished |
| Realness-first panel composition (training-free + one-class as load-bearing, fake-trained as perishable plugins) | **Partially** — ZED, RIGID, SpAN each exist | The *architectural doctrine* (perishability management, expiry-dated modules, quarterly-retrain-budgeted judge) is a systems contribution, not an algorithmic one |
| Evidence-court verdict schema (checkable claims, mandatory contradictions, unavailable-evidence disclosure) | **Yes, as an artifact standard** — no existing tool emits anything comparable | Standards-track contribution rather than a paper |
| LOGO × laundering-ladder evaluation with de-confounded encoding | **Partially** — components exist scattered; the combined protocol + de-confounding discipline as a released benchmark would be a real contribution | GenImage-confound critiques exist (2024–25); a clean benchmark fixing them is overdue |

## 9.2 Publishable Units (ranked by expected acceptance value)

1. **"Reliability-Conditioned Evidence Fusion for Open-Set Image Authenticity Assessment"** — the `r_m(d)` mechanism + conformal verdicts + LOGO×Ladder results showing reduced confident-wrong rate vs monolithic detectors. Venue: CVPR/ICCV (vision) or IEEE TIFS / WIFS (forensics — likely friendlier reviewers for systems-forensics work). The MVP evaluation *is* the experiments section.
2. **"LaunderBench: a Degradation-Aware, De-Confounded Benchmark for AI-Image Detection"** — the laundering simulator + platform presets + de-confounded splits + per-rung leaderboard. Benchmarks earn citations disproportionate to effort and establish the lab as the evaluation authority. Venue: NeurIPS D&B track.
3. **"Abstention-with-Content: Partial Conformal Verdicts for Media Forensics"** — shorter, theory-flavored: semantics and coverage analysis of partial hypothesis-set verdicts under stratified calibration, with the exchangeability-under-drift caveat treated formally. Venue: WIFS / SatML.
4. (Post-MVP) **Drift-detection-as-generator-discovery** — the Phase 6 telemetry mechanism with a real "we detected generator X at week N" case study. Needs production data; highest impact if it lands.

## 9.3 What Competitors Are Likely Missing (gap analysis, 2026)

| Competitor class | What they ship | The gap ARGUS exploits |
|---|---|---|
| Commercial detectors (Hive, Reality Defender, Sensity et al.) | Ensemble classifiers, single score, periodic retrains | No degradation conditioning (confident on thumbnails); no formal abstention; explainability = heatmap at best; perishability hidden from customers |
| Open-source detectors (HF-hub classifiers, AIorNot-style) | Single fine-tuned model | Everything in Phase 1; most still benefit from the JPEG/PNG confound |
| Academic SOTA (training-free probes, TruFor lineage) | Strong single modules | No system integration: no triage, no fusion, no verdict layer; evaluated i.i.d. or lightly augmented |
| Provenance camp (C2PA ecosystem) | Cryptographic provenance | No answer for the 97% unsigned traffic; ARGUS treats them as Tier 0 of a larger machine — complementary, not competing |
| The structural gap | — | **Nobody sells calibrated honesty.** The market consists of confidence vendors; institutions burned by them (newsrooms, courts, platforms post-incident) are the customers who will pay for "guaranteed-coverage abstention + checkable evidence." That positioning is itself the moat |

## 9.4 Intellectual Property Prospects

**Patentable (method claims, subject to counsel review):**
1. *Method for estimating per-module forensic reliability conditioned on a measured image-degradation state, via offline simulated-laundering calibration* — the cleanest, most defensible claim; detectable in competitor products (their outputs would have to vary with degradation state in the claimed way).
2. *Authenticity assessment with conformal hypothesis-set outputs and degradation-stratified calibration* (system claim combining gate + judge + wrapper).
3. *Conflict-pattern-based novel-generator detection from evidence-vector telemetry* (Phase 6.3.2–6.3.3).

**Better kept as trade secrets:** platform laundering presets (measured, perishable, costly to reproduce); reliability-curve tables themselves; the real-corpus curation recipe.

**Deliberately open (strategic):** verdict schema (standard-setting — the schema becoming an industry contract is worth more than licensing it); LaunderBench (authority + recruiting); module contract spec. Openness here also feeds the C2PA/CAI policy conversation where ARGUS wants a seat.

**Defensive note:** the space has active patent filings from large platforms; a freedom-to-operate scan on "ensemble deepfake detection" claims is a week-1-of-funding task, not a someday task.
