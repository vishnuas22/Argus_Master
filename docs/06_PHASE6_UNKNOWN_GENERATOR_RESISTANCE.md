# PHASE 6 — Unknown-Generator Resistance

> Scenario: month 0 after ARGUS ships, a generator G* appears — new architecture, never seen in any training or calibration set, spectrally clean, photorealistic. What still works, what breaks, and how does ARGUS *know* it's facing something new?

---

## 6.1 The Threat Model, Made Precise

G* is assumed to: (a) defeat all generator-specific fingerprints by construction; (b) produce a cleaner spectrum than its predecessors; (c) emit images that humans cannot distinguish from photos; (d) be deployed by adversaries who also launder outputs through social platforms. G* is *not* assumed to: simulate full ray-traced physics, fabricate internet history, possess valid capture-device provenance, or perfectly reproduce the statistical manifold of camera-pipeline imagery (only to approach it).

These asymmetries are exactly where ARGUS's panel was aimed (Phase 2 axiom A7, Phase 3 family synthesis). What follows is the per-module audit.

## 6.2 Evidence Audit Under G*

### Remains useful (by design)

| Module | Why it survives G* | Expected degradation |
|---|---|---|
| **Tier 0 — provenance** | G* output has no valid capture-chain manifest; ecosystem images increasingly do. Also: if G*'s vendor signs outputs (the responsible-AI trend), G* images *self-identify* | None; improves with adoption |
| **I — retrieval/context** | G* cannot fabricate a pre-existing internet history; novel-image-with-viral-claims is itself a context signal | None |
| **E — real-distribution probe** | Never trained on any fake; measures distance from *real*, not similarity to known-fake. As G* → real, signal weakens **monotonically and gracefully — it never inverts** (a fake never starts looking *more real than real*) | Gradual signal shrinkage; no cliff |
| **F — perturbation-sensitivity probe** | Training-free; exploits representation-geometry differences between natural images and *any* learned generator's manifold — a property of how generators synthesize, not of which one | Unknown but non-targeted; the 2026 literature shows these probes transfer across a dozen+ generator families |
| **G — physics/geometry** | G* would need physical simulation in its loop to guarantee shadow/reflection/perspective coherence; no known architecture class does | Slow decline; per-image recall unchanged |
| **B + Tier 1 — compression history** | Measures the file's transmission life, not its origin; also catches the "pristine PNG claiming to be a phone photo" inconsistency | None |
| **A — metadata (asymmetric mode)** | Absence/inconsistency of plausible acquisition metadata remains a (weak, forgeable) signal | None structurally |

### Becomes obsolete (by design, contained)

| Component | Failure mode under G* | Containment |
|---|---|---|
| Generator-fingerprint plugins (Phase 3 #10) | Simply don't fire on G* | They are *attribution* metadata, never load-bearing; silence ≠ "real" |
| Latent-inversion plugins (#11) | G* isn't on any known family's manifold | Off by default; output labeled "family attribution", not authenticity |
| **C — spectral probe** (partially) | G*'s cleaner spectrum shrinks the signal | Degrades to neutral (e≈0), not to wrong; quarterly re-calibration restores what's restorable |
| **H — semantic plausibility** (partially) | G* makes fewer hand/text/logic errors | Long-tail world-logic persists; module re-weighted by fusion retraining |
| **Fusion judge calibration** (the subtle one) | The LightGBM's learned weightings reflect pre-G* fake statistics | This is the *managed* perishable: tiny model, quarterly retrain, plus the safeguards below |

**The structural claim:** ARGUS under G* loses *resolution*, not *integrity*. The panel's surviving majority (provenance, retrieval, realness probes, physics, compression history) was selected precisely for generator-independence; the perishable minority is labeled, contained, and cheap to refresh.

---

## 6.3 Knowing That You Don't Know — the Uncertainty Machinery

The harder requirement: ARGUS must *detect that it is facing G**, not merely happen to survive it. Four layers:

### 6.3.1 Conformal abstention (per-image, guaranteed)
G* images produce atypical evidence vectors → high nonconformity → larger prediction sets → abstention at guaranteed coverage. **Honest caveat:** conformal validity assumes exchangeability with calibration data; G* violates it. What conformal still provides under shift: sets *widen* in practice (graceful), and coverage on non-G* traffic is untouched. What it cannot provide: guaranteed coverage *on G* itself* until G* samples enter recalibration. This gap is exactly why the next three layers exist.

### 6.3.2 Evidence-pattern density monitoring (per-image, heuristic)
A lightweight one-class model (Mahalanobis / isolation forest) over the *evidence vector itself* `[e, r, c]×9 ⊕ K`, fit on historical traffic. G* tends to produce a **novel conflict signature** — e.g., realness probes mildly negative + spectral neutral + semantics clean, a combination rare in history. Per-image output: `novelty_score`, reported in the verdict's risk section ("evidence pattern unlike 99.7% of reference traffic").

### 6.3.3 Population-level drift detection (fleet, statistical)
Production telemetry monitors distributions of: per-module evidence scores, conflict-pattern frequencies, abstention rate, conformal set-size, per-degradation-bucket score means. CUSUM / population-stability tests alarm on shifts. **A new generator appears as a population anomaly before any single image is confidently caught** — abstention rate climbs, a new conflict cluster emerges. This converts "we were fooled for eight months" (the classifier-era failure) into "anomaly cluster #14 flagged in week 2, samples queued for analysis."

### 6.3.4 The recalibration loop (organizational, scheduled)
Flagged clusters → human/automated triage → confirmed G* samples → (a) fusion judge retrain (minutes of CPU), (b) conformal recalibration (seconds), (c) optional new fingerprint plugin for G* attribution, (d) laundering-ladder re-run to refresh `r_m(d)` if G* interacts oddly with degradation. Total response cost: hours-to-days, by architectural intent — compare months-scale full-model retraining in classifier-centric systems.

---

## 6.4 Designed Behavior Summary Under G*

| Time | System behavior |
|---|---|
| Day 0 | G* images get: weakened-but-correctly-signed realness/physics evidence; neutral perishables; wider conformal sets; elevated novelty scores. Many abstain *with content* ("camera-original excluded at 95%; AI vs manipulated indeterminate") — already actionable |
| Week 1–2 | Drift monitors flag a novel evidence-pattern cluster; samples queued |
| Week 2–4 | Fusion + conformal recalibrated on confirmed samples; resolution restored; optional G* attribution plugin shipped |
| Structural floor | At no point does ARGUS systematically output confident "camera-original" verdicts for G* images — the surviving evidence majority cannot produce that pattern, and the conflict machinery flags the attempt |

That last row is the design's central promise: **the worst case is honest ignorance, never confident error.**
