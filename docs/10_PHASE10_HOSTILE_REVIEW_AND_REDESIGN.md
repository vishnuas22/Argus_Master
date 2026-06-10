# PHASE 10 — Hostile Peer Review, Redesign, and Final Score

> Role: Reviewer #2 with tenure and a grudge. The task is to break ARGUS. Then: redesign against every break that lands. Then: score the post-redesign architecture out of 100, justifying every point lost.

---

## 10.1 Attack: Hidden Assumptions

**H1. "The real-image distribution is stationary." It is not.**
Computational photography is *synthesizing* the real class: night-mode stacking, AI-HDR, on-device generative zoom (2025–26 flagship phones ship diffusion-based zoom enhancement), skin "beautification" by default. Module E's real-corpus (COCO/RAISE/Dresden) skews 2010s; a 2026 phone photo may legitimately sit far from its neighborhoods → systematic false "synthetic" evidence on *exactly the most common authentic images*. The realness-first doctrine inherits a drift problem it claimed the fake class had.
**Severity: HIGH. The single best attack in this review.**

**H2. "The laundering simulator covers real degradation pipelines."**
Platform pipelines are unobserved, proprietary, and change without notice (codec swaps to AVIF/HEIC, ML-based "enhancement-on-upload"). Reliability curves calibrated on `launder-v1` mis-estimate `r_m(d)` on unmodeled pipelines — and a mis-calibrated reliability is *worse* than none, because the system trusts blind witnesses with mathematical confidence. **Severity: HIGH.**

**H3. "Degradation state `d` is identifiable from the image."**
An adversary can *forge the degradation state*: synthesize an image, then add fake sensor noise + plausible JPEG history + camera-consistent quant tables ("anti-forensics", a literature ARGUS's design never names). Triage reads "lightly processed phone photo", grants high reliability to modules the adversary has specifically prepared to fool. The triage layer is itself an attack surface. **Severity: HIGH.**

**H4. "Modules are independent witnesses."**
C, E, F share DINOv2 and/or the frequency domain; B and Tier 1 share JPEG math. The "panel of independent witnesses" rhetoric overstates diversity; correlated failures (e.g., anything that fools DINOv2 representations fools E *and* F) are understated in Phases 2–5, even if the GBM partially learns them. **Severity: MEDIUM.**

**H5. "Conformal guarantees mean what users will think they mean."**
Marketing will say "guaranteed 95% coverage"; the guarantee is *marginal, on-calibration-distribution, under exchangeability* — precisely the conditions adversarial traffic violates (admitted in 6.3.1, but the verdict schema still prints "guaranteed"). Overclaiming risk: legal and scientific. **Severity: MEDIUM.**

## 10.2 Attack: Scientific Weaknesses

**S1. "Authenticity" is never operationalized.** Three-class {camera-original, AI-generated, manipulated} shatters on the real continuum: AI-denoised real photo? Real photo through AI-upscaler? Composite of two real photos? Night-mode stack (multi-frame synthesis of a real scene)? The ontology decides the labels, the labels decide everything downstream, and Phase 2 never defines the cut points. **Severity: HIGH — this is a validity threat to all reported metrics.**

**S2. Evaluation circularity.** Reliability curves are calibrated on `launder-v1`; evaluation runs on `launder-v1`. The headline "degradation-aware" results are partially self-graded. Independent test laundering (held-out *real* platform round-trips, not simulator rungs) is mentioned nowhere in Phase 8. **Severity: HIGH.**

**S3. Physics modules: anecdote-grade recall.** Shadow-line analysis fires on a minority of images (clear cast shadows, discernible contact points); published projective-geometry forensics is largely *manual* — automated versions have high false-positive rates on soft/multi-source lighting. Phase 3 scores it REC 9 on robustness while quietly conceding recall; the fusion layer will learn to ignore a module that rarely fires and sometimes fires wrong, making the "laundering-resistant core" mostly aspirational at MVP. **Severity: MEDIUM.**

**S4. Dataset bias laundering via de-confounding.** The de-confounding pass (8.3) equalizes *encoding*, not *content*: GenImage/Synthbuster fakes are prompt-generated (aesthetic, centered, well-lit subjects) while COCO reals are cluttered snapshots. A content-level confound (composition, subject statistics) survives re-encoding and inflates module E exactly as the JPEG confound inflated published classifiers. **Severity: MEDIUM-HIGH.**

**S5. Fusion training set size.** "A few thousand fused examples" across 3 classes × generators × degradation rungs × conflict patterns is thin; the GBM will see some cells of that hypercube empty (e.g., manipulated × heavy-laundering × physics-available), and conformal strata thin out the calibration further. Coverage claims per-stratum (8.5: ±2%) need n per stratum the plan never budgets. **Severity: MEDIUM.**

## 10.3 Attack: Engineering & Operational Risks

**E1. TruFor license.** Research-only. The MVP's strongest localizer may be unshippable commercially; the "retrain Noiseprint-style on open data, 3 GPU-days" contingency is hand-waved — Noiseprint++ training requires careful camera-ID curriculum and is a known-fiddly reproduction. **Severity: MEDIUM (plan-level), LOW (architecture-level — module D is swappable by design).**

**E2. The 1.2M-image FAISS index + DINOv2-CPU on commodity hardware.** ~3 s/image embedding is fine for analyst workflows, fatal for feed-scale screening; memory for index + model + panel concurrency strains 32 GB under load. The brief said commodity hardware; the *throughput envelope* (analyst tool vs platform filter) is never stated, and the two products have different architectures. **Severity: MEDIUM.**

**E3. Maintenance treadmill is budgeted optimistically.** Quarterly retrain assumes fresh labeled fakes of current generators continuously arrive. Who generates them, with what budget, under what licenses (FLUX outputs have usage restrictions)? The recalibration loop (6.3.4) is an *organizational* commitment dressed as an architectural property. **Severity: MEDIUM.**

**E4. Telemetry-based drift detection requires traffic.** An on-prem/court deployment sees hundreds of images, not millions; population-level CUSUM never triggers. Phase 6's safety story silently assumes SaaS-scale data. **Severity: LOW-MEDIUM.**

## 10.4 Redesign — Answering Every Landed Hit

| Attack | Redesign response (now part of the architecture) |
|---|---|
| **H1 real-class drift** | (a) Real corpus becomes a *versioned, dated, stratified* asset with a mandatory **modern-mobile stratum** (recent-phone photos incl. night mode/HDR, sourced via open collections + partner contributions); (b) module E reports neighborhood *era/device composition* ("nearest neighbors are 2014 DSLR images") so semantic-vs-temporal outlierness is distinguishable; (c) scheduled corpus refresh enters the same quarterly cycle as the judge; (d) eval gains a **modern-phone-authentic** stratum with its own false-positive budget (≤5%) |
| **H2 simulator gap** | (a) **Held-out real-platform round-trips** (manually pushed through WhatsApp/X/Telegram/Instagram quarterly) become the *test* laundering set — simulator is for training/calibration only, breaking the circularity (also answers S2); (b) reliability curves get uncertainty bands; outside calibrated `d`-support, modules are gated by default (fail-closed, toward abstention); (c) pipeline-change canaries: a fixed probe-image set re-uploaded monthly, alerting on platform codec changes |
| **H3 anti-forensics** | (a) Named explicitly as a threat class; module A gains **consistency cross-checks as first-class evidence**: claimed-camera vs noise-floor vs quant-table vs CFA-remnant mutual compatibility (forging all *jointly consistently* is the expensive part — raise adversary cost, axiom A7); (b) `risk_score` gains anti-forensics indicators (noise too textbook, history too clean for claimed provenance); (c) honest concession in docs: a sufficiently resourced adversary who forges joint consistency defeats statistical tiers — surviving defense is provenance + retrieval + physics, which is *why they are the strategic core* |
| **H4 correlated witnesses** | (a) Module F's perturbation probe re-based on a **second backbone** (e.g., ConvNeXt or MAE-pretrained ViT) so E and F fail differently — cost: one extra forward pass; (b) module-correlation matrix estimated during calibration and published in the verdict's methods appendix; conflict metric K reweighted by estimated correlation |
| **H5 conformal overclaim** | Verdict field renamed `conformal: {nominal_coverage, validity_scope: "calibration-distribution; see drift_status"}`; `drift_status` (from 6.3.3, or "insufficient-traffic") rides along in every verdict; documentation rule: the word "guaranteed" always carries its conditions |
| **S1 ontology** | An explicit **authenticity ontology document** becomes deliverable #0: process taxonomy (capture → in-camera computational processing → post-hoc enhancement → semantic edit → partial synthesis → full synthesis) with the 3 MVP classes defined as *regions* of it and edge cases (night mode, AI-upscale) assigned by written rule; labels in all datasets re-audited against it. The 4th output "AI-processed real" enters post-MVP roadmap |
| **S2 eval circularity** | Solved by H2(a): simulator ≠ test laundering. Added: one *unseen platform* held out entirely (e.g., calibrate without Instagram; test on it) |
| **S3 physics recall** | Re-scored honestly: module G's MVP value = *high-precision, low-recall tiebreaker* + UI-side assisted-manual tooling (draw-the-lines workflow for analysts — automation assists, human confirms). REC stays high for the *post-MVP* horizon; MVP exit criteria no longer depend on G recall |
| **S4 content confound** | Fake corpus regenerated with **COCO-caption-matched prompts** (content distribution matched to real corpus); plus an eval slice of *real* aesthetic/staged photography (Unsplash-style) to measure module E's composition bias directly |
| **S5 fusion data thinness** | Fusion set budget raised to ~20k fused examples (the calibration sweep already produces module outputs at scale; labeling is free — it's the same corpora); conformal strata coarsened to meet n≥1000/stratum; per-stratum coverage CIs reported instead of point claims |
| **E1 TruFor license** | Week-1 license audit already in plan; contingency upgraded from hand-wave to scoped task with acceptance test (open-data residual net matching ≥90% of TruFor's localization F1 on CASIA/IMD2020) and budget line ($300 GPU rental); module D ships behind a license flag |
| **E2 throughput envelope** | Product split made explicit: **ARGUS-Analyst** (full panel, seconds/image, the MVP) and **ARGUS-Screen** (post-MVP: Tier 0/1 + C + F-lite at <300 ms, escalating to full panel on suspicion) — same architecture, two operating points; 32 GB sizing validated for Analyst concurrency=4 |
| **E3 maintenance economics** | The quarterly cycle gets a named bill: ~2 person-weeks + <$500 compute per quarter, documented as COGS, with the in-house generation pipeline (SDXL/FLUX, license-checked) as the fake-supply mechanism. If unfunded, the documented failure mode is graceful: perishable modules stale-flagged in verdicts ("spectral calibration 9 months old") |
| **E4 low-traffic drift blindness** | On-prem deployments ship with (opt-in) federated telemetry or, failing that, quarterly calibration-pack subscriptions from the SaaS fleet; verdicts in telemetry-blind mode disclose `drift_status: "unmonitored"` |

## 10.5 Final Architecture Score: **86 / 100**

Scoring the *post-redesign* architecture against the mission: "how authentic is this image, and what evidence supports that conclusion" — under the stated constraints, against a 2026–2028 threat horizon. Every point lost is itemized; no partial credit hidden in rounding.

| Lost | Where | Why the points are gone (and unrecoverable by design alone) |
|---|---|---|
| −4 | **Anti-forensics ceiling (H3)** | Against a resourced adversary forging jointly-consistent acquisition traces, the statistical tiers are defeated; the surviving core (provenance/retrieval/physics) has coverage gaps today. This is a fundamental limit of pixel-evidence — no redesign removes it, only prices it |
| −3 | **Real-class drift (H1)** | Mitigated by corpus versioning, not solved: the real distribution will keep absorbing generative processing, structurally eroding the realness-first doctrine. The ontology work (S1) manages, but cannot dissolve, the blur |
| −2 | **Physics recall at MVP (S3)** | The most future-proof evidence is the least automatable today; until reflection/reflectance automation matures, the "laundering-resistant core" is thinner than the architecture diagram implies |
| −2 | **Conformal validity under adversarial shift (H5)** | Stratification and drift disclosure narrow, but cannot close, the exchangeability gap; the formal guarantee weakens exactly when it is most needed. Honest labeling ≠ solved problem |
| −1 | **Correlated-witness residual (H4)** | Two backbones and a correlation matrix reduce but cannot eliminate shared-representation failure modes within budget |
| −1 | **Maintenance dependency (E3)** | The architecture is honest about being a *process*, but its safety claims still depend on an organization executing quarterly forever; that dependency is a real fragility, fairly priced at a point |
| −1 | **Evaluation residual risk (S2/S4)** | Held-out platforms and caption-matched fakes are strong fixes, yet benchmark-vs-wild gaps in this field have humbled every prior system; epistemic humility costs a point |

**Why 86 and not lower:** the failure modes that destroy current production systems — silent confidence on laundered inputs, generator overfitting, unexplainable scores, absent abstention — are addressed *structurally*, not by tuning: triage-conditioned reliability, realness-first perishability management, conformal abstention, and checkable-claim verdicts are mechanisms, not aspirations, and each was stress-tested above without collapsing. **Why not higher:** the remaining losses are concentrated where *no* image-side architecture can fully win — adversaries who forge consistency, a real class that is itself becoming synthetic, and guarantees that bend under the distribution shifts that matter most. ARGUS's defining virtue is that it knows and *says* this: the worst case is designed to be honest ignorance rather than confident error. An architecture that earns 100 in this domain is one that has misunderstood the domain.
