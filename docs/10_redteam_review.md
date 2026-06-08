---
title: "ARGUS — Adversarial Red-Team Review"
subtitle: "A Hostile Multi-Institution Panel Review (NeurIPS · CVPR · DARPA · OpenAI · DeepMind · FAIR · MSR)"
author: "Review Panel — Senior Reviewers"
date: "June 2026"
version: "RT-1.0"
---

# ARGUS — ADVERSARIAL RED-TEAM REVIEW

### Mandate: break it. Not improve it.

> **Panel composition.** This review is written as a joint program-committee / red-team panel: a **NeurIPS** statistical-ML reviewer (calibration, conformal theory, fusion), a **CVPR** media-forensics reviewer (forensic signal physics), a **DARPA** program manager (operational deployment, adversaries, chain-of-custody), an **OpenAI / DeepMind / FAIR** trio (future generators 2027–2030, foundation-model dynamics), and a **Microsoft Research** systems reviewer (cost, scale, security). We assume ARGUS is deployed for governments, courts, newsrooms, and enterprises against generators that **do not yet exist**. We assume the adversary has read this exact document.
>
> **Scoring convention for "Impact":** stated as the expected degradation of the metric that matters in the relevant deployment, plus the operational consequence. Numbers are reasoned estimates, explicitly labeled as such; they are directional, not measured. **Cost** is stated as engineering effort (FTE-months), recurring spend, and/or data acquisition.

---

## 0. The one-sentence kill shot

ARGUS rebrands "we cannot reliably detect synthetic images" as "reliability-aware abstention," then leans its open-set survival on **two exogenous crutches it does not control (C2PA coverage and retrieval corpus coverage)** and **one decaying physical prior it overclaims (image-only physics consistency)**. Against a 2027–2030 generator, **all three fail simultaneously on the exact inputs that matter most** — novel, laundered, provenance-stripped images — and the system's honest response is to abstain, which a court, an election war-room, or a fraud-detection pipeline cannot act on. **The architecture is intellectually honest and operationally evasive at the same time.** Below, in detail, why.

---

## 1. SCIENTIFIC & STATISTICAL FLAWS (NeurIPS lens)

### 1.1 The conformal abstention guarantee is void exactly when invoked — and the doc admits it, then ships it anyway
- **Why it fails.** Split/Mondrian conformal prediction guarantees marginal coverage **only under exchangeability** between calibration and test data. The entire reason ARGUS exists is **distribution shift** (unseen 2027–2030 generators + evolving platform laundering). Under shift, the coverage guarantee is not "weakened," it is **mathematically inapplicable**; weighted/shift-adaptive conformal requires a *known or estimable* likelihood ratio between train and test distributions, which for an **unknown future generator is unknowable by construction** (you cannot importance-weight toward a distribution you have never sampled). ARGUS's headline "honest uncertainty" is therefore a guarantee that evaporates precisely in the regime it is sold for.
- **Impact (quantified).** Empirically, conformal coverage degrades roughly in proportion to the total-variation distance between calib and test. For a genuinely novel generator, expect the *advertised* 90% coverage set to deliver **60–75% real coverage** — i.e., the abstention threshold silently mis-fires 1.5–4× more than claimed. In a court setting, that is the difference between admissible and inadmissible.
- **Mitigation.** Stop claiming distribution-free coverage. Replace with (a) **conservative venn-abers / cross-conformal** for *seen* generators only, and (b) for OOD, a **purely heuristic, monotone, externally-audited abstention rule** advertised as heuristic. Add continuous **coverage monitoring in production** with automatic threshold re-tightening when realized coverage drifts. This is a *labeling/honesty* fix, not a technical rescue — there is no technical rescue.
- **Cost.** 2–3 FTE-months for monitoring + recalibration tooling; ongoing labeling spend for drift detection (~$5–15k/quarter for human-verified production samples).

### 1.2 Reliability is treated as a deterministic scalar; it is a random variable with heavy-tailed error
- **Why it fails.** The degradation transfer function maps a quality vector → a single `reliability_score`. But the quality vector is itself *estimated* (JPEG-Q estimation, screenshot heuristics, double-compression detection all have error), and the transfer function is fit on *simulated* laundering. Compounding two miscalibrated stages multiplicatively (the doc uses `min/product`) **propagates and amplifies error**. A reliability of "0.31" has no error bars, yet it multiplies every LLR. **Garbage reliability × good LLR = garbage contribution, silently.**
- **Impact.** Reliability misestimation of ±0.2 (entirely plausible from sim-to-real gap) swings the fused log-odds enough to flip ~**8–15% of borderline verdicts** (the 0.3–0.7 authenticity band — i.e., the only verdicts anyone cares about). The system is *most* fragile exactly where decisions are *hardest*.
- **Mitigation.** Model reliability as a **distribution** (Beta posterior per module), propagate via Monte-Carlo or a moment-matched fusion, and widen Trust by reliability variance. Calibrate transfer functions on **real platform round-trips** (upload→download through actual platforms), not simulations, refreshed quarterly because platform encoders change.
- **Cost.** 3–4 FTE-months for probabilistic fusion refactor; a continuous laundering-harvest pipeline (~2 FTE-months to build, ~$2k/mo proxies/accounts).

### 1.3 Naïve-Bayes core + LightGBM correction head is a contradiction that re-imports the arms race
- **Why it fails.** The Bayesian core assumes conditional independence (false: frequency, residual, diffusion, texture modules are heavily correlated — they read overlapping pixel statistics). The "fix" is a learned head. But **the head is trained on labeled real/fake data from *known* generators** — it is, definitionally, the closed-world classifier ARGUS spent Phase 1 condemning, now smuggled into the fusion layer. On a 2027–2030 generator the head extrapolates, and the doc's own safeguard ("trust the core, raise uncertainty when head disagrees") means the head **adds value only on seen generators and is disabled on unseen ones** — i.e., it does nothing in the regime that matters and adds overfitting risk in the regime that doesn't.
- **Impact.** The head likely contributes **+5–12% AUROC on seen generators and ≈0 (or negative) on LOGO**. Net open-set benefit ≈ 0, with added maintenance, drift, and a new attack surface. You paid complexity for nothing where it counts.
- **Mitigation.** Drop the learned head from the *open-set* path entirely. Replace independence-correction with an explicit **copula / learned covariance among the correlated Tier-C modules only**, estimated *unsupervised* on unlabeled images (no real/fake labels) so it does not encode generator identity. Keep the head, if at all, as a *seen-generator booster* clearly fenced off and OOD-gated to zero.
- **Cost.** 2 FTE-months (copula covariance estimation + gating logic).

### 1.4 "Calibrated probability" is meaningless without a defined, stable base rate — and the base rate is adversarially controlled
- **Why it fails.** A calibrated posterior `P(authentic)` requires a prior `P(authentic)` (the `context_prior` in the pseudocode). In deployment, the base rate of fakes is **set by the adversary** and is non-stationary (a coordinated disinformation campaign can shift a feed from 1% fake to 40% fake overnight). A posterior calibrated to last month's prior is **systematically wrong** during exactly the events (elections, conflicts, market manipulation) the system is bought for.
- **Impact.** A 1%→30% prior shift moves a "calibrated" 0.5-threshold's precision/recall operating point by **tens of points**; verdicts that were calibrated become 10–25% miscalibrated (ECE) during crises. The tool fails on its highest-stakes day.
- **Mitigation.** Output **likelihood ratios, not posteriors**, as the primary product (LR is prior-independent), and let the deployer supply the prior explicitly and per-context. Provide a prior-sensitivity band on every verdict. Detect prior-shift via input-stream monitoring and flag "elevated base-rate uncertainty."
- **Cost.** 1–2 FTE-months (API/contract change + stream monitor).

### 1.5 Selective risk / risk–coverage is gameable and hides the real failure
- **Why it fails.** ARGUS's flagship metric — high accuracy at low coverage — is trivially achieved by **abstaining on everything hard**. A system that answers confidently on the 30% of trivially-easy images (valid C2PA, or exact retrieval hit) and abstains on the 70% that are actually contested will post a *beautiful* risk–coverage curve while being **useless on every genuinely adversarial case**. The metric rewards the failure mode.
- **Impact.** Headline "95% selective accuracy at 30% coverage" can coexist with **~chance performance on the contested 70%**. For a journalist, the 70% *is the job*.
- **Mitigation.** Report **coverage-stratified accuracy on the *hard* subset specifically** (no C2PA, no retrieval hit, laundered), and a mandatory **"forced-decision" accuracy** at 100% coverage. Treat abstention rate on the hard subset as a first-class *failure* metric, not a feature.
- **Cost.** Evaluation-harness work, ~1 FTE-month. Cheap; the issue is honesty, not engineering.

---

## 2. COMPUTER-VISION & FORENSIC FLAWS (CVPR lens)

### 2.1 The "physics consistency is durable" thesis is the load-bearing assumption — and it is already collapsing
- **Why it fails.** The 2027–2030 threat is precisely **physics-aware generation**: diffusion models conditioned on explicit 3-D scene representations, neural radiance / Gaussian-splat priors, differentiable-rendering supervision, and inverse-graphics consistency losses are *already* research-active in 2025–2026. Shadow geometry, vanishing-point consistency, and lighting-direction coherence are **differentiable and therefore trainable as objectives**. Anything you can write as a forensic consistency test, the generator can add as a loss term. ARGUS's *durable backbone* is a **roadmap of objectives for the adversary**.
- **Impact.** Physics-module AUROC against a 2028 physics-aware generator likely falls from ~0.80 (today's naive generators) to **0.55–0.62** — barely above chance — and the doc itself scores 0 of the lost points to this beyond a vague "−4." Since physics is the open-set backbone, its collapse takes the **entire open-set value proposition** with it.
- **Mitigation.** There is no durable mitigation via image-only physics. Pivot the backbone to **inter-image and cross-modal** physics: multi-frame/burst consistency, cross-source corroboration, and **capture-time attestation** (see §3.3). Accept physics as a *decaying* Tier-B, not a Tier-A backbone, and re-score the architecture accordingly.
- **Cost.** Strategic — re-architecture, 6–12 FTE-months, plus a hardware-attestation dependency ARGUS cannot build alone.

### 2.2 VLM-based semantic/physics module is a hallucination engine wearing a forensic badge
- **Why it fails.** Using an open VLM (Qwen2-VL/LLaVA-class) to assert "six fingers" or "inconsistent shadow" produces **confident, fluent, wrong** outputs. The doc's fix ("VLM proposes, geometry verifies") is sound for *geometry* but **un-implementable for semantics**: there is no geometric verifier for "this hand is anatomically impossible" or "this uniform insignia is fabricated." For the semantic class, the VLM *is* the verifier, and it is a closed foundation model with its own unknown training distribution, its own biases, and its own adversarial vulnerabilities (typographic/visual prompt injection inside the image itself).
- **Impact.** Expect a **5–15% false-positive rate** from VLM semantic claims on unusual-but-real images (medical, cultural dress, rare objects, non-Western contexts), each delivered with a *persuasive natural-language rationale* — i.e., **worse than a black box because juries and editors believe the story.** This is a defamation and wrongful-censorship liability.
- **Mitigation.** Demote VLM semantics to **non-scoring advisory only** (never contributes LLR; shown to analysts as "unverified flag, human-check required"). Add **image-borne prompt-injection defense** (the image may contain text crafted to steer the VLM). Require a second independent detector for any semantic claim that affects the score.
- **Cost.** 2–3 FTE-months + ongoing VLM red-teaming; recurring API/compute for the VLM (~$0.002–0.02 per image at scale — see §4).

### 2.3 PRNU / CFA / sensor-noise modules are dead-on-arrival for the real corpus
- **Why it fails.** These require near-native resolution and ≤1 JPEG generation. **The median internet image is none of those.** The doc concedes they self-silence — but that means on ~70–90% of real traffic, **a third of the evidence modules contribute nothing**, and the system runs on a starved evidence set. Worse: PRNU's *positive* value (matching to a known device) requires a reference device almost never available in OSINT/journalism. These modules are benchmark theater.
- **Impact.** On laundered traffic, effective evidence dimensionality drops from ~11 modules to **~3–4** (retrieval, gross physics, semantic-advisory, C2PA-if-present), most of which are *also* compromised (§2.1, §3). The "orthogonal evidence" pitch is **2–3 actually-firing channels** in the wild.
- **Mitigation.** Keep PRNU/CFA only for the high-end provenance lane (forensic labs with original files). For the wild, invest the saved compute in retrieval-corpus growth and attestation.
- **Cost.** Net negative cost (remove from default pipeline); reallocation, ~1 FTE-month.

### 2.4 Retrieval is a coverage lottery and an active attack surface
- **Why it fails.** Retrieval is ARGUS's strongest *claimed* open-set signal, but (a) a **freshly fabricated** image — the entire point of a disinformation op — has **zero prior web presence**, so retrieval abstains on the highest-stakes inputs; (b) the index is **poisonable**: an adversary can pre-seed near-duplicates with fake "early" timestamps on indexable sites to manufacture false provenance, or flood the known-fake DB with real images to induce false positives; (c) **perceptual-hash collisions** can be crafted adversarially (hash-collision attacks on pHash/NeuralHash are published). Retrieval gives a false sense of an oracle while being manipulable by anyone who understands it.
- **Impact.** On novel harmful images: **retrieval contributes 0** (no match) → abstention on the exact case the customer paid to catch. On poisoned indices: **targeted false verdicts** at attacker's choosing. Both are catastrophic for a forensic tool.
- **Mitigation.** Treat retrieval timestamps as **claims requiring corroboration**, not ground truth; require multiple independent sources and cryptographic timestamping (OpenTimestamps / trusted archives) before a retrieval hit is decisive. Use robust, non-invertible embeddings + collision monitoring. Never let a single retrieval hit be dispositive.
- **Cost.** 3–5 FTE-months (provenance-corroboration logic, timestamp anchoring, collision defense); ongoing index-integrity ops.

### 2.5 "Implausible perfection" / too-clean detector false-positives the entire smartphone era
- **Why it fails.** Computational photography (deep fusion, multi-frame HDR, learned denoise/super-res) makes **authentic flagship-phone images statistically "too clean."** Billions of real images now look like the synthetic signature this module flags. The doc's fix ("device-conditioned model") requires a maintained database of every phone pipeline's noise signature — a moving target across hundreds of devices and yearly firmware changes — and **adversaries can spoof a known-clean device signature trivially** (it's just a target statistic).
- **Impact.** Without per-device modeling: **double-digit false-positive rate on modern phone photos**. With it: a permanent, expensive data-maintenance treadmill, still spoofable. Either way the module is a liability for a tool that flags real citizens' photos as fake.
- **Mitigation.** Demote to weak corroboration only; never score-driving. Pair with EXIF device claims (when present) and treat divergence as the signal, not raw cleanliness.
- **Cost.** Reframe + device-signature corpus, 2–4 FTE-months + ongoing.

---

## 3. SECURITY & ADVERSARIAL FLAWS (DARPA lens)

### 3.1 The whole system is a white-box adversarial target with a published gradient
- **Why it fails.** ARGUS is **open-source by mandate** and **explainable by design** — it tells the attacker *exactly which evidence moved the verdict and by how much* (signed LLR contributions). That is a **free, per-image attack oracle**. An adversary queries ARGUS, reads the contribution bars, and minimally perturbs the image to zero out the firing modules — a textbook **transferable evasion / oracle attack**, made trivial because the explanation hands over the gradient direction in plain language. Explainability and adversarial robustness are in **direct tension**, and ARGUS chose maximal explainability.
- **Impact.** A motivated adversary (state actor, fraud ring) achieves **near-100% evasion** with imperceptible, laundering-surviving perturbations, because they can iterate against the open model and the explanation tells them when they've won. The system is robust only against adversaries who don't try.
- **Mitigation.** **Rate-limit and log** verdict queries; do **not** expose per-module contributions in the public/API tier (analyst-only, audited). Add **randomized smoothing** / input-transformation ensembles to break gradient transfer. Maintain a **private model variant** for high-stakes use so the open model is not the deployed model. Accept a robustness/transparency trade-off explicitly.
- **Cost.** 4–6 FTE-months (smoothing ensemble, tiered explanation access, query monitoring); ongoing adversarial red-team (~2 FTE permanent).

### 3.2 C2PA is forgeable-in-practice and strips-in-practice — and ARGUS over-trusts a valid manifest
- **Why it fails.** ARGUS treats a valid signed manifest as "near-dispositive positive evidence." But (a) **manifests are routinely stripped** by platforms → near-zero coverage in the wild; (b) the signing **trust root is the weak link**: a compromised/rogue/coerced signer, a leaked device key, or a "conformant" capture app that will sign *anything* fed to its sensor (point a phone at a screen showing a deepfake → genuine signature over fake content) **defeats the cryptography without breaking it.** The analog-hole (re-photographing a synthetic image with a real C2PA camera) produces a **cryptographically valid manifest over fully synthetic content.** Over-trusting valid C2PA is a direct exploit path.
- **Impact.** An adversary with one conformant camera and a 4K monitor mints **unlimited "authentic," cryptographically-signed deepfakes.** For a court that weights C2PA heavily, this is a **manufactured-evidence pipeline.**
- **Mitigation.** Treat valid C2PA as **"provenance asserted," not "authentic"**; cross-check signed content against analog-hole tells (moiré, screen-reflection, bezel, refresh-banding) and require the manifest's *capture hardware attestation* + liveness, not just signature validity. Maintain a **signer-revocation / trust-list** and weight by signer reputation.
- **Cost.** 3–4 FTE-months (analog-hole detector, trust-list infra); dependency on C2PA ecosystem maturity ARGUS doesn't control.

### 3.3 No defense against the strongest 2027–2030 attack: a perfect generator + forged-but-coherent capture pipeline
- **Why it fails.** The doc concedes this case → "abstain." But a forensic system whose answer to the **most important future threat** is "I don't know" is, for the buyer, **equivalent to having no system** on that threat. Worse, the *adversary* knows ARGUS abstains here, so they **deliberately operate in the abstention zone** — they engineer every harmful image to be novel + laundered + provenance-free, guaranteeing abstention. ARGUS's honesty becomes a **published safe-harbor for attackers.**
- **Impact.** The deployment's *actual* adversarial inputs are, by attacker design, **100% in the abstain region**. Effective detection rate on intentional attacks → **near 0**. The system catches careless fakes and misses every competent one.
- **Mitigation.** The only real answers are **outside image-only forensics**: capture-time hardware attestation (secure enclave signing at the sensor), broad C2PA adoption, and network/behavioral/provenance signals (who posted, when, propagation graph). ARGUS must be positioned as **one layer in a multi-modal pipeline**, never a standalone forensic authority. This is a **product-scope** admission, not a fix.
- **Cost.** Strategic; depends on ecosystem (years, industry-wide).

### 3.4 Chain-of-custody, reproducibility, and Daubert admissibility are unaddressed
- **Why it fails.** For courts (a stated target), a forensic tool must meet **Daubert/Frye**: known error rate, peer review, standards, reproducibility. ARGUS's verdict depends on **a live retrieval corpus, a VLM, and calibration that drift over time** — meaning the **same image yields different verdicts on different dates**, and the error rate is *undefined on unseen generators by design*. That is **inadmissible** and arguably worse — an opposing expert dismantles it in cross-examination by changing one query date.
- **Impact.** Legal deployments are **non-viable** without major changes; a verdict introduced as evidence is impeachable, risking case loss and institutional liability.
- **Mitigation.** Offer a **frozen, versioned, offline "forensic mode"** (pinned models, snapshotted corpus, no live calls, published error rates on a fixed benchmark, full audit log + deterministic re-run). Separate this entirely from the live "triage mode."
- **Cost.** 4–6 FTE-months (deterministic pipeline, versioning, snapshotting, documentation for legal standards).

---

## 4. COMPUTATIONAL & SYSTEMS FLAWS (Microsoft Research lens)

### 4.1 Per-image cost is 1–3 orders of magnitude above a single classifier — the cascade only partly saves it
- **Why it fails.** A full ARGUS pass runs CLIP embedding + FAISS query + a VLM (the expensive part) + classical DSP. The VLM dominates: a 2–7B VLM is **~100–1000 ms/image on GPU**, or ~$0.002–0.02/image via API. The cascade defers this to "ambiguous/high-stakes" images, but **adversaries deliberately produce ambiguous images** (§3.3), so the *adversarial* workload **forces the expensive path on exactly the inputs that matter** — the cascade's cost savings assume a benign distribution the threat model denies.
- **Impact.** At platform scale (1B images/day), even 10% escalation = **100M VLM calls/day** = **$200k–$2M/day** or a GPU fleet in the **thousands**. For a startup, unaffordable; for a platform, a hard cost ceiling that caps coverage. Adversaries can also **cost-attack** by mass-uploading ambiguous images to exhaust the expensive lane (economic DoS).
- **Mitigation.** Hard per-tenant budget caps + a cheap distilled "ambiguity gate" so only truly high-value images hit the VLM; replace the VLM with **specialized small geometric/semantic models** where possible; rate-limit + cost-DoS detection.
- **Cost.** 3–4 FTE-months for distillation + budget governor; the recurring spend is the real cost.

### 4.2 Latency is incompatible with real-time moderation
- **Why it fails.** Full-pass latency (retrieval + VLM + DSP + fusion) is **0.5–3 s/image**, single-threaded. Platform moderation needs **<100 ms** at ingestion. ARGUS cannot run inline; it runs as a slow async second-pass, by which time disinformation has **already gone viral** (median time-to-virality for breaking-news imagery is minutes).
- **Impact.** For prevention use cases, ARGUS is **too slow to matter** — it does post-hoc forensics, not interdiction. Mismatch with the "stop it before it spreads" value prop.
- **Mitigation.** Two-tier: a <50 ms classical+hash+C2PA gate inline; full ARGUS async for flagged content only. Set product expectations to **forensic review, not real-time filtering.**
- **Cost.** Architecture work, 2–3 FTE-months.

### 4.3 Operational complexity = 8+ modules × per-module calibration × drift monitoring × corpus ops = an MLOps swamp
- **Why it fails.** Each module needs independent calibration that **drifts** (platform encoders change, generators change, devices change). The retrieval corpus needs continuous ingestion + integrity defense. The VLM and CLIP need version management. This is **not a 3-month MVP's steady state** — it's a permanent team. The "limited budget / commodity hardware" constraint and this operational reality are **mutually contradictory.**
- **Impact.** Realistic sustaining cost: **3–6 FTE permanent** + infra, before any growth. Many "viable MVP" claims die at month 6 of maintenance, not month 3 of building.
- **Mitigation.** Aggressively cut to a **maintainable core** (C2PA + metadata + JPEG/ELA + retrieval + one calibrated fusion) and add modules only when each earns its maintenance cost. Automate calibration via continuous self-supervised pipelines.
- **Cost.** Honest budgeting; the mitigation is *scope reduction*, which weakens the open-set story (§2.3 trade-off).

---

## 5. DATASET & EVALUATION FLAWS (FAIR lens)

### 5.1 LOGO is a weak proxy and the headline number will overstate real open-set performance
- **Why it fails.** Leave-One-Generator-Out holds out *known* generators from the *same era and paradigm*. It cannot simulate a **2028 architecture that doesn't exist yet** (new sampler, new representation, new training objective). LOGO tests interpolation across *contemporary* generators and is reported as if it measures extrapolation to *future* ones. **It does not.** Every detector paper that posted strong LOGO numbers still collapsed on the next year's model.
- **Impact.** LOGO AUROC likely **overstates true future-generator AUROC by 0.10–0.25**. The architecture's open-set claims rest on a metric that has **historically failed to predict** exactly what it's used to predict here.
- **Mitigation.** Treat each newly released generator as a **prospective, pre-registered test** (lock the model, then evaluate on the new generator's outputs *before* any adaptation). Report a **time-series of degradation**, not a single LOGO number. Publish the falsification, not just the success.
- **Cost.** Ongoing eval discipline, ~1 FTE continuous.

### 5.2 Calibration and semantic data are demographically and geographically biased — a civil-rights liability
- **Why it fails.** Open authentic corpora (FFHQ, RAISE, Flickr-CC) skew Western, light-skinned, high-end-camera, English-text, well-lit. Reliability transfer functions, "implausible perfection," and VLM semantics will be **systematically less accurate on under-represented populations, low-end devices, and non-Latin scripts.** For a tool that can **brand a real person's photo as fabricated**, biased error rates are not a footnote — they are a **disparate-impact and defamation exposure** that a government or court deployment will be sued over.
- **Impact.** Plausible **2–4× higher false-positive rate** on under-represented subgroups/devices. In a journalism or asylum/immigration context (real DARPA-adjacent use), this directly harms vulnerable people and is institutionally disqualifying.
- **Mitigation.** Mandatory **per-subgroup error reporting** as a release gate; targeted data acquisition across devices/regions/scripts; **lower Trust (abstain) rather than guess** on under-represented inputs, with explicit disclosure. Independent bias audit before any high-stakes deployment.
- **Cost.** Data acquisition $50–250k; 4–6 FTE-months for subgroup eval + audit; recurring.

### 5.3 No defense against training-set / benchmark contamination by future generators
- **Why it fails.** By 2027–2030, generator outputs **saturate the public web**, which is the source of both authentic corpora and retrieval indices. ARGUS's "authentic" training data and its retrieval corpus will be **silently contaminated with synthetic images mislabeled as real**, poisoning calibration and creating false retrieval "authentic" matches. This is **unavoidable with web-scraped data** and gets worse every year.
- **Impact.** Calibration drifts toward "synthetic looks authentic"; retrieval corroborates fakes. Estimated **steady accuracy bleed of a few points per year**, compounding, with no clean recovery from public data.
- **Mitigation.** Anchor "authentic" ground truth to **provenance-verified or pre-2022 (pre-diffusion-saturation) captures** and hardware-attested sources only; never trust web-"real" labels. Continuous contamination auditing.
- **Cost.** Data-curation pipeline, 3–4 FTE-months + ongoing; shrinking pool of trustworthy authentic data is a structural headwind.

---

## 6. PRODUCT & DEPLOYMENT FLAWS (cross-panel)

### 6.1 "Trust score" + abstention is a liability-laundering mechanism, not a product feature
- **Why it fails.** A government/court/newsroom needs a **decision**. "Authentic 0.5, Trust 0.3, abstain" pushes the decision — and the liability — **back onto a non-expert user** who will either over-trust (treat abstain as "probably fake") or ignore it. The tri-axial output is epistemically honest but **operationally unusable** for the stated buyers, and it conveniently shields ARGUS from being wrong by **never committing** on hard cases.
- **Impact.** Low decision-yield: on the contested inputs that justify the purchase, ARGUS frequently returns "can't tell," and users substitute their own bias. Procurement teams will ask "what's your accuracy on the cases we care about?" and the honest answer is bad.
- **Mitigation.** Pair every abstention with a **concrete next-action workflow** (request original file, escalate to human forensic lab, cross-source check) so the product delivers a *process*, not a shrug. Define and publish decision-yield on the hard subset.
- **Cost.** UX + workflow, 2–3 FTE-months.

### 6.2 Adversarial liability of false *positives* (calling real authentic content fake) is under-weighted
- **Why it fails.** The doc focuses on catching fakes; the larger societal/legal risk is **false positives**: branding a genuine journalist's photo, a real victim's evidence, or authentic war footage as "likely manipulated." Authoritarian actors will **weaponize ARGUS to discredit true content** ("even the AI tool says it's fake"). The system's authority becomes a censorship instrument.
- **Impact.** Reputational and human-rights harm; the tool's existence enables a new disinformation tactic (the "liar's dividend" at scale). This is the *opposite* of the mission.
- **Mitigation.** Asymmetric thresholds biased *strongly* against false-positive "fake" verdicts; require corroborated multi-module evidence before any "likely manipulated" label; legal/ethics review board; refuse to output a "fake" verdict at low Trust (abstain instead). Public methodology to resist "the AI said so" misuse.
- **Cost.** Policy + threshold engineering, 2 FTE-months; governance ongoing.

### 6.3 Open-source mandate undermines the only durable moat
- **Why it fails.** The doc's claimed moats are the calibration process and the corpus — but **open-sourcing the stack hands the calibration methodology and module set to competitors and adversaries.** The corpus is the only real moat, and it's the most expensive, slowest-compounding asset, also poisonable (§2.4). There is **no defensible technical moat**; the business is a services/trust play, not a product play.
- **Impact.** Fast-following by better-funded incumbents (platforms with bigger corpora) is trivial; ARGUS-the-startup has weak defensibility.
- **Mitigation.** Keep the **fusion calibration, OOD models, and corpus private**; open-source only the interface/forensic primitives. Compete on **trust, audits, and integrations**, not algorithms.
- **Cost.** Strategic positioning; minimal eng cost.

---

## 7. IMPACT SUMMARY TABLE

| # | Flaw | Domain | Severity | Est. impact on the case that matters | Mitigation cost |
|---|------|--------|:--------:|--------------------------------------|-----------------|
| 1.1 | Conformal void under shift | Stats | **Critical** | Advertised 90%→~60–75% real coverage | 2–3 FTE-mo + ongoing |
| 1.2 | Reliability as scalar, not RV | Stats | High | Flips 8–15% of borderline verdicts | 3–4 FTE-mo |
| 1.3 | Learned head re-imports closed-world | Stats/ML | High | ~0 open-set benefit, added risk | 2 FTE-mo |
| 1.4 | Base rate adversarial/non-stationary | Stats | High | 10–25% ECE during crises | 1–2 FTE-mo |
| 1.5 | Selective-risk metric gameable | Eval | High | Hides ~chance on hard 70% | 1 FTE-mo |
| 2.1 | Physics backbone trainable away | CV | **Critical** | Physics AUROC 0.80→0.55–0.62 by 2028 | 6–12 FTE-mo (strategic) |
| 2.2 | VLM hallucination (semantics) | CV | **Critical** | 5–15% persuasive false positives | 2–3 FTE-mo + red-team |
| 2.3 | PRNU/CFA dead on wild corpus | CV | Medium | Evidence dim 11→~3–4 in the wild | net-negative cost |
| 2.4 | Retrieval = lottery + poisonable | CV/Sec | **Critical** | 0 on novel; targeted false verdicts | 3–5 FTE-mo + ops |
| 2.5 | Too-clean FP on smartphones | CV | High | Double-digit FP on phone photos | 2–4 FTE-mo + ongoing |
| 3.1 | White-box + explanation = attack oracle | Sec | **Critical** | ~100% evasion by motivated adversary | 4–6 FTE-mo + 2 FTE perm |
| 3.2 | C2PA analog-hole / rogue signer | Sec | **Critical** | Unlimited signed deepfakes | 3–4 FTE-mo + ecosystem |
| 3.3 | Adversaries live in the abstain zone | Sec | **Critical** | ~0 detection on competent attacks | strategic / years |
| 3.4 | Not Daubert-admissible | Sec/Legal | **Critical** | Legal deployment non-viable as-is | 4–6 FTE-mo |
| 4.1 | Cost 1–3 OOM over a classifier | Sys | High | $200k–$2M/day at platform scale | 3–4 FTE-mo + spend |
| 4.2 | Latency 0.5–3 s vs <100 ms needed | Sys | High | Too slow to interdict virality | 2–3 FTE-mo |
| 4.3 | MLOps maintenance swamp | Sys | High | 3–6 FTE permanent, breaks budget | scope cut |
| 5.1 | LOGO overstates future perf | Eval | High | AUROC overstated 0.10–0.25 | 1 FTE continuous |
| 5.2 | Demographic/device bias | Data | **Critical** | 2–4× FP on under-represented groups | $50–250k + 4–6 FTE-mo |
| 5.3 | Web-data contamination by 2027+ | Data | High | Few points/yr compounding bleed | 3–4 FTE-mo + ongoing |
| 6.1 | Abstention = unusable decision | Product | High | Low decision-yield on hard cases | 2–3 FTE-mo |
| 6.2 | False-positive / liar's-dividend harm | Product | **Critical** | Weaponizable censorship tool | 2 FTE-mo + governance |
| 6.3 | Open-source kills the moat | Product | Medium | Weak defensibility | strategic |

**Critical flaws: 10. High: 10. Medium: 3.** The ten criticals cluster on the **same failure surface**: novel + laundered + provenance-free + adversarial inputs — i.e., the operational reality the system is sold for.

---

## 8. WHAT ARGUS ACTUALLY IS, STRIPPED OF MARKETING

- **It is excellent** at: organizing forensic evidence, calibration discipline, explainability for analysts, and **not lying about uncertainty** — a genuine and rare virtue.
- **It is good** at: catching *today's careless* fakes, triaging the easy majority cheaply, and producing court-presentable *analyst-assist* reports (in frozen mode).
- **It is weak-to-failing** at: **competent 2027–2030 adversaries**, real-time interdiction, legal admissibility as-is, cost at platform scale, demographic fairness, and adversarial robustness as an open white-box system.
- **Its core intellectual move — abstain when unsure — is simultaneously its greatest integrity and its greatest product liability**, because a knowing adversary will keep every harmful image inside the abstention zone.

---

## 9. FINAL VERDICT

> **Ladder:** Impossible · Research Project · **Viable MVP** · Startup Viable · Industry Competitive · State-of-the-Art

### Verdict by deployment context (because a single label would be dishonest):

| Deployment | Verdict | One-line justification |
|------------|---------|------------------------|
| **As a general-purpose, standalone "is this fake?" authority against 2027–2030 generators** | **Research Project** | The criticals (§2.1, §3.1, §3.3) are unsolved and partly unsolvable image-only. |
| **As an analyst-assist forensic *triage + evidence-organization* tool (frozen mode, human-in-loop)** | **Startup Viable** | Calibration + explainability + abstention genuinely beat single classifiers for expert workflows. |
| **As a court-admissible forensic instrument today** | **Research Project** (→ Viable MVP after §3.4 frozen mode) | Daubert/reproducibility gaps are disqualifying until fixed. |
| **As a real-time platform moderation filter** | **Impossible** (as designed) | Latency + cost + abstain-zone attacks defeat the use case. |
| **As one layer in a multi-modal provenance pipeline (with attestation + network signals)** | **Industry Competitive** | This is the only framing where ARGUS's honest-evidence design is a strength, not a dodge. |

### Consolidated single verdict: **VIABLE MVP — trending Startup Viable, capped below State-of-the-Art.**

**Justification.** ARGUS is a **buildable, honest, expert-assist forensic system** that is meaningfully better-engineered than the single-classifier field it critiques. But it is **not** — and given image-only constraints, **cannot be** — a standalone state-of-the-art detector against the generators it is explicitly designed for. Its ten critical flaws all converge on the adversarial open-set case, and its signature feature (abstention) is exploitable as a published safe-harbor. **It earns "Viable MVP" for the analyst-assist / frozen-forensic / multi-layer-pipeline framing, and is correctly rated no higher because its open-set value proposition rests on exogenous crutches (C2PA, retrieval) and a decaying prior (physics) that competent 2027–2030 adversaries will defeat by construction.**

**The brutal truth:** ARGUS does not solve image authenticity. **Nothing image-only will.** ARGUS's real contribution is admitting that — and building the most honest scaffolding around the admission. That is worth funding as a *layer* and a *research program*. It is not worth deploying as an *oracle*, and any government, court, or platform that treats it as one will be embarrassed by a 19-year-old with a 2028 model and a 4K monitor.
