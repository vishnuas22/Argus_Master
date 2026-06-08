---
title: "ARGUS — A Reliability-Aware Authenticity Assessment Platform"
subtitle: "First-Principles Architecture for Image Authenticity in the Post-Deepfake Era"
author: "Principal AI Research Scientist · Digital Forensics · Computer Vision · Security Architecture · Startup CTO"
date: "June 2026"
version: "1.0"
---

# ARGUS

### *Authenticity Reasoning via Generalized Uncertainty-aware Synthesis*

> **Thesis.** The deepfake-detection field is losing an arms race because it framed the wrong question. ARGUS does not ask *"is this a deepfake?"* — a binary that collapses the moment a new generator appears. ARGUS asks *"how authentic is this image, with what reliability, and what evidence supports that conclusion?"* This reframing — from **classification** to **evidence-weighted forensic reasoning under uncertainty** — is the central contribution of this document.

---

## Abstract

Production deepfake detectors fail predictably: they are single classifiers trained on a finite set of known generators, they overfit to generator-specific fingerprints, they evaporate under social-media laundering (recompression, screenshotting, resizing), and they emit a single number with no calibrated uncertainty and no human-auditable rationale. As generative models improve, every classifier-centric system inherits a structurally declining accuracy curve.

This document designs ARGUS from first principles as an alternative. ARGUS is a **reliability-aware evidence fusion engine**. Twenty-plus orthogonal forensic modules each emit a structured tuple `{evidence_score, reliability_score, confidence_score}`. A **self-aware reliability layer** down-weights evidence that the image's own degradation has destroyed (you cannot read sensor noise from a 480p screenshot — and ARGUS *knows* it cannot). A **calibrated probabilistic fusion core** (Bayesian belief combination with a learned correction head, plus an out-of-distribution gate) integrates evidence into three orthogonal outputs — **Authenticity**, **Trust**, and **Risk** — and produces a ranked, contradiction-aware, human-readable explanation.

The design is explicitly engineered for the case that matters most: **the generator was never seen during training.** ARGUS leans on evidence classes that are physically or statistically *generator-agnostic* (camera-pipeline consistency, physics/lighting consistency, provenance/C2PA, retrieval-based provenance) rather than generator-specific fingerprints that are obsolete on arrival. We present a complete evidence taxonomy with robustness/laundering/explainability/cost scoring, a 3-month MVP buildable on commodity hardware with open-source models only, a research-contribution analysis, a hostile peer review that attacks the design, and a redesign that survives the attack. Final self-assessed architecture score: **84/100**, with every lost point justified.

---

## How to read this document

| Phase | Question answered | Page focus |
|------:|-------------------|------------|
| 1 | Why do existing detectors fail, fundamentally? | Problem deconstruction |
| 2 | What is the strongest possible architecture from scratch? | First-principles design |
| 3 | What evidence exists, and how good is each source? | Evidence taxonomy + scorecard |
| 4 | How does each module know when to trust itself? | Reliability-aware module contract |
| 5 | How should evidence be fused optimally? | Fusion strategy comparison |
| 6 | How do we survive an unseen future generator? | OOD / open-set robustness |
| 7 | How do we explain a verdict to a human? | Production XAI |
| 8 | What is the smallest buildable version? | 3-month MVP |
| 9 | What here is genuinely novel? | Research & IP |
| 10 | Where does this design break, and how do we fix it? | Hostile review + redesign |

---

## Glossary of key terms

- **PRNU (Photo-Response Non-Uniformity):** A near-unique multiplicative noise pattern stamped on every photo by physical sensor imperfections; the closest thing to a camera "fingerprint."
- **CFA (Color Filter Array):** The Bayer mosaic over a camera sensor; demosaicing leaves predictable inter-channel correlations that synthetic images rarely reproduce.
- **ELA (Error Level Analysis):** Visualizing JPEG recompression error to reveal regions at inconsistent compression levels.
- **C2PA / Content Credentials:** A cryptographically signed provenance manifest (Coalition for Content Provenance and Authenticity) attached to an asset's history.
- **Azimuthal / radial power spectrum:** The 1-D profile of a 2-D Fourier transform, used to expose periodic grid artifacts from up-sampling layers in GANs/diffusion models.
- **Reliability:** *How much this image lets a module do its job* (a property of the evidence channel given the image's quality), as distinct from confidence (*how sure the module is in its reading*).
- **Open-set / OOD:** The realistic regime where the test-time generator was never in the training distribution.

---

\newpage
