# PHASE 6 — Unknown Generator Resistance

> *The decisive test. A generator nobody has seen — new architecture, new sampler, post-hoc noise, learned super-resolution — appears tomorrow. Most of the field's accuracy evaporates. What in ARGUS still works, and why?*

## 6.1 The threat model

Assume an adversary's generator that: (a) was never in any training set; (b) produces no known spectral fingerprint; (c) optionally adds realistic sensor-like noise and a fake but internally plausible imaging pipeline; (d) outputs are laundered through social media. This is the *worst realistic case*, and it is the case incumbents are structurally guaranteed to fail.

## 6.2 Which evidence remains useful (and why it must)

The durability ranking from §1.4 is precisely the survival ranking here:

**Survives — because it does not depend on the generator's mistakes:**
1. **Retrieval / provenance-by-corroboration.** The single most powerful open-set signal. If the image (or its source elements) exists earlier on the indexed web, authenticity/synthesis is established *externally*. A new generator cannot rewrite history or pre-plant authentic provenance. **Generator-architecture-independent by construction.**
2. **C2PA / provenance.** Cryptographic. A new generator does not get the capture device's private key. *Presence* of a valid capture manifest remains strong positive evidence; a *missing or AI-declaring* manifest is consistent with synthesis. Unaffected by generator novelty.
3. **Physics consistency (lighting, shadow, perspective, reflection).** Governed by optics and 3-D geometry, not by the generator's learned distribution. A new generator that is better at *texture* is not automatically better at *globally consistent illumination and projective geometry* — these lag, badly and persistently. **Durable.**
4. **Semantic / world-knowledge consistency.** Anatomy, counts, causal plausibility. Improves with generators but remains among the last failures. **Durable-ish.**
5. **Internal consistency / contradiction detection.** ARGUS does not need to recognize the *specific* generator; it needs to detect that the image's evidence is *mutually inconsistent* (e.g., "sensor noise statistically perfect AND lighting impossible"). Inconsistency detection is generator-agnostic.
6. **"Too-clean" / implausible-perfection signal.** Real captures carry irreducible sensor noise, optical aberration, and minor inconsistencies. An image that is *statistically too perfect* for a real capture is suspicious *regardless of which generator made it.*

**Becomes obsolete — because it depends on a *specific* generator's fingerprint:**
- Frequency azimuthal-peak detectors (new up-samplers move/erase the peaks).
- Residual/SRM CNN "fakeness" probes (the canonical overfit feature).
- Diffusion-artifact detectors tuned to a specific noise schedule.
- Latent-inversion against *known* model families (a new family inverts poorly → ambiguous).
- Any learned classifier trained on a closed generator set.

**ARGUS's structural advantage:** these obsolete modules **silence themselves automatically** via the OOD reliability term (§6.4), so they do not *poison* the verdict — they simply drop to reliability≈0 and contribute LLR≈0. Incumbents have no such mechanism; their single classifier *is* the obsolete module, with nothing to fall back on.

## 6.3 Design mechanisms that keep working

1. **Provenance-first, fingerprint-last weighting.** The fusion prior structurally favors generator-agnostic evidence. On an unknown generator, the system naturally leans on retrieval + physics + provenance.
2. **Per-module OOD gating (the keystone).** Every *learned* module ships with a model of its own training manifold and discounts itself when the input is off-manifold (details §6.4). This converts "I've never seen this" from a silent failure into an explicit, calibrated abstention.
3. **Anomaly-as-evidence, not classification.** Physics/semantic modules are framed as **one-class / consistency** detectors ("is this *internally consistent* with a real capture?") rather than two-class ("is this generator-X?"). One-class framing generalizes to unseen generators; two-class does not.
4. **Implausible-perfection detector.** A dedicated module that flags statistically-too-clean images — a generator-agnostic tell that *strengthens* as generators get better (better generators are *cleaner*, tripping this harder).
5. **Conformal abstention.** When all reliable evidence is exhausted and ambiguous, ARGUS returns *"insufficient evidence — abstain"* with low Trust, instead of a confident wrong answer. **Abstention is a feature**: a system that knows what it doesn't know is the only honest open-set system.
6. **Contradiction amplification.** A novel generator that fakes one channel well (e.g., adds realistic noise) but not others (impossible shadows) trips the contradiction graph → Risk rises even though no single module "recognizes" the generator.

## 6.4 How to estimate uncertainty against the unknown

Uncertainty is **decomposed** into two kinds, because they demand different responses:

- **Aleatoric (data) uncertainty:** the image is intrinsically ambiguous/degraded. → reflected in low reliability, low Trust. Response: gather more evidence or abstain.
- **Epistemic (model/knowledge) uncertainty:** the input is outside what the system has learned. → the dangerous, novel-generator kind. Response: discount learned modules, lean on physics/provenance/retrieval, possibly abstain.

**Concrete OOD/epistemic estimators (open-source, cheap):**
- **Feature-space density / energy score** per learned module (e.g., Mahalanobis distance to training-feature mean, or a normalizing-flow / energy score). High distance ⇒ OOD ⇒ reliability discount.
- **kNN distance to training features** (deep-kNN). Simple, strong, non-parametric.
- **Deep ensembles / MC-dropout disagreement** for learned modules — variance across the ensemble is an epistemic proxy.
- **Conformal prediction** over the fused output — yields a *distribution-free* coverage guarantee and a principled abstention rule (non-singleton prediction set ⇒ abstain).
- **Evidence-coverage entropy** — how many *independent* high-reliability channels actually fired. One channel firing = high epistemic uncertainty regardless of its confidence.
- **Core-vs-head disagreement** — when the Bayesian core and the LightGBM head disagree, the input likely lies where the head is extrapolating ⇒ raise epistemic uncertainty.

**Resulting behavior on a brand-new generator:**
> Fingerprint modules go OOD → silence themselves. Retrieval finds no match (novel image) → abstains. Physics finds impossible shadows + implausibly clean noise → fires with high reliability. Contradiction graph lights up. **Output: Authenticity ≈ 0.2, Trust ≈ 0.5, Risk ≈ HIGH, explanation: "lighting geometry inconsistent and sensor statistics implausibly clean; no known-generator fingerprint required for this conclusion."** ARGUS catches a generator it has never seen — using physics, not fingerprints. *That is the whole thesis, demonstrated.*

And the honest failure case:
> A *perfect* future generator with globally consistent physics, realistic noise, and a forged-but-coherent pipeline, on a novel laundered image with no web provenance → ARGUS correctly returns **low Trust + abstain** ("insufficient reliable evidence"), rather than a confident wrong label. Being *honestly uncertain* on the genuinely undetectable is the correct, and only defensible, behavior.

\newpage
