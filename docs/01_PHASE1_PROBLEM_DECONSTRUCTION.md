# PHASE 1 — Problem Deconstruction

> Goal: understand *precisely* why the current generation of deepfake / AI-image detectors fails in production, derive the fundamental (not incidental) limits of the classifier paradigm, and identify which evidence types will still matter in five years.

---

## 1.1 Why Existing Detectors Fail — A Causal Analysis

The seven failure modes listed in the brief are symptoms. Below, each is traced to its root cause, because a design that treats symptoms reproduces the disease.

### F1. Single-classifier monoculture
**Symptom:** one CNN/ViT, one logit, one threshold.
**Root cause:** the field framed authenticity as *binary classification*, importing the i.i.d. assumption from ImageNet-style ML. But the deployment distribution is **adversarial and non-stationary**: new generators appear monthly, and adversaries actively optimize against published detectors. A single decision boundary is a single point of failure — once a generator (or an adversarial perturbation) crosses it, the system fails *silently and confidently*. There is no second opinion, no consistency check, no mechanism to notice that the input is unlike anything seen in training.

### F2. Overfitting to known generators
**Symptom:** 99% AUC on StyleGAN, coin-flip on a new diffusion model.
**Root cause:** generator-specific artifacts dominate the training signal. GANs leave characteristic checkerboard/upsampling spectra; early diffusion models leave their own spectral signatures. A discriminative model latches onto the *easiest separating feature* — which is almost always a generator fingerprint, not a property of "fakeness" in general, because **"fakeness in general" does not exist as a stable visual property**. The label "fake" is a statement about *process*, not appearance; classifiers can only see appearance. This is the deepest problem in the field and is unfixable by adding more fakes to training — you only enlarge the closed set, you never close the open set.

### F3. Benchmark success, production failure
**Root causes (three compounding ones):**
- **Dataset confounds.** Benchmark real/fake splits differ in ways unrelated to authenticity. Documented example: in widely used benchmarks (e.g., GenImage), real images are JPEG-compressed while many fakes are pristine PNGs — detectors learn to detect *JPEG vs PNG*, scoring brilliantly in-benchmark and failing the moment a fake is JPEG-compressed. (This confound is dissected further in Phase 10.)
- **Selection bias.** Benchmarks oversample faces, celebrity imagery, and a handful of generators at default settings. Production sees memes, screenshots, document photos, CCTV crops, and generators at every checkpoint/sampler/CFG setting.
- **Goodhart's law.** Once a benchmark becomes the target, methods overfit to it via hyperparameter selection even without training on it.

### F4. Collapse under social-media laundering
**Symptom:** detector accuracy halves after one Telegram/WhatsApp round-trip.
**Root cause:** ~80% of classical forensic evidence is **high-frequency**: sensor noise, demosaicing correlations, generator upsampling artifacts, fine residual statistics. JPEG quantization, downscaling, and re-screenshotting are *precisely* high-frequency erasers. A 2025 benchmark of 15 SOTA forgery-localization methods on social-media-processed images confirmed near-uniform collapse. Crucially, most systems **do not even know they are blind** — they emit the same confident scores on a 5×-recompressed thumbnail as on a pristine RAW. The information is gone, but the confidence is not. *This single observation motivates ARGUS's entire Tier-1 triage design.*

### F5. Poor explainability
**Root cause:** post-hoc saliency (Grad-CAM etc.) on a black-box classifier explains *where the network looked*, not *why the image is fake*. There is no causal, checkable claim a journalist or court can verify. Explainability cannot be bolted on; it must be an architectural property — which requires the evidence to be *named and heterogeneous* from the start.

### F6. No uncertainty quantification
**Root cause:** softmax confidence is not probability of correctness; deep nets are systematically overconfident, *most severely on out-of-distribution inputs* — which is exactly where authenticity systems operate. Without calibrated uncertainty and an abstention mechanism, every output is an unbounded liability.

### F7. No adaptation to evidence quality
**Root cause:** architectures process a 64×64 recompressed thumbnail and a 45 MP RAW through the same pipeline with the same implicit weighting. No notion of "this image can no longer carry evidence type X." Evidence quality is a *per-image latent variable* and almost no deployed system estimates it.

---

## 1.2 Fundamental Limits of Classifier-Based Detection

These are not engineering shortcomings — they are structural properties of the problem.

**L1. The open-set asymmetry.**
Real images form a (slowly drifting) distribution shaped by physics, optics, and camera pipelines. Fake images form the *complement of nothing*: any future generator defines a new fake distribution. A discriminative boundary `p(fake|x)` is only meaningful for fakes *inside the training hull*. Formally: the classifier minimizes risk under `P_train(fake)`, but deployment risk is under `P_future(fake)`, and no bound connects them without assumptions that the adversary deliberately violates.

**L2. The moving-target / red-queen dynamic.**
Generator developers use detectors as discriminators (literally, in GANs; implicitly, via preference tuning and artifact-removal post-processing). Every published detector is a gradient signal for its own obsolescence. A classifier-centric defense is structurally one step behind; the half-life of a fake-trained detector is measured in months.

**L3. Process labels vs. appearance features.**
"Authentic" is a claim about *causal history* (photons → sensor → file), not about pixels. Two pixel-identical images can have different authenticity (a photo of a screen showing a real photo; an AI image of a real scene). No function of pixels alone can recover causal history in general — pixels can only provide *evidence about* it. This reframes the task: **estimation of process from evidence, under uncertainty** — i.e., forensics, not classification.

**L4. The convergence limit.**
As generators improve, `P_fake → P_real` in distribution. Any detector relying on distributional gaps has vanishing signal in the limit. The only evidence classes that survive convergence are those that generators are *not optimized to satisfy*: global physical consistency (ray-traced shadow geometry), cryptographic provenance, and external context (where else does this image exist?). Phase 3 scores all evidence types against this limit.

**L5. Base-rate dominance.**
In production feeds, fakes may be 0.1–5% of traffic. A 95%-accurate detector at a 1% base rate produces ~84% false positives among its alarms (precision ≈ 16%). Classifier thresholds tuned on balanced benchmarks are operationally meaningless.

---

## 1.3 Why Accuracy Metrics Are Insufficient

| What accuracy/AUC hides | Why it matters in production | What ARGUS measures instead |
|---|---|---|
| **Base rates** | Balanced-test accuracy says nothing about precision at 1% prevalence | Precision/recall at deployment-realistic priors; cost-weighted risk |
| **Calibration** | A "0.93" that's wrong a third of the time is worse than useless — it transfers false confidence to humans | ECE, Brier score, reliability diagrams |
| **Coverage / abstention quality** | The decision *to decide* is itself a decision; accuracy ignores it | Risk–coverage curves; conformal coverage validity |
| **Generalization structure** | A single number averages over generators, hiding total failure on the newest one | Leave-one-generator-out matrices (per-generator, never averaged away) |
| **Degradation behavior** | Average over pristine + laundered hides cliff-edge collapse | Laundering-ladder curves: metric vs. degradation severity |
| **Asymmetric costs** | Calling a war-crime photo "fake" ≠ calling a fake "real"; costs differ by 10–1000× and by use case | Per-error-type cost reporting; user-settable operating points |
| **Explanation correctness** | A right answer for a wrong reason fails under cross-examination | Evidence-attribution audits (does the cited artifact actually exist?) |

**Conclusion:** ARGUS's primary metric is **calibrated risk–coverage under leave-one-generator-out × laundering-ladder evaluation** (full protocol in Phase 8). Accuracy is reported but never optimized as the sole target.

---

## 1.4 Evidence Types Most Likely to Survive Future Generators

Ranked by expected longevity (full 22-source scoring in Phase 3):

**Tier S — survives generator convergence (limit L4):**
1. **Cryptographic provenance (C2PA/Content Credentials).** Orthogonal to image content entirely; survives *any* generator improvement. Weakness is adoption and strippability, not generator progress. In 2026, capture-time signing ships in Leica/Sony/Canon bodies — the ecosystem is forming.
2. **Retrieval & external context.** "This image first appeared on X at time T, cropped from Y" is evidence no generator can erase. Generators create images; they cannot create *histories*.
3. **Real-distribution modeling.** Methods trained only on real images (ZED-style one-class, density/reconstruction probes) degrade gracefully as fakes converge — signal weakens but never inverts, and crucially they **cannot overfit to generators they never saw**.

**Tier A — durable for years, degrades slowly:**
4. **Physics & projective geometry** (shadow/light-source consistency, reflection vanishing points, perspective coherence). Generators learn *local texture statistics* superbly but global ray-traced consistency only implicitly and imperfectly; enforcing it would require physical simulation in the loop, which current architectures do not do. Bonus: geometry is **low-frequency → laundering-resistant**.
5. **Semantic/logical consistency** (typography that says nothing, impossible object interactions, mismatched earrings — the DARPA SemaFor target class). Improving steadily but the long tail of world-logic is enormous.
6. **Degradation-history forensics** (JPEG generation counting, double-quantization, resampling traces). Not authenticity evidence per se, but *evidence about the evidence* — and it survives because it measures the laundering itself.

**Tier B — useful today, dying slowly:**
7. **Spectral/frequency artifacts of upsampling** — still strong in 2026 (SpAN's results), but each generator iteration cleans its spectrum further.
8. **Learned noise residuals** (Noiseprint++-class) — durable against *manipulation/splicing*, moderate against whole-image generation, fragile under heavy compression.

**Tier C — already failing or structurally narrow:**
9. **PRNU** — requires per-camera reference sets; destroyed by compression/resize; irrelevant to "is this AI-generated" (only answers "did *this camera* take it").
10. **Generator-specific fingerprints & latent inversion (DIRE-class)** — by construction family-specific; expensive; the canonical example of evidence that a new generator obsoletes overnight.

**Design directive derived:** ARGUS weights its evidence panel toward Tiers S and A, uses Tier B with degradation-conditioned reliability, and includes Tier C only as optional, clearly-labeled plugins.
