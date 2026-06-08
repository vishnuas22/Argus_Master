# PHASE 1 — Problem Deconstruction

> *Goal: understand precisely why the incumbent paradigm fails, so the new design is not just a better classifier but a different category of system.*

## 1.1 Why existing deepfake detectors fail

The failure is not a tuning problem; it is **structural**. Six independent failure mechanisms compound:

### 1.1.1 The single-classifier monoculture
Most detectors are one CNN/ViT head (XceptionNet, EfficientNet, a CLIP-linear-probe, etc.) trained to separate "real" from "fake." A single decision surface has a single attack surface. Any input that lands on the wrong side of that one boundary is misclassified, and there is no second opinion, no orthogonal corroboration, and no way to know *which* cue drove the decision. Biological and intelligence systems never trust a single sensor; forensic science never convicts on a single trace. Detection inherited the ImageNet reflex — *one model, one softmax* — and it is the original sin.

### 1.1.2 Generator-fingerprint overfitting (the core arms-race trap)
GAN and early diffusion images carry generator-specific spectral fingerprints — periodic peaks in the Fourier spectrum from transposed-convolution / up-sampling stages, characteristic noise residuals, color-channel statistics. Classifiers learn these *shortcuts* because they are the lowest-loss features on the training set. But a fingerprint is a property of an *architecture*, not of *fakeness*. The instant a new generator (new up-sampler, new sampler, learned super-resolution, post-hoc noise injection) is released, the fingerprint moves and the classifier's learned feature is pointing at empty space. This is why detectors that report 99% on FaceForensics++ collapse to near-chance on a generator released six months later.

### 1.1.3 Benchmark-to-production distribution shift
Benchmarks are clean: known generators, controlled resolutions, lossless or single-JPEG storage, frontal faces, lab lighting. Production images are laundered through an adversarial pipeline of platform recompression, chroma subsampling, resizing, watermark overlays, screenshots, and re-uploads. The features that scored 99% in the lab live in exactly the high-frequency bands that laundering destroys. Reported accuracy is therefore measuring performance on a distribution that **does not exist in the wild.**

### 1.1.4 Laundering as an unintentional adversarial attack
You do not need a malicious adversary. Instagram's encoder, a WhatsApp forward, and a phone screenshot together constitute a strong, free, ubiquitous adversarial transform. They strip metadata, normalize quantization tables, re-introduce *new* JPEG grids, blur sensor noise, and resample away spectral peaks. A detector tuned to subtle pixel statistics is reading mostly the *platform's* fingerprint after laundering, not the *generator's*.

### 1.1.5 No uncertainty quantification
A detector that outputs `0.91 fake` gives no way to distinguish (a) *"strong, corroborated evidence of synthesis"* from (b) *"the model is extrapolating wildly on an out-of-distribution image and 0.91 is noise."* Softmax probabilities are **not** calibrated confidences, especially off-distribution, where neural nets are famously, dangerously overconfident. Without calibrated uncertainty, every score is unfalsifiable.

### 1.1.6 No explainability
A bare score cannot be audited, contested in court, or trusted by a journalist or moderator. "The model said 0.91" is not evidence — it is an oracle. When the model is wrong (and it will be, off-distribution), there is no trace to inspect, no contradictory cue surfaced, no chain of reasoning. This is disqualifying for any high-stakes deployment (legal, journalistic, electoral, insurance).

### 1.1.7 Static evidence weighting
Real images arrive with wildly varying evidence quality — a RAW DSLR file vs. a 5th-generation meme screenshot. Incumbent detectors weight their internal features identically regardless of input quality. They will confidently read "sensor noise" from an image that physically has none left. They cannot **adapt the weight of a cue to the quality of the evidence that cue depends on.** This single missing capability is, in our view, the largest practical gap.

## 1.2 Fundamental limitations of classifier-based detection

Beyond engineering, there are limits *in principle*:

1. **Closed-world assumption.** A classifier partitions a fixed feature space learned from a fixed label set. "Fake" is defined *extensionally* (by examples of known generators), not *intensionally* (by what makes something inauthentic). Anything outside the example set is undefined behavior.
2. **Convergence of the two distributions.** As generators improve, the pixel-statistical distribution of synthetic images converges toward that of real images *by construction* — that is literally the generator's training objective. Any discriminator operating purely on the image's own pixel statistics is therefore attacking a target that is being actively, adversarially erased. The Bayes-optimal error rate of pure-pixel detection trends upward over time toward 50%.
3. **Shortcut learning is the default, not the exception.** Given a choice between a robust-but-hard feature (physics/lighting consistency) and a fragile-but-easy feature (a spectral peak), gradient descent takes the easy one. You cannot reliably force generalization through loss design alone.
4. **No decomposition.** A monolithic score cannot tell you *which* aspect is anomalous, so it cannot be partially right. Forensics needs partial, composable conclusions ("metadata absent, lighting consistent, retrieval finds an authentic 2019 original").
5. **Calibration degrades off-distribution exactly when you need it most.** The regime where you most need honest uncertainty (novel generator) is precisely where softmax is least calibrated.

**Conclusion:** A classifier is a legitimate *evidence source*, but it is a catastrophic *system*. ARGUS demotes the deepfake classifier to one module among many — and one of the *least* trusted for unseen generators.

## 1.3 Why accuracy metrics alone are insufficient

- **Accuracy hides the base-rate problem.** In the real world the prior probability of "fake" varies enormously by context (a dating profile vs. a wire-service photo). A fixed-threshold accuracy number is meaningless without the deployment prior; the relevant quantity is a *calibrated posterior*, not a label.
- **Accuracy ignores cost asymmetry.** Falsely flagging a real journalist's photo as fake (censorship, defamation) and missing a fabricated piece of evidence (fraud) have radically different costs. A single accuracy number averages over an asymmetry that the deployer, not the model, must own. ARGUS must output a *calibrated probability + uncertainty* so the deployer can set their own operating point.
- **Accuracy is non-stationary.** Reported on a frozen benchmark, it predicts nothing about next quarter's generator. The only honest metrics are **time-split, leave-one-generator-out (LOGO)** accuracy and, above all, **calibration under distribution shift** (ECE, AURC, selective-risk curves).
- **Accuracy rewards shortcut learning.** Optimizing for benchmark accuracy directly incentivizes latching onto the fragile fingerprints that will not survive. The metric and the goal are misaligned.
- **What to measure instead:** (1) **Leave-one-generator-out AUROC**; (2) **Expected Calibration Error** and reliability diagrams, computed *separately* on laundered subsets; (3) **Selective risk / risk–coverage curves** (accuracy when the system is allowed to abstain on low-reliability inputs — ARGUS's core advantage); (4) **robustness curves** vs. JPEG-Q, downscale factor, and screenshot simulation; (5) **explanation faithfulness** (do the cited cues causally drive the score?).

## 1.4 Evidence likely to survive future generator improvements

The strategic question. Rank evidence by *durability* against a generator whose explicit objective is to look real:

**Tier A — Durable (physics, provenance, and the "absence" signal).** These do **not** depend on a generator's mistakes and so do not decay as generators improve:
- **Provenance / C2PA / content credentials.** Cryptographic, generator-independent. A valid signed capture chain is positive authenticity evidence no generator can forge without the private key. *(Caveat: covers presence, not absence — most real images also lack it.)*
- **Camera-pipeline *presence*.** The *positive* presence of a coherent physical capture pipeline — consistent PRNU across the frame, valid CFA/demosaicing correlations, a single self-consistent JPEG history, plausible optical aberration. Generators must *fabricate* an entire imaging physics chain coherently; today they don't, and doing so is strictly harder than improving perceptual realism.
- **Retrieval / provenance-by-corroboration.** If near-duplicates of this image exist on the indexed web with an earlier timestamp and a credible source, authenticity is established *externally*, independent of pixels. Generators cannot retroactively plant history.
- **Physics consistency** (lighting direction, shadow geometry, perspective, reflections, vanishing points). Governed by optics and geometry, not by the generator's training loss. Generators get *better* at local texture far faster than at globally consistent 3-D physics.

**Tier B — Medium durability:** Semantic/world-knowledge consistency (anatomy, text legibility, count consistency, physical plausibility of scene). These improve in generators but lag perceptual realism and require world modeling that remains hard.

**Tier C — Decaying (the entire incumbent paradigm):** Generator-specific spectral fingerprints, learned CNN "fakeness" features, diffusion-residual fingerprints, frequency-peak detectors. Useful **today**, near-worthless against the unseen generator of tomorrow. ARGUS uses them but treats them as *short-half-life* evidence with low reliability on unknown inputs.

**Design implication:** invert the field's priorities. The incumbent stack is 90% Tier-C fingerprinting. ARGUS makes **Tier A its backbone** and treats Tier C as corroborating, fast-decaying side-evidence. This single inversion is what gives ARGUS a chance against unseen generators.

\newpage
