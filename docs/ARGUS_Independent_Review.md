# ARGUS — Independent Adversarial Feasibility Review

Reviewer stance: Principal Research Scientist / Distinguished Engineer / DARPA PM / CVPR AC / Startup CTO.
Verdict in one line: **The classical-forensics kernel is buildable and mildly novel; the headline "calibrated Bayesian reliability fusion" is mathematically overclaimed; the open-set thesis rests on a decaying prior and two exogenous crutches. Fundable as a research program and an analyst-assist tool, not as an oracle.**

---

## 0. The two things the authors' own red-team did NOT catch (most important)

1. **`LLR_i' = reliability_i · LLR_i` is not Bayesian and breaks "calibrated by construction."**
   Linearly scaling a log-likelihood ratio by a reliability scalar in [0,1] is equivalent to raising the likelihood ratio to the power `r` (logarithmic-opinion-pool / likelihood tempering). That is a *heuristic* pooling rule with no probabilistic guarantee of calibration. The Bayesian-correct treatment of a noisy channel is to **marginalize over the channel's reliability** (a mixture: `P(e) = r·P(e|works) + (1-r)·P(e|dead)`), which is **not** linear attenuation of log-odds. So the system's central selling point — "calibrated by construction" — is false. It is calibrated *only if* every module is conditionally independent AND every reliability equals 1. Neither holds. Calibration here is an empirical hope, not a construction.

2. **Trust and Risk are not "orthogonal outputs" — they are algebraic re-projections of the same evidence vector.**
   `Trust = f(mean_reliability, coverage, 1−contradiction, 1−OOD)` and `Risk = g(1−Authenticity, contradiction, prior, harm)` are deterministic functions of the *same* underlying quantities that produce Authenticity. Three numbers derived from one evidence vector are three views, not three independent axes. The "three orthogonal questions" framing is presentation, not information-theoretic orthogonality. Useful for a UI; not a scientific contribution.

These two undercut the document's two proudest claims (calibration + tri-axial novelty). Everything below is secondary to them.

---

## 1. Missing assumptions

- **A target-definition assumption.** ARGUS trains on a binary authentic/synthetic label, yet the real distribution is a *spectrum*: RAW → Lightroom edit → AI denoise → generative fill of a 200px region → full synthesis. "Authentic capture of a real scene" has no clean decision boundary, and the ground-truth label that every calibrator and the LightGBM head consume is therefore ill-defined. The tri-axial output hides, but does not resolve, this.
- **Stationarity of degradation transfer functions.** The reliability layer assumes the map (quality-vector → max reliability) is stationary and roughly monotone. Real pipelines are non-monotone and interacting: computational-photography denoise can *raise* a module's apparent self-consistency while *destroying* the ground-truth signal — the transfer function then points the wrong way.
- **Calibration-target existence per module.** Physics/semantic/contradiction scores have no natural probabilistic ground truth to isotonic-calibrate against. The doc assumes every module's raw score can be mapped to a calibrated LLR; for the durable Tier-A modules this requires labeled inconsistency data that is hand-waved.
- **A context_prior is available at inference.** The whole posterior depends on a prior the MVP has no mechanism to obtain. (Red-team 1.4 catches the *non-stationarity*; the deeper gap is that there is **no prior source at all** in the MVP.)
- **Module orthogonality is asserted, never measured.** No mutual-information / covariance analysis. "20+ orthogonal modules" is a marketing count; the wild-corpus reality is 2–4 firing, heavily correlated channels (red-team 2.3 is right).
- **Image-only scope.** The 2026 threat is disproportionately *video* face-swap/lip-sync. Restricting to single images is a silent, large narrowing of the actual problem.
- **Provenance/attestation adoption rises.** C2PA coverage and a useful retrieval corpus are assumed to grow — both exogenous and outside ARGUS's control (the red-team's "crutches" point; correct).

## 2. Hidden engineering risks

- **The calibration-data requirement is underestimated by an order of magnitude.** "Small, cheap, self-supervised" is wrong: per-module isotonic calibration × a degradation grid (JPEG-Q × downscale × screenshot × double-compression) × multiple generators is a combinatorial matrix needing hundreds–thousands of labeled samples *per cell*. This is the real cost center, and it must be **re-run as platform encoders and generators drift**.
- **Tooling fragility.** `c2pa-python` is young and ABI-churny; `exiftool` is a Perl subprocess dependency (packaging/licensing/version drift); `open_clip` weight/version changes silently invalidate the entire FAISS index (embeddings are not comparable across versions); FAISS memory/build time for any non-toy corpus is non-trivial.
- **VLM non-determinism breaks the audit-log reproducibility claim.** Phase 7 promises reproducible, contestable verdicts; an API VLM in the loop makes the same image yield different narratives/flags on different days (red-team 3.4 touches this for legal; it's also a plain engineering contradiction).
- **The "faithfulness checker" is an unsolved problem presented as a guarantee.** "Every claim must map to an evidence item or be rejected" requires reliable NLI/hallucination detection — itself an open research area. As specified it is aspirational.
- **Three-stage fusion = three separately-fit, jointly-coupled components.** Debugging a single wrong verdict across Bayesian core → LightGBM head → conformal wrapper is genuinely hard; failure attribution is muddy.
- **"Parallel modules" ignores Python/GIL + heavy OpenCV/FFT/VLM realities.** No latency budget is given; red-team 4.2's 0.5–3 s estimate is credible and kills inline use.

## 3. Scientific weaknesses

- **(Headline) Reliability-tempered LLR is heuristic, not calibrated** — see §0.1.
- **Conditional independence is false and the "fix" re-imports closed-world** — red-team 1.3 is correct; the LightGBM head is the very classifier Phase 1 condemned, fenced to ≈0 value on the open set.
- **Conformal coverage is void under shift** — red-team 1.1 is correct and, to their credit, conceded. But shipping a layer whose guarantee evaporates exactly when invoked is still a net-negative complexity in the MVP.
- **"Too-clean" detector logically collides with "self-silencing."** Laundering smooths images *and* lowers reliability; "implausibly clean" and "low-reliability degraded" are entangled and cannot be cleanly separated from pixel statistics alone. The signal and its own silencing condition share a cause.
- **One-class anomaly framing does not escape the distribution problem.** A one-class "is this consistent with a real capture?" model still needs a model of *real* — i.e., the same biased, web-contaminated distribution. One-class hides the dependence; it doesn't remove it.
- **Durability of physics is overclaimed.** Any forensic consistency test you can write as a differentiable check is a *training objective* for the next generator (red-team 2.1). The doc's backbone is a to-do list for the adversary. The −4 they self-assign is far too small for a load-bearing assumption.

## 4. Dataset weaknesses

- **Real/fake confounding is never addressed.** Pairing FFHQ-style authentic faces against GAN/diffusion faces (where FFHQ *is* the GAN training source) makes the "fakeness" classifier learn alignment/resolution/source artifacts, not synthesis. This is the field's classic confound and it silently inflates every seen-generator number.
- **No public C2PA corpus at scale** to train or validate the provenance backbone — the module's real-world behavior is essentially untested.
- **Retrieval needs a corpus + temporally-grounded query set that does not exist publicly**; building one with trustworthy timestamps is a major, unbudgeted project (and poisonable — red-team 2.4).
- **No "suspicious-but-real" negative set.** Without a curated set of *authentic* images that look weird (medical, cultural dress, rare optics, low-end devices), the false-positive rate — the liability-critical metric — cannot be measured.
- **Web contamination + demographic skew** — red-team 5.2/5.3 are correct and among the strongest points; the bias issue is a deployment-blocking liability, not a footnote.

## 5. Evaluation weaknesses

- **Selective-risk is gameable; report hard-subset forced-decision accuracy** — red-team 1.5 correct.
- **LOGO measures interpolation, sold as extrapolation** — red-team 5.1 correct.
- **No significance testing / CIs.** AUROC/ECE on a handful of held-out generators are high-variance; single point numbers are not publishable as-is. Need bootstrap CIs and proper scoring rules (Brier), not just ECE (binning-biased at small N).
- **Explanation-faithfulness metric is confounded by module correlation.** "Remove top cue, does the verdict move?" falsely reads *low* faithfulness when a correlated backup module compensates. Faithfulness must be measured with correlation-aware ablations.
- **Weak baseline set.** A lone Xception/CLIP probe is a strawman. To claim the *calibration* win they must beat temperature-scaled / ensemble / recent open-set baselines (e.g., CLIP-based detectors with post-hoc calibration). Otherwise the headline "better ECE" is uncontested and unconvincing.

## 6. Components that are overengineered (for what they deliver)

1. **Conformal/Mondrian/shift-weighted abstention layer** — guarantee conceded void; a thresholded reliability+coverage abstention rule does the same job at 5% of the complexity.
2. **LightGBM correction head** — adds closed-world coupling for ≈0 open-set benefit.
3. **Contradiction *graph*** — pairwise contradiction flags over the ≤4 high-reliability modules suffice; "graph" is presentation.
4. **PRNU + latent-inversion + reference-device matching** — near-useless on wild traffic; research-grade cost.
5. **Risk axis with harm-weighting** — needs a context/harm model that does not exist in MVP; derive it later.
6. **Three-stage cascade fusion** — one well-fit pooled+abstention model proves the thesis.

## 7. Components to postpone to V2+

CLIP+FAISS retrieval at scale (corpus + integrity defense); VLM physics/semantic (and then advisory-only, non-scoring); LightGBM head; conformal layer; OOD Mahalanobis gating (only matters once learned modules carry weight); device-conditioned too-clean model; active-evidence-acquisition workflow; frozen Daubert/forensic mode; any video support.

## 8. Components mandatory for MVP

Ingest + quality profiler; metadata/EXIF/XMP internal-consistency + C2PA presence/validity parsing; JPEG forensics (double-quant, grid discontinuity) + ELA; FFT azimuthal spectrum; noise-residual / weak too-clean; the `{evidence, reliability, confidence}` module contract; **degradation transfer functions for the classical modules** (this is the actual testable novelty); a simple reliability-tempered log-odds pool + threshold abstention; tri-axial display + evidence-ranking XAI + heatmaps; an evaluation harness (degradation sweep + small public LOGO + ECE/Brier + risk–coverage on a hard subset).

---

## A. MVP FREEZE SPECIFICATION

**Thesis to prove (and nothing more):** *reliability-aware tempered fusion of cheap classical forensics yields better calibration and selective risk under laundering than a single calibrated classifier — and degrades by abstaining rather than lying.*

- **Modules (7, all CPU, no VLM, no learned head):**
  1. Quality Profiler (resolution, JPEG-Q estimate, double-compression, blockiness, Laplacian blur, noise floor, screenshot heuristic, EXIF presence).
  2. Metadata/EXIF/XMP internal-consistency.
  3. C2PA presence + signature validity (parse only; absence = neutral).
  4. JPEG forensics: double-quantization + grid discontinuity (+ heatmap).
  5. ELA (recompression-diff heatmap).
  6. FFT azimuthal/radial power-spectrum peakiness.
  7. Noise-residual consistency / weak "too-clean" corroborator.
- **Models:** none trained as backbone. One *optional, fenced, OOD-gated* CLIP-ViT-B/32 linear probe purely as the demoted-classifier contrast baseline — not in the fusion floor.
- **Libraries:** Python; OpenCV, NumPy, SciPy, PyWavelets, scikit-image, Pillow; `exifread`/`piexif`, `c2pa-python`; scikit-learn (isotonic); FastAPI + a queue; MongoDB (audit log); React dark console UI. (open_clip/faiss/lightgbm/mapie/shap **excluded from MVP**.)
- **Datasets:** Authentic — RAISE + Dresden (camera-pipeline/noise calibration), pre-2021 Flickr-CC subset (contamination-safe). Synthetic — GenImage / DiffusionForensics public subsets for the contrast probe only. Degradation calibration — self-supervised sweep over the authentic pool. **Add a small curated "suspicious-but-real" set** (low-end phones, non-Western, unusual optics) for false-positive measurement.
- **Metrics:** primary = ECE **and Brier** computed *separately* on pristine vs laundered; risk–coverage / selective accuracy reported **on the hard subset** + a mandatory forced-decision (100% coverage) number; robustness curves vs JPEG-Q and downscale; small LOGO AUROC with **bootstrap CIs**; the decisive ablation = **reliability layer on vs. all-reliability-=1** (see Kill Test).

## B. RESEARCH ROADMAP

- **V1 (3 months) — Prove or kill the kernel.** Build the 7 classical modules + degradation transfer functions + reliability-tempered pooling + threshold abstention + XAI + eval harness. Run the ablation kill test. Deliverable: a paper-grade result on *calibration-under-laundering* and *graceful degradation* vs a calibrated single classifier, on open data. If the ablation shows no gain, stop here — that is success-as-falsification.
- **V2 (6 months) — Open-set + retrieval, honestly scoped.** Add domain-scoped CLIP+FAISS retrieval with timestamp-corroboration and collision monitoring; OOD/Mahalanobis gating on the contrast probe; VLM as **non-scoring advisory** with prompt-injection defense; per-subgroup error reporting as a release gate; real-platform (not simulated) degradation recalibration. Add a frozen/versioned "forensic mode" for reproducibility.
- **V3 (12 months) — Layer, not oracle.** Position ARGUS as one layer in a multi-signal pipeline: capture-time/hardware attestation hooks, network/propagation signals, multi-frame & cross-source corroboration, video support, continuous prospective generator testing, Daubert-mode error-rate publication, bias audit + targeted data acquisition.

## C. KILL TEST (minimum experiment that proves ARGUS fundamentally wrong)

**Primary kill test — the reliability ablation (cheap, decisive, runnable in V1):**
Hold the entire pipeline fixed. Produce two variants on the same LOGO + laundering benchmark:
- **ARGUS-full**: reliability-tempered pooling (`LLR' = r·LLR`).
- **ARGUS-null**: identical modules and calibrators but **all reliabilities forced to 1** (plain pooled evidence, no self-silencing).

ARGUS's *entire* novel claim is that the reliability layer matters. **If ARGUS-null matches ARGUS-full (within bootstrap CIs) on ECE-under-laundering and on hard-subset selective risk, the central contribution is empirically null** and the architecture reduces to "an ensemble of known forensics." Add a third arm — a temperature-scaled single CLIP probe — and if it matches both on calibration, the *raison d'être* is gone.

**Secondary kill test — durability falsification:** take one recent physics-/geometry-aware generator, measure the physics/shadow/lighting module AUROC. **If ≤ 0.60, the Tier-A "durable backbone" is already chance-level** and the open-set value proposition collapses (this directly tests red-team 2.1 with one experiment).

Either failure independently invalidates the thesis; both are runnable on consumer hardware in days.

## D. PUBLISHABILITY ANALYSIS

**Genuinely novel (modest-but-real):**
- Reliability **decoupled from confidence** via empirically-calibrated **degradation transfer functions**, with demonstrable self-silencing. This is the one fresh, paper-sized idea — but only if backed by the ablation above showing it *causes* the calibration gain.
- The **evaluation reframing** (ECE-under-laundering + hard-subset selective risk + prospective generator testing) is a solid position/benchmark contribution with citation potential.

**Weak / not novel:**
- "Calibrated-by-construction Bayesian fusion" (mathematically overclaimed — §0.1).
- LightGBM correction head, conformal-under-shift, contradiction graph, tri-axial "orthogonality" — incremental or presentational.
- The individual forensic modules are all prior art.

**Acceptance odds — conditional on real experiments (a pure document is 0% everywhere):**
- **WIFS:** 50–65%. Reliability-calibration + forensic rigor + reproducible open-data eval is squarely in scope; the degradation-transfer-function paper fits best.
- **IH&MMSec:** 45–60%. Values reproducibility and forensic method; the calibration-method paper lands here. Reviewers will press on real/fake confounding and real-platform calibration.
- **CVPR Workshop (media forensics / DFAD / WMF):** 55–70%. Workshops reward the framing + a working demo + tri-axial UX story.
- **ICCV Workshop:** 55–70%. Comparable to CVPR workshop.
- **Main CVPR/ICCV/NeurIPS:** <15% — no single strong empirical result; reviewers hit §0–§5 flaws.
Best single shot: the **degradation-transfer-function reliability-calibration** paper at WIFS/IH&MMSec, with the ablation kill test as the centerpiece.

---

## FINAL SCORES (independent, brutal)

| Axis | Score | One-line justification |
|---|---|---|
| **Research Score** | **57 / 100** | Real reframing of metrics + one fresh idea (reliability≠confidence), undermined by a flawed "calibrated" core and a backbone that's already decaying. |
| **Engineering Score** | **47 / 100** | The classical kernel is very buildable; the *specified full system* is an MLOps swamp with underestimated calibration-data cost, non-deterministic audit path, and fragile three-stage fusion. |
| **Startup Viability** | **36 / 100** | No defensible tech moat (open-source mandate), abstention-as-liability, two exogenous crutches, niche analyst-assist market. Services/trust play at best. |
| **Scientific Novelty** | **50 / 100** | Reliability-decoupling + degradation transfer functions is genuinely fresh but modest; ~90% of the stack is known forensics re-packaged; system "novelty" is incremental. |

**Bottom line.** ARGUS is intellectually honest and, in its *classical-forensics + reliability-calibration kernel*, a buildable and mildly publishable research artifact. But its proudest claims — "calibrated by construction," "tri-axial orthogonal outputs," "durable physics backbone" — are respectively a mislabeled heuristic, an algebraic re-projection, and a depreciating asset. The authors' own red-team is unusually good and lands ~20 real hits; it nonetheless missed the non-Bayesian reliability tempering and the false orthogonality, which are the two cleanest ways to puncture the paper. **Fund the kernel and the metric-reframing as a research program; run the one-day reliability-ablation kill test before anything else; do not deploy as an oracle, and never as real-time or court-of-record without a frozen mode.**
