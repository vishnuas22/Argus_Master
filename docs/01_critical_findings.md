# CRITICAL FINDINGS (Items 1–8)

## 1. Missing assumptions
- **Ill-defined target.** Binary authentic/synthetic labels over a continuous edit spectrum (RAW → Lightroom → AI-denoise → 200 px generative fill → full synthesis). Every calibrator and the learned head consume a label that has no clean boundary.
- **Stationary, monotone degradation transfer functions** — false. Computational-photography denoise can *raise* a module's apparent self-consistency while *destroying* the ground-truth signal; the transfer function then points the wrong way.
- **A calibration target exists per module** — physics / semantic / contradiction scores have no natural probabilistic ground truth to isotonic-fit against; the doc hand-waves the labeled inconsistency data these need.
- **A `context_prior` exists at inference** — the whole posterior depends on it, and the MVP has *no mechanism* to obtain it.
- **Module orthogonality** is asserted, never measured (no mutual-information / covariance analysis). Wild-corpus reality is 2–4 firing, heavily correlated channels.
- **Image-only scope** silently excludes the 2026 video-deepfake majority.
- **C2PA + retrieval coverage rises** — both exogenous, outside ARGUS's control.

## 2. Hidden engineering risks
- **Calibration-data cost underestimated ~10×.** Per-module isotonic × degradation grid × generator families is a combinatorial labeled matrix needing hundreds–thousands of samples per cell, re-run as encoders/generators drift. Not "small and cheap."
- **Tooling fragility.** Young `c2pa-python`; `exiftool` Perl subprocess (packaging/licensing/version drift); `open_clip` version bumps silently invalidate the FAISS index (embeddings incomparable across versions); FAISS memory/build cost.
- **VLM non-determinism breaks the reproducible-audit-log claim** (same image → different verdict by date).
- **The "faithfulness checker" is an unsolved NLI problem** presented as a guarantee.
- **Three-stage fusion** makes single-verdict debugging muddy; "parallel modules" ignores the GIL + heavy CV/VLM latency (0.5–3 s ⇒ no inline use).

## 3. Scientific weaknesses
- Reliability-tempered LLR is heuristic, not calibrated (proof §F).
- Conditional independence false; LightGBM head re-imports the closed-world classifier Phase 1 condemned (≈0 open-set value).
- Conformal coverage void under shift (conceded, yet shipped).
- **"Too-clean" detector logically collides with "self-silencing"** — laundering both smooths images and lowers reliability; signal and silencing share a cause and cannot be cleanly separated from pixel statistics alone.
- **One-class anomaly framing does not escape the distribution problem** — it still needs a model of "real," i.e., the same biased, contaminated distribution; it hides the dependence rather than removing it.
- **Physics durability overclaimed** — any differentiable consistency test is a training objective for the next generator. The −4 the doc self-assigns is far too small for a load-bearing assumption.

## 4. Dataset weaknesses
- **Real/fake confounding never addressed.** Pairing FFHQ-style authentic against GAN/diffusion faces (where FFHQ *is* the GAN source) teaches alignment/resolution/source artifacts, not fakeness — inflating every seen-generator number.
- **No public C2PA corpus at scale** to train/validate the provenance backbone.
- **Retrieval needs a temporally-grounded corpus that does not exist publicly** (and is poisonable).
- **No "suspicious-but-real" negative set** → the liability-critical false-positive rate is unmeasurable.
- **Web contamination + Western / high-end / English skew** → deployment-blocking bias liability.

## 5. Evaluation weaknesses
- Selective-risk is gameable — report **hard-subset forced-decision** accuracy.
- LOGO measures interpolation across contemporary generators, sold as extrapolation to future ones.
- **No significance testing / CIs**; ECE is binning-biased at small N — add Brier + bootstrap CIs.
- **Faithfulness-by-ablation is confounded by module correlation** (a correlated backup masks true reliance).
- **Strawman baseline.** A lone Xception/CLIP probe is too weak to claim a calibration win; must beat temperature-scaled / ensemble / recent open-set detectors.

## 6. Components that are overengineered
1. **Conformal/Mondrian abstention** — guarantee conceded void; a threshold rule does the same job at 5% complexity.
2. **LightGBM correction head** — closed-world coupling for ≈0 open-set benefit.
3. **Contradiction *graph*** — pairwise flags over ≤4 high-reliability modules suffice; "graph" is presentation.
4. **PRNU + latent-inversion + reference-device** — near-useless on wild traffic; research-grade cost.
5. **Risk harm-weighting** — needs a context/harm model that does not exist in MVP.
6. **Three-stage cascade fusion** — one well-fit pooled+abstention model proves the thesis.

## 7. Components to postpone to V2+
CLIP+FAISS retrieval at scale; VLM physics/semantic (advisory-only, non-scoring); LightGBM head; conformal layer; OOD/Mahalanobis gating; device-conditioned too-clean; active-evidence-acquisition workflow; frozen Daubert/forensic mode; any video.

## 8. Components mandatory for MVP
Ingest + quality profiler; EXIF/XMP internal-consistency + C2PA presence/validity; JPEG (double-quant + grid) + ELA; FFT azimuthal spectrum; noise-residual / weak too-clean; the `{evidence, reliability, confidence}` contract; **degradation transfer functions (the real testable novelty)**; reliability-tempered log-odds pool + threshold abstention; tri-axial display + evidence-ranking XAI + heatmaps; an evaluation harness (degradation sweep + small public LOGO + ECE/Brier + risk–coverage on a hard subset).

\newpage
