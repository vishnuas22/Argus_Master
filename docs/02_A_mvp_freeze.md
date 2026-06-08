# A. MVP FREEZE SPECIFICATION

> **Thesis to prove and nothing more:** *reliability-aware tempered fusion of cheap classical forensics yields better calibration and selective risk under laundering than a single calibrated classifier — and degrades by abstaining rather than lying.*

## A.1 Exact modules (7) — computation, evidence, reliability gate

| # | Module | Exact computation | `evidence` from | `reliability` hard-zeroed when… |
|---|--------|-------------------|-----------------|--------------------------------|
| 0 | **Quality Profiler** | resolution; JPEG quant-table → Q; double-compression via DCT first-digit / Benford deviation; blockiness (8×8 grid energy); blur (variance of Laplacian); noise floor (MAD of wavelet HH); screenshot heuristic (no EXIF + exact-multiple dims + uniform Q) | — (produces the **quality vector** all gates consume) | never (it *is* the gate source) |
| 1 | **Metadata / EXIF / XMP** | cross-check camera↔lens↔resolution↔timestamp↔thumbnail↔maker-note | satisfied constraints / total | EXIF absent/stripped → r=0 (absence = **neutral**, never "fake") |
| 2 | **C2PA** | validate manifest signature chain + claim hashes | valid+capture → authentic; invalid → synthetic; AI-asserting → synthetic | manifest absent → r=0 (neutral) |
| 3 | **JPEG double-quant + grid** | per-8×8 DCT histograms; DQ-peak detection; grid-discontinuity map | inconsistency score | Q<60 or downscale flag or ≥2 JPEG generations → r→0 |
| 4 | **ELA** | recompress Q=90, abs-diff, normalize; region-variance of error | spatial error heterogeneity | screenshot / heavy recompression → r low |
| 5 | **FFT azimuthal spectrum** | 2-D FFT → log-mag → radial average → peak/periodicity | up-sampling peak strength | resample/blur → r→0; **OOD on unknown generator → r→0** |
| 6 | **Noise-residual / too-clean** | denoise → residual; residual energy + stationarity | departure from expected sensor-noise floor | low-res/screenshot → r→0; **corroborator only, capped weight** |

**Frozen module contract:** the Appendix-A JSON schema **plus** a mandated `reliability_var` (Beta posterior) propagated into Trust — closes the "reliability is a point estimate" flaw.

## A.2 Exact models
- **MVP backbone: ZERO trained models.** Fusion is the reliability-aware analytic pool (Bayesian mixture form — see §F repair). Reproducible, clean for the kill test.
- **Single optional contrast model (NOT in fusion floor):** `open_clip` **ViT-B/32**, weights **`laion2b_s34b_b79k`** → frozen features → logistic-regression probe → isotonic calibration → Mahalanobis OOD gate. Used **only** as the demoted-classifier baseline; OOD-gated to ≈0 on held-out generators.

## A.3 Exact libraries (pinned majors)
`python==3.11` · `numpy>=1.26` · `opencv-python-headless>=4.9` · `scipy>=1.11` · `PyWavelets>=1.5` · `scikit-image>=0.22` · `Pillow>=10` · `exifread>=3.0` · `piexif>=1.1` · `c2pa-python>=0.5` · `scikit-learn>=1.4` (isotonic) · `fastapi>=0.110` + `uvicorn` · `rq`/`celery` (queue) · `pymongo>=4.6` · React 18 + Vite.
**Frozen OUT of MVP:** `faiss`, `open_clip` (except the flagged contrast probe), `lightgbm`, `mapie`/`crepes`, `shap`, any VLM.

## A.4 Exact datasets

| Pool | Exact source | Use | Guard |
|------|--------------|-----|-------|
| Authentic-pristine | **RAISE-1k/8k** (RAW+JPEG) · **Dresden Image DB** | transfer-function fitting; pristine ECE | device-diverse subset enforced |
| Authentic-wild | **Flickr-CC pre-2021** (~5k) | laundered-real ECE | hard date cutoff < 2021 (pre-diffusion-saturation) |
| Synthetic (contrast only) | **GenImage** + **DiffusionForensics** subsets | fenced probe; LOGO arms | never enters fusion floor |
| **Suspicious-but-real** (build, ~500) | low-end phones, non-Western faces/dress, unusual optics, non-Latin script | **false-positive measurement** | the eval set the doc omits |
| Laundering transforms | JPEG-Q∈{95,85,75,60,40}, downscale∈{1.5,2,3}×, screenshot-sim, double-compress | reliability fitting + robustness curves | V2 → **real platform round-trips** |
| LOGO split | partition synthetic by generator family; hold out ≥2 families | open-set AUROC | report bootstrap CIs |

## A.5 Exact metrics (formula + pass threshold)
- **ECE** (15-bin adaptive) **and Brier**, reported **separately on pristine vs laundered**. *Pass:* laundered-ECE(ARGUS) < laundered-ECE(temperature-scaled CLIP probe), non-overlapping bootstrap CIs.
- **Risk–coverage** on the **hard subset** (no C2PA, no retrieval hit, laundered) **plus mandatory forced-decision accuracy at 100% coverage.**
- **Robustness curves:** AUROC vs JPEG-Q and vs downscale.
- **LOGO AUROC** with 1000× bootstrap CIs.
- **Decisive ablation metric:** Δ(laundered-ECE, hard-subset selective risk) between reliability-on and all-r=1 — *this is the publishable number.*

\newpage
