# REFERENCES

Sources underpinning the major claims in this design. Grouped by topic; all verified accessible as of June 2026.

## Training-Free / Zero-Shot AI-Image Detection
- **SpAN** — spectral analysis of upsampling artifacts with power calibration; +0.241 AUROC over prior training-free methods on Synthbuster. ICLR 2026 submission. https://openreview.net/forum?id=G9Oj0dMQIJ
- **Efficient Zero-Shot AI-Generated Image Detection** (structured frequency-perturbation sensitivity in frozen ViTs; outperforms AEROBLADE/RIGID, evaluated on OpenFake/GenImage/Semi-Truth across 12+ generators). arXiv 2026. https://arxiv.org/html/2603.21619v1
- **Intermediate Representations are Strong Training-Free AI-Generated Image Detectors.** ICLR 2026 submission. https://openreview.net/forum?id=NfM92qRuew
- **ZED** — zero-shot detection trained on real images only. ECCV 2024. https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02665.pdf
- **B-Free** (GRIP-UNINA, bias-free training discussion). https://grip-unina.github.io/B-Free/
- Survey/aggregation: Awesome-AIGC-Image-Video-Detection. https://github.com/ant-research/Awesome-AIGC-Image-Video-Detection

## Backbone Robustness (CLIP vs DINO/DINOv2)
- CLIP vs DINO/DINOv2 under transformations — CLIP avg. 41.6% on transformed fakes vs DINO 91.8% / DINOv2 92.3%; CLIP std-dev 24.6% across transformations. ICIAP 2023. https://iris.unimore.it/retrieve/handle/11380/1309209/576860/2023-iciap-deepfake.pdf
- SSL ViTs (DINOv2 > supervised ViT and EVA-CLIP for deepfake detection). arXiv 2024. https://arxiv.org/html/2405.00355v1
- **GenD** — parameter-efficient DINO-based method, best cross-dataset AUROC over 14 benchmarks (2019–2025). WACV 2026. https://cmp.felk.cvut.cz/ftp/articles/cech/Yermakov-WACV-2026.pdf
- **FatFormer** — CLIP-adapted detector, ~98% unseen GANs / ~95% unseen diffusion (reported setup). CVPR 2024. https://github.com/Michel-liu/FatFormer
- **UniAIDet** benchmark — category-shift and partial-synthesis failures. arXiv 2025. https://arxiv.org/html/2510.23023v1
- Beyond-benchmark generalization limits of deepfake detectors in the wild. UC Berkeley I-School 2025. https://www.ischool.berkeley.edu/projects/2025/beyond-benchmark-generalization-limits-deepfake-detectors-wild

## Noise-Residual Forensics & Localization
- **TruFor** — Noiseprint++ residual + RGB fusion; anomaly map + confidence map. CVPR 2023. https://grip-unina.github.io/TruFor/ · paper: https://arxiv.org/pdf/2212.10957
- GRIP-UNINA image forgery detection overview (incl. Comprint, compression fingerprints). https://www.grip.unina.it/multimedia-forensics/image-forgery-detection
- Social-media forgery-localization benchmark — 15 SOTA methods degrade under compression/resizing/filtering. Ulster University, 2025. https://pure.ulster.ac.uk/en/publications/a-benchmark-for-image-forgery-detection-and-localization-on-socia/

## Uncertainty Quantification
- Conformal prediction — distribution-free coverage; gentle introduction: Angelopoulos & Bates. https://arxiv.org/abs/2107.07511
- Conformal prediction applied to deepfake-detection reliability. Nature Scientific Reports 2024. https://www.nature.com/articles/s41598-024-65954-w
- Evidential deep learning overview (epistemic/aleatoric separation, OOD uncertainty inflation) — surveyed for Phase 4 design comparison.

## Evidence Fusion
- **FRAME** — adaptive multi-path forensic routing and fusion for manipulation detection. arXiv 2026. https://arxiv.org/abs/2605.12826
- Fontani et al. — Dempster-Shafer multi-clue fusion in image forensics. IEEE WIFS 2013. http://carmelatroncoso.com/papers/Fontani-WIFS13.pdf
- Mixture-of-experts for deepfake detection. arXiv 2024. https://arxiv.org/html/2409.11909v1
- Bayesian vs Dempster-Shafer fusion comparison. https://www.lusispayments.com/uploads/4/4/8/2/44826195/bayesian_dempster-shafer_models.pdf
- Farid — Digital Image Forensics tutorial (classical multi-cue doctrine). https://farid.berkeley.edu/downloads/tutorials/digitalimageforensics.pdf

## Physics, Geometry & Semantic Forensics
- Photo forensics from lighting, shadows, reflections (Farid/CAI). https://contentauthenticity.org/blog/photo-forensics-from-lighting-shadows-and-reflections
- Evaluating projective geometry of AI-generated images. https://projective-geometry.github.io/static/files/Evaluating_Projective_Geometry.pdf
- **Light2Lie** — physically-grounded reflectance inconsistencies for deepfake detection. NDSS 2026. https://www.ndss-symposium.org/wp-content/uploads/2026-s923-paper.pdf
- Shadow/reflection checks in practice (Amped Authenticate workflow). https://www.forensicfocus.com/articles/how-to-reveal-ai-generated-images-by-checking-shadows-and-reflections-in-amped-authenticate/
- DARPA SemaFor — semantic forensics program. https://www.darpa.mil/research/programs/semantic-forensics

## Provenance / C2PA
- C2PA Specification v2.4 (April 2026). https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html
- State of Content Authenticity 2026 (CAI). https://contentauthenticity.org/blog/the-state-of-content-authenticity-in-2026
- Camera adoption matrix (Leica M11-P/SL3-S; Sony A1 II/A9 III/FX-line; Canon R1/R5 II). https://www.lumethic.com/en/articles/cameras-with-c2pa-content-credentials · https://global.canon/en/news/2026/20260511.html
- Security/conformance weaknesses of current C2PA (revocation, expiry). arXiv 2026. https://arxiv.org/html/2604.24890v1
- NSA/CISA guidance on Content Credentials. https://media.defense.gov/2025/Jan/29/2003634788/-1/-1/1/CSI-CONTENT-CREDENTIALS.PDF

## Datasets & Benchmarks (Phase 8)
- GenImage (8-generator benchmark; JPEG/PNG confound discussed in Phases 8/10) · Synthbuster (recent commercial generators) · DF40 · CASIA v2 · Columbia · IMD2020 · DEFACTO · COCO · RAISE · Dresden Image Database · OpenImages · FFHQ.

## Tooling (Phase 8)
- PyTorch · timm (DINOv2) · FAISS · OpenCV · Pillow · jpegio · scikit-learn · LightGBM · SHAP · MAPIE (conformal) · pyexiftool/exiftool · c2pa-python (CAI SDK) · Tesseract · FastAPI · React.
