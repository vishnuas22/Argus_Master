# PHASE 3 — Evidence Source Taxonomy

> A complete, scored taxonomy of the 22 requested evidence sources. Scores are 1–5 unless noted. **Robustness** = expected usefulness against future (2027+) generators. **Laundering vulnerability** = how badly social-media processing (JPEG ~q70, resize, screenshot, re-encode) damages it (5 = nearly destroyed). **Explainability** = can a human verify the claim? **Cost** = compute on commodity CPU. **REC** = recommendation score for ARGUS (0–10), the bottom line.

---

## 3.0 Summary Scorecard

| # | Evidence source | Family | Future robustness | Laundering vuln. | Explainability | Cost | REC |
|---|---|---|---|---|---|---|---|
| 1 | Metadata (EXIF/XMP/container) | Provenance-adj. | 3 | 4 (often stripped) | 5 | trivial | **8** |
| 2 | Camera pipeline artifacts (general) | Acquisition | 3 | 4 | 3 | low | 6 |
| 3 | CFA / demosaicing analysis | Acquisition | 2 | **5** | 3 | low | 3 |
| 4 | Sensor noise (statistical, non-PRNU) | Acquisition | 3 | 4 | 3 | low | 6 |
| 5 | PRNU (camera fingerprint) | Acquisition | 2 | **5** | 4 | med + ref. DB | **2** |
| 6 | JPEG traces (ghosts, DQ, quant tables) | Compression history | 4 | 2 (it *measures* laundering) | 4 | low | **9** |
| 7 | Frequency analysis (FFT/DCT spectra) | Statistical | 3 (declining) | 4 | 4 | trivial | **8** |
| 8 | Wavelet statistics | Statistical | 2 | 4 | 2 | low | 4 |
| 9 | Residual analysis (learned, Noiseprint++) | Statistical | 3 | 4→conditioned | 4 (maps) | med | **8** |
| 10 | Diffusion artifacts (generator-specific) | Generative trace | **1** | 4 | 3 | low | 3 |
| 11 | Latent inversion (DIRE/AEROBLADE-class) | Generative trace | **1–2** | 4 | 2 | **high** | **3** |
| 12 | Retrieval systems (reverse search) | External context | **5** | **1** | **5** | network | **9** |
| 13 | Embedding similarity (DINOv2 real-distribution) | Realness model | **4–5** | 2 | 3 | med | **9** |
| 14 | Physics consistency (lighting/perspective) | Physical | **5** | **1–2** | **5** | med | **9** |
| 15 | Reflection consistency | Physical | 5 | 2 | 5 | med | 8 (recall-limited) |
| 16 | Shadow consistency | Physical | 5 | 1 | 5 | med | 8 (recall-limited) |
| 17 | Semantic consistency (world-logic) | Semantic | 3 (declining) | **1** | **5** | med | **8** |
| 18 | Typography analysis | Semantic | 3 | 2 | 5 | low | 7 |
| 19 | Texture analysis (micro-texture stats) | Statistical | 2 | **5** | 2 | low | 4 |
| 20 | Provenance systems (general) | Provenance | **5** | n/a (binary: present/absent) | **5** | trivial | **9** |
| 21 | C2PA specifically | Provenance | **5** | strippable | 5 | trivial | **9** |
| 22 | Content Credentials ecosystem | Provenance | **5** | strippable | 5 | trivial | **9** |

**Panel selection logic:** ARGUS's nine modules (Phase 2) are the REC ≥ 7 rows, merged into coherent modules; REC ≤ 4 rows are excluded or demoted to optional plugins, with reasons below.

---

## 3.1 Provenance Family

### 1. Metadata (EXIF / XMP / IPTC / container structure)
What it is: camera make/model/settings, software history, GPS, timestamps; plus container-level features — quantization tables (camera-vendor-characteristic), thumbnail consistency, marker ordering, file-structure fingerprints of known tools.
- **Robustness 3:** generators/tools leave tell-tale absence or wrong structure; sophisticated adversaries forge it perfectly. Hence **asymmetric use**: anomalies (e.g., "Photoshop 25.3" in history; quant table matching no known camera; thumbnail ≠ image) are meaningful; pristine metadata proves little.
- **Laundering 4:** platforms strip most metadata — but *stripping is itself detected* (feeds the degradation state `d`).
- **Explainability 5:** "EXIF says iPhone 14, but quantization tables match Stable Diffusion's PIL export" is courtroom-grade.
- **Cost:** trivial (exiftool). **REC 8** — cheap, asymmetrically useful, feeds triage.

### 20–22. Provenance systems / C2PA / Content Credentials
2026 status: spec v2.4; capture-time signing in Leica M11-P/SL3-S, Sony A1 II/A9 III/FX-line, Canon R1/R5 II (paid activation); Google, Meta, TikTok, OpenAI, LinkedIn in the ecosystem; OpenAI signs DALL·E/GPT-image outputs (valid manifest can *prove AI origin* — equally valuable).
- **Robustness 5:** orthogonal to generator progress — the only evidence class that gets *stronger* every year.
- **Laundering:** binary — survives intact or is stripped entirely; absence ≈ no evidence (in 2026, most legit images are unsigned).
- **Known weaknesses (must be engineered around):** revoked-cert handling and post-expiry unverifiability (2026 preprint); trust-list governance; "analog hole" (photograph a screen with a signing camera → signed photo *of* a fake). ARGUS therefore reports provenance as "manifest valid, signer X, chain Y" — a *checkable claim*, not blind trust.
- **REC 9** — Tier 0 fast-path; the long-term strategic bet.

### 12. Retrieval systems (reverse image search, near-duplicate indexing)
- **Robustness 5:** generators cannot fabricate an image's *history on the internet*. Earliest-appearance dating, crop/re-edit ancestry, and matches against known-fake corpora are decisive when available.
- **Laundering 1:** modern perceptual hashing/embedding retrieval is robust to crops/recompression by design.
- **Cost:** requires an index or third-party API (network-bound) → **optional at MVP**, first post-MVP addition. **REC 9.**

---

## 3.2 Acquisition-Pipeline Family

### 2. Camera pipeline artifacts (general: lens distortion, chromatic aberration, vignetting consistency)
Real optics impose radially-consistent aberrations; generators produce them only as *style*. Moderate signal, moderate explainability. Laundering-sensitive (resize destroys radial structure). **REC 6** — folded into module D/G rather than standalone.

### 3. CFA / demosaicing analysis
Detects the periodic inter-pixel correlations of Bayer interpolation. **Excluded (REC 3):** the first resize or recompression annihilates the 2×2 periodic lattice — near-zero reliability on social-media traffic, which is the design center. Worth a plugin for the "pristine upload" niche where it is excellent.

### 4. Sensor noise (statistical: noise-floor level, shot-noise vs ISO consistency, spatial uniformity)
Unlike PRNU, needs no reference camera. AI images often show *implausibly clean or spatially uniform* noise; real high-ISO photos have characteristic luminance-dependent noise. Moderately laundering-sensitive (denoise + compression). **REC 6** — folded into module D.

### 5. PRNU (Photo-Response Non-Uniformity)
The classic camera fingerprint. **Demoted (REC 2)** for ARGUS's mission: (i) answers "did camera X take this?", not "is this AI?" — wrong question without a reference database; (ii) destroyed by resize/compression; (iii) reference sets are proprietary-by-nature (per-device images). Correct tool for *device attribution* in law enforcement; wrong tool here. This is a deliberate break from classical forensics tradition.

---

## 3.3 Compression-History Family

### 6. JPEG traces (ELA, JPEG ghosts, double-quantization, quant-table forensics)
The workhorse. ELA/ghosts localize regions with divergent compression history (splices, inpainting); DQ-histogram analysis counts recompression generations; quant tables fingerprint producing software.
- **Robustness 4:** measures *file history*, not generator output — generators don't obsolete it.
- **Laundering 2 (inverted!):** laundering doesn't destroy this evidence; laundering **is** this evidence. The module powers Tier-1 triage.
- Caveat: blind to whole-image generations (uniform history) — by design covered by modules C/E/F.
- **REC 9.**

---

## 3.4 Statistical / Frequency Family

### 7. Frequency analysis (FFT/DCT radial spectra, peak detection)
Upsampling layers imprint spectral peaks/slope anomalies; 2026 training-free SOTA (SpAN: +0.24 AUROC over prior training-free on Synthbuster) shows this is still strong *if* power-calibrated. Must be disambiguated from resize-laundering peaks → uses `d`. Declining as generators clean spectra, but nearly free. **REC 8.**

### 8. Wavelet statistics
Higher-order wavelet-coefficient statistics (the 2000s Farid-school approach). Largely superseded by learned residuals; weak explainability. **REC 4** — skip; module D dominates it.

### 9. Residual analysis (learned: Noiseprint++ / TruFor-class)
A network trained (on *real* images, self-supervised, camera-model discrimination) to extract a "pipeline fingerprint" residual; anomalies in the residual localize edits, residual *uniformity/absence* flags synthesis. Ships with its own confidence map — the published precedent for ARGUS's reliability outputs.
- **Laundering:** suppressed by compression — *the* prime client of degradation-conditioned reliability `r_m(d)`.
- License caveat (research-only weights) handled in Phase 10. **REC 8.**

### 19. Texture analysis (micro-texture statistics: LBP/GLCM-class)
High-frequency, fragile, weakly explainable, superseded. **REC 4** — skip.

---

## 3.5 Generative-Trace Family

### 10. Diffusion artifacts (generator-specific fingerprints)
Per-family spectral/spatial signatures. Excellent on the generator they describe; obsolete on the next one (**robustness 1**). Acceptable only as *named, perishable* plugins ("matches SDXL-class fingerprint" is useful attribution metadata when it fires). **REC 3** — optional plugin, never load-bearing.

### 11. Latent inversion (DIRE / AEROBLADE-class)
Invert through a diffusion autoencoder/model; small reconstruction error ⇒ image lies on that generator-family's manifold.
- **Cost: high** (multiple network passes — worst cost/benefit in the taxonomy on CPU).
- **Robustness 1–2:** family-specific by construction; 2026 training-free literature already treats AEROBLADE as an outperformed baseline.
- **Explainability 2:** "low reconstruction error" is opaque to humans.
- **REC 3** — optional plugin for attribution, off by default. *Deliberate demotion vs. the original brief's implied emphasis.*

### 13. Embedding similarity / real-distribution probe (DINOv2)
The A2 flagship (Phase 2, module E): frozen DINOv2 ViT-B/14, kNN/one-class scoring against a large real-only reference set; plus the perturbation-sensitivity probe (module F) sharing the same forward pass.
- **Robustness 4–5:** trained on nothing fake → cannot overfit to generators; degrades gracefully under convergence (L4), never inverts.
- **Laundering 2:** DINOv2 representations empirically robust under transformations (~92% vs CLIP ~42%).
- **Explainability 3** (weakest trait): "far from real-image neighborhoods" needs the verdict layer to translate it (nearest-neighbor exemplars help). **REC 9.**

---

## 3.6 Physical-Consistency Family

### 14–16. Physics / reflection / shadow consistency
Projective-geometry checks: object→shadow lines must co-intersect (light source); object↔reflection correspondences must satisfy mirror geometry; perspective/vanishing points must cohere. NDSS 2026 ("Light2Lie") extends to physically-grounded reflectance.
- **Robustness 5:** generators would need physical simulation in the loop to defeat this; none do.
- **Laundering 1–2:** geometry is low-frequency — survives thumbnails. *The* evidence for degraded images.
- **Explainability 5:** the system literally draws the lines on the image.
- **Limitation — recall:** requires visible shadows/reflections and tractable scenes; automation of these classically-manual checks is the hard engineering (MVP scopes to: light-direction consistency from shadow lines + face/object lighting-direction estimation; full reflection geometry post-MVP).
- **REC 9 / 8 / 8.**

---

## 3.7 Semantic Family

### 17. Semantic consistency (world-logic)
Hands/fingers topology, impossible object interactions, asymmetric earrings/glasses, melted backgrounds — the DARPA SemaFor class. Survives *any* compression (semantics are the last thing to go). Declining as generators improve, but the long tail of world-logic is vast. Most persuasive artifact class for human reviewers. **REC 8.**

### 18. Typography analysis
OCR text in image → lexicon/character-validity check; garbled glyphs remain a top diffusion failure (text rendering requires exact symbolic structure, the antithesis of diffusion's strengths). Narrow but high-precision and trivially explainable. **REC 7** — sub-component of module H.

---

## 3.8 Family-Level Synthesis

| Family | Verdict for ARGUS |
|---|---|
| Provenance + retrieval | **Strategic core** — only family that strengthens over time; Tier 0 + post-MVP module I |
| Compression history | **Always-on infrastructure** — powers triage; evidence about evidence |
| Realness modeling (embeddings, training-free probes) | **Statistical core** — the generator-agnostic detectors |
| Physics + semantics | **Robustness core** — what still works on a thumbnail in 2028 |
| Acquisition pipeline | Selective (statistical sensor noise yes; PRNU/CFA no) |
| Generative traces | **Perishable plugins only** — attribution metadata, never load-bearing |
