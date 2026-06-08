# PHASE 3 — Evidence Source Taxonomy

> *A complete catalog of authenticity evidence, scored on the four axes that actually predict production value. We score each source 0–10. Scores are engineering judgments for the 2026 generator landscape and are deliberately conservative about Tier-C durability.*

## 3.1 Scoring rubric

- **Future-robustness (FR):** expected usefulness against an *unseen, improved* generator (1 = obsolete on arrival; 10 = generator-independent).
- **Laundering survival (LS):** fraction of signal surviving recompression/resize/screenshot (1 = destroyed by any recompression; 10 = unaffected).
- **Explainability (EX):** how human-auditable the cue is (1 = black box; 10 = a layperson can verify it).
- **Compute cost (CC):** inverse-scored for readability — **higher = cheaper** (1 = expensive GPU/seconds; 10 = trivial CPU/ms).
- **Recommendation (REC):** overall priority for ARGUS, weighting FR and LS most heavily (these decide open-set survival), then EX, then CC. *REC ≈ 0.35·FR + 0.30·LS + 0.20·EX + 0.15·CC*, rounded with expert override.

## 3.2 Master evidence scorecard

| # | Evidence source | Tier | FR | LS | EX | CC | **REC** | Role in ARGUS |
|--:|-----------------|:----:|:--:|:--:|:--:|:--:|:------:|---------------|
| 1 | **C2PA / Content Credentials** | A | 10 | 6\* | 10 | 9 | **9.2** | Backbone (positive provenance) |
| 2 | **Retrieval / reverse-image / near-dup** | A | 9 | 8 | 9 | 5 | **8.4** | Backbone (external corroboration) |
| 3 | **Embedding similarity (CLIP/known-fake DB)** | A/C | 7 | 7 | 7 | 6 | **6.9** | Strong corroboration |
| 4 | **Metadata / EXIF / XMP consistency** | A | 8 | 2 | 9 | 10 | **6.6** | Positive when present; self-silencing |
| 5 | **Physics — lighting direction** | A | 8 | 7 | 8 | 4 | **7.2** | Durable anomaly backbone |
| 6 | **Physics — shadow geometry** | A | 8 | 7 | 9 | 4 | **7.3** | Durable, highly explainable |
| 7 | **Physics — reflection consistency** | A | 8 | 6 | 8 | 4 | **6.9** | Durable, situational |
| 8 | **Physics — perspective/vanishing point** | A | 7 | 8 | 8 | 5 | **7.1** | Durable geometric check |
| 9 | **Semantic consistency (anatomy, counts, world)** | B | 7 | 8 | 8 | 3 | **6.8** | Durable, VLM-assisted |
| 10 | **Typography / text legibility** | B | 6 | 7 | 9 | 6 | **6.8** | Durable today, decaying |
| 11 | **Texture stationarity / over-smoothing** | B/C | 5 | 5 | 6 | 6 | **5.3** | Corroboration |
| 12 | **Camera-pipeline / demosaicing coherence** | A | 7 | 4 | 6 | 6 | **5.8** | Positive capture evidence |
| 13 | **CFA (Color Filter Array) analysis** | A | 7 | 3 | 6 | 6 | **5.4** | Splice/synth localization |
| 14 | **Sensor noise / noise-floor consistency** | A | 6 | 3 | 6 | 6 | **5.1** | Tamper localization |
| 15 | **PRNU (sensor fingerprint)** | A | 6 | 2 | 5 | 4 | **4.4** | Strong w/ reference; rare in wild |
| 16 | **JPEG — double-quantization / ghosts** | A | 6 | 5 | 7 | 8 | **6.1** | Splice localization, cheap |
| 17 | **JPEG — grid/blocking discontinuity** | A | 6 | 5 | 7 | 9 | **6.2** | Cheap tamper map |
| 18 | **Frequency — azimuthal spectrum peaks** | C | 4 | 4 | 6 | 8 | **4.9** | Known-generator corroboration |
| 19 | **Wavelet residual statistics** | C | 4 | 4 | 5 | 7 | **4.5** | Corroboration |
| 20 | **Residual / SRM / CNN "fakeness" probe** | C | 3 | 4 | 3 | 6 | **3.7** | Fast but fragile; discounted on OOD |
| 21 | **Diffusion artifact detector** | C | 3 | 4 | 4 | 5 | **3.6** | Today-only; OOD-gated |
| 22 | **Latent inversion (reconstruction error)** | C | 4 | 5 | 5 | 2 | **4.0** | Research-grade; expensive |

\* C2PA *absence* is uninformative (LS low for the absence case); C2PA *presence/validity* is laundering-proof (cryptographic). The LS=6 reflects that platforms often strip manifests in transit, lowering *coverage*, not *trustworthiness when present*.

## 3.3 Per-source deep evaluation

Condensed analysis of each, emphasizing the failure modes that determine reliability weighting.

**1. C2PA / Content Credentials.** Cryptographically signed capture-and-edit history. *Robustness:* maximal — forgery requires a private key. *Laundering:* manifests are frequently stripped by platforms (low coverage) but, when present and valid, are tamper-evident. *Explainability:* perfect (a signed claim chain). *Verdict:* highest-value positive signal; the strategic bet of the entire provenance ecosystem. ARGUS treats a *valid* manifest as near-dispositive positive evidence, and a *broken/invalid* one as strong negative.

**2. Retrieval / reverse-image / near-duplicate.** Find earlier or authoritative copies. *Robustness:* generator-independent — you cannot fake the past. *Laundering:* perceptual hashing + embedding ANN survives recompression well. *Explainability:* a human can click the match. *Verdict:* the strongest *open-set* signal; corpus coverage is the only limiter (→ abstention, not error). Backbone.

**3. Embedding similarity.** CLIP/DINOv2 neighborhood + a curated known-AI / stock-gallery / known-fake embedding index. *Robustness:* medium-high (semantic embeddings are laundering-robust). *Verdict:* strong corroboration; doubles as the retrieval index.

**4. Metadata / EXIF / XMP.** Internal-consistency checks (camera↔lens↔resolution↔timestamps↔maker-notes↔embedded thumbnail mismatch). *Robustness:* high *when present*; *Laundering:* near-zero (stripped immediately). *Verdict:* valuable positive evidence on pristine files; correctly self-silences on laundered ones. Never treat absence as guilt.

**5–8. Physics consistency (lighting, shadows, reflections, perspective).** Estimate dominant light direction from shading/specular cues and test global consistency; verify shadow-to-object geometry and a common light source; check planar reflections; recover vanishing points and test perspective coherence. *Robustness:* high — optics/geometry are generator-loss-independent and improve slowly in generators. *Laundering:* robust (geometry survives compression). *Explainability:* excellent (you can *draw the inconsistent shadow*). *Verdict:* the durable, explainable backbone of anomaly detection. Hardest to engineer well → MVP uses a strong subset (lighting + shadow + a VLM cross-check).

**9. Semantic / world-knowledge consistency.** Anatomy (hands, teeth, ears, jewelry continuity), object counts, text-scene plausibility, physical impossibility. *Robustness:* medium-high; the last things generators fix. *Verdict:* VLM-assisted, highly explainable, durable.

**10. Typography.** Rendered text in synthetic images is often subtly malformed or non-uniform. *Robustness:* decaying (modern generators improving fast) but currently strong & extremely explainable. *Verdict:* keep, expect decay.

**11. Texture stationarity / over-smoothing.** Synthetic regions are often too locally smooth or too stationary. *Robustness:* low-medium (generators target this). *Verdict:* corroboration only.

**12–15. Camera-pipeline / CFA / Sensor noise / PRNU.** The physical-capture-presence family. CFA demosaicing leaves predictable inter-pixel correlations; sensor noise has a characteristic floor; PRNU is a near-unique multiplicative fingerprint. *Robustness:* high *in principle* (synthesis must fabricate imaging physics), but *Laundering:* fragile — recompression/resize attenuates high-frequency sensor signal severely (PRNU worst). *Verdict:* powerful positive/localization evidence on pristine images; reliability must collapse to near-zero on laundered inputs (the self-silencing principle). PRNU is high-value only with a reference device, so it is *opportunistic*, not core.

**16–17. JPEG forensics (double-quantization, ghosts, grid discontinuity).** Detect inconsistent compression history → splice/insertion localization; cheap and explainable (heatmaps). *Robustness:* medium; *Laundering:* re-saving normalizes some traces but introduces others. *Verdict:* cheap, explainable tamper localization — high value/cost ratio. Keep in MVP.

**18–19. Frequency / wavelet.** Azimuthal power-spectrum peaks from up-sampling; wavelet residual stats. *Robustness:* low — the canonical arms-race-fragile fingerprint. *Laundering:* resampling smears the peaks. *Verdict:* cheap corroboration on known generators; OOD-gated hard.

**20–21. Residual-CNN / diffusion-artifact detectors.** The incumbent paradigm in module form. *Robustness:* lowest (overfits generators). *Verdict:* included as one fast, *low-reliability-on-OOD* opinion — never the backbone. This is the demotion at the heart of ARGUS.

**22. Latent inversion.** Invert the image into a generator's latent space; low reconstruction error ⇒ likely produced by a similar model. *Robustness:* medium but model-family-bound; *Cost:* high (optimization per image). *Verdict:* research-grade; out of MVP, candidate for later.

## 3.4 Strategic reading of the scorecard

- The **top of the table is dominated by Tier-A** (provenance, retrieval, physics) — exactly the inverse of the incumbent stack. This is the intended design signal.
- The **bottom is the entire classifier/fingerprint paradigm** — present but demoted.
- **Cheap-and-durable winners** to prioritize in the MVP: C2PA, metadata consistency, JPEG forensics, retrieval/embedding, lighting+shadow physics (via a VLM), frequency (as cheap corroboration).
- **Expensive-but-durable** to schedule later: full PRNU with reference, latent inversion, advanced multi-view physics.

\newpage
