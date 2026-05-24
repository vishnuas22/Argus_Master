"# Multimodal Deepfake Detection — Master Plan (v1.3)

> **Status:** Plan only — no code yet. Single source of truth for implementation.
> **Last updated:** 2026-02 | **Supersedes:** v1.2
> **Primary runtime targets:** `cloud_lite` (Emergent CPU container) + `mac_full` (Apple M1 Max 32 GB) + `cuda_full` (RTX 3050 4 GB, when host has it)
> **First-finish scope (M0→M3):** image modality, all 4 tiers, `cloud_lite` end-to-end + `mac_full`/`cuda_full` enhancements

---

## 0. Change Log

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-01 | Initial COEF master plan |
| 1.1 | 2026-02 | Dual-profile (CLOUD_LITE/LOCAL_FULL), TTA, patch voting, DIRE→P1, CLIP zero-shot, confidence fix, SHA pinning, prosody, ArcFace identity, active learning, rule-based narrator fallback, AGENTS.md compliance |
| 1.2 | 2026-02 | Tri-profile + device auto-detect, Tier-0 Provenance Gate, Tier-2 Embedding Retrieval, Tier-3 VLM-as-judge, Adaptive Fusion (uniform→LR→GBDT), weak-label scraping for refDB, SHA256 self-leak guard |
| **1.3** | **2026-02** | **First-finish narrowed to M0→M3 image-only (faster validation). Added: (a) Tier-2.5 Reverse-Image Search via SerpAPI as a near-deterministic signal, (b) Cold-start calibration on refDB (ship calibrated, not \"uncalibrated\"), (c) Content-type-aware abstention thresholds (CLIP zero-shot routes to 6 type-specific gates), (d) Compression/container forensics layer (PNG chunks, double-JPEG, codec fingerprints), (e) Cross-modal multiplicative fusion bonus, (f) Hard-negative memory (corrected jobs feed back into refDB instantly, no retraining), (g) Developer/debug mode UI for live signal inspection + threshold override, (h) ECE-on-refDB live health metric, (i) WavLM-base-plus default audio embedder (MIT, replaces CC-BY-NC CLAP), (j) HF Inference API fallback for heavy models on `cloud_lite`. Fixed: explicit \"95% on non-abstained\" framing as headline KPI. Reverse-search invocation gated by extremity/agreement to conserve SerpAPI quota.** |

---

## 1. Executive Summary

### 1.1 Product
A forensic-grade web application that ingests an **image** (Phase 1 first-finish), **audio** or **video** (Phase 1 follow-up), or **text** (Phase 1.5), and returns:
- A calibrated probability the media is AI-generated
- One of three verdicts: `AI-GENERATED` | `REAL` | `INCONCLUSIVE`
- Per-signal visual XAI evidence
- A plain-English narrative
- A downloadable JSON report

### 1.2 The real production-grade KPI
> **≥95 % accuracy on *non-abstained* uploads, with a tunable abstention rate.**

This is the framing that beats \"75–80 %.\" A model that abstains on 25 % of edge cases but is 96 % correct on the rest is strictly better than a model that is 78 % across the board — because the user gets a *trustworthy* answer when they get one. The abstention rate is itself a knob (`ABSTAIN_AGREE`, `ABSTAIN_HIGH`, `ABSTAIN_LOW`) tunable per deployment.

### 1.3 Why prior single-model attempts stalled at 75–80 %
Any single learned detector overfits to its training distribution. In-distribution AUROC > 95 % collapses to 50–60 % on unseen generators (SD3, Flux, ElevenLabs v2, Sora, HeyGen). Fine-tuning shifts the distribution; it does not solve generalization.

### 1.4 Strategy — Calibrated Orthogonal Evidence Fusion (COEF), 5 tiers

```
Tier 0:   Provenance Gate          → short-circuit if C2PA / SynthID / SD watermark hit
Tier 1:   Forensic + Learned       → k orthogonal detectors per modality (the core)
Tier 2:   Embedding Retrieval      → CLIP/DINOv2/WavLM k-NN against curated ref DB
Tier 2.5: Reverse Image Search     → SerpAPI lookup → priors (pre-2022 hits = REAL strong)   [NEW v1.3]
Tier 3:   VLM Tiebreaker           → Gemini 3 Flash vision — only on uncertain cases
                  ↓
Adaptive Fusion (uniform → L2-LR → LightGBM)
                  ↓
Content-type-aware Abstention Gate    [NEW v1.3]
                  ↓
XAI Renderer + Calibrated Narrative
```

Each tier is **orthogonal** and **training-free** (Tier 0, 2, 2.5, 3) or **pretrained-only** (Tier 1). **No fine-tuning. No upfront training budget.**

### 1.5 Revised OOD AUROC targets (with all 5 tiers, image modality first-finish)

| | Single-model baseline | `cloud_lite` v1.3 | `mac_full` v1.3 | `cuda_full` v1.3 |
|---|---|---|---|---|
| Raw AUROC | 65–78 % | **86–91 %** | **92–96 %** | **91–95 %** |
| Accuracy on non-abstained slice | — | **≥95 %** | **≥97 %** | **≥96 %** |
| Expected abstention rate | — | 18–25 % | 10–15 % | 12–17 % |

`mac_full` slightly edges `cuda_full` because 32 GB unified memory permits ensemble parallelism that 4 GB VRAM cannot.

---

## 2. Scope

### 2.1 First-finish scope (M0→M3, image modality only)
Everything in this list MUST work end-to-end before declaring first-finish success:

- Upload **image** → verdict + confidence + agreement + XAI + JSON report
- Three verdicts: `AI-GENERATED` | `REAL` | `INCONCLUSIVE`
- **Tier 0 Provenance Gate**: C2PA validate + SD invisible-watermark + SynthID (guarded import)
- **Tier 1 Forensic + Learned** (image): `prithiv`, `frequency`, `clip0`, `metadata`, `compression_forensics` on all profiles; `npr`, `ufd`, `dire` on `mac_full`/`cuda_full`
- **Tier 2 Embedding Retrieval**: CLIP-B/32 → FAISS k=15 against curated refDB (1500 real + 1500 AI images, built once via scraper)
- **Tier 2.5 Reverse Image Search**: SerpAPI Google reverse-image lookup, gated by uncertainty
- **Tier 3 VLM Tiebreaker**: Gemini 3 Flash vision, gated by `extremity<0.25 OR agreement<0.63`
- **Adaptive fusion** (uniform mode at launch; LR mode auto-promotes at n≥100)
- **Cold-start calibration on refDB** — Platt scaling fit on refDB-vs-known-AI before first user upload
- **Content-type-aware abstention** — CLIP zero-shot routes to 1 of 6 type-specific threshold sets
- **Cross-modal multiplicative bonus** for super-additive confidence when ≥3 tiers agree
- **Hard-negative memory** — corrected verdicts append embedding to refDB hard-negatives partition
- **Developer/debug mode** — UI toggle exposes raw per-signal scores, fusion weights, threshold overrides
- **ECE-on-refDB** metric in `/api/health`
- Per-signal XAI: GradCAM, FFT radial, retrieval neighbors, VLM rationale, reverse-search top-5
- Gemini-authored narrative + rule-based fallback
- Profile auto-detection (`cloud_lite` / `mac_full` / `cuda_full`)
- Single-user local app (no auth)
- Download JSON report
- Job history (last 20)

### 2.2 Phase 1 follow-up (after first finish, before \"v1 done\")
- M4: Audio modality (`w2v2df`, `spectral`, `prosody`, `aasist3`, refDB retrieval via **WavLM-base-plus** [MIT, replaces CC-BY-NC CLAP])
- M5: Video modality (frames, faces, img_ens, flicker, audio, syncnet, blink, identity)
- M6: Polish, E2E with `testing_agent_v3`, M11 docker compose

### 2.3 Phase 1.5
- **Text deepfake detection** via Binoculars-style perplexity-ratio + GLTR (free, lightweight)
- `PARTIALLY-MODIFIED` verdict once real INCONCLUSIVE samples accumulate
- PDF report export
- Active learning UI (\"was this real or AI?\" button)

### 2.4 Explicit non-goals
- Training or fine-tuning any model
- User authentication / multi-tenancy
- Real-time streaming detection
- Mobile app
- Adversarial-robust detection vs. targeted attacks (defended attackers)
- PDF export (Phase 1.5)

---

## 3. Core Strategy — 5-Tier COEF (Detailed)

### 3.1 Tier 0 — Provenance Gate

A pre-fusion short-circuit. If any check returns a hit, the verdict is fixed at **p=0.99** and the ensemble result is shown as \"secondary evidence.\"

| Check | Library | Hit verdict |
|---|---|---|
| Valid C2PA manifest with active producer signature | `c2pa-python` | REAL, p=0.99 |
| Google **SynthID** watermark (image) | `synthid-text` (guarded) | AI, p=0.99 |
| Stable Diffusion invisible watermark | `invisible-watermark` | AI, p=0.99 |
| Meta IM watermark (when public detector ships) | guarded import | AI, p=0.99 |

**Output schema:** every result includes
```json
\"provenance\": {\"hit\": true|false, \"source\": \"c2pa|synthid|sd_wm|meta_wm|none\", \"details\": {...}}
```

When `hit=true`, the ensemble still runs in background for telemetry, but the headline verdict is provenance-based.

### 3.2 Tier 1 — Forensic + Learned Detectors

Multiple **pretrained** detectors per modality + training-free forensic signals. Each catches a different failure mode. Image modality detail in §5.

### 3.3 Tier 2 — Embedding Retrieval

A frozen embedding model produces a 512–768-d vector for the upload. **k-NN (k=15)** against a curated reference DB of known-real and known-AI samples.

```python
# Distance-weighted retrieval score
p_retrieval = sum(w_i * y_i) / sum(w_i)    where  w_i = 1 / (1 + d_i)
```

- **ANN index:** FAISS-CPU (works on all three profiles equally; ~50 ms / query at 5k entries)
- **Embedder per modality:** CLIP-B/32 (image), DINOv2-base (video), **WavLM-base-plus** (audio — MIT-licensed default; CLAP is opt-in)
- **Why it generalizes:** new generators' outputs cluster near older generators' outputs in embedding space because they share denoising-induced texture statistics. Similarity-based — does not require having seen that exact generator.
- **Self-leak guard:** SHA256 of upload checked against refDB at query time; matched entry excluded from k-NN; UI warns \"this exact file is in the reference DB; retrieval signal disabled.\"
- **Hard-negative partition (NEW v1.3):** corrected verdicts append the upload's embedding + correction label to `refdb/image_ai_hard.npy` (or `_real_hard.npy`). Future queries benefit immediately, no retraining.

### 3.4 Tier 2.5 — Reverse Image Search (NEW v1.3)

The single biggest free accuracy lever the previous plan was missing.

**How it works.** When invoked, the upload is sent to **SerpAPI** `google_reverse_image` endpoint (or `bing_visual_search`). The response is parsed for:

| Signal | Logic | Output |
|---|---|---|
| Earliest indexed date | If earliest hit pre-dates known AI image generators for that style (e.g., pre-2022 photo-realistic) → strong REAL prior | `p_reverse = 0.05` (very-real) |
| News/journalism domain hits | If image appears on Reuters/AP/BBC/national news → strong REAL prior | `p_reverse = 0.10` |
| Civitai / Lexica / OpenArt / Midjourney gallery hits | Strong AI prior | `p_reverse = 0.95` |
| Stock-photo agency hits (Getty, Shutterstock with consistent metadata) | Strong REAL prior | `p_reverse = 0.15` |
| AI-art subreddits as top hits (r/StableDiffusion, r/midjourney) | Strong AI prior | `p_reverse = 0.92` |
| No hits / pages return empty | Neutral, no signal | signal absent from fusion vector |

**Invocation gate (to conserve SerpAPI free quota — 100 searches/month default):**
```
extremity < 0.30 OR agreement < 0.70 OR (p_retrieval - 0.5).abs() < 0.15
```
i.e. only when other tiers are uncertain. Expected hit rate: **15–25 % of jobs**.

**Caching:** SHA256-keyed cache of SerpAPI responses (24 h TTL) under `storage/cache/serpapi/`. Re-uploads of the same image cost zero quota.

**Fallback:** if `SERPAPI_KEY` missing/exhausted → signal dropped silently from fusion vector (system stays functional).

**Library:** `requests` direct call (SerpAPI has no required SDK); response parsed via simple JSON traversal. Per-call timeout 8 s.

### 3.5 Tier 3 — VLM-as-Judge Tiebreaker

Gemini 3 Flash vision (via `emergentintegrations` library) prompted as a forensic analyst to inspect the image and rate it 0–1, citing specific defects (see Appendix B prompt).

**Invocation gate (to conserve free quota — Gemini Flash ~1500 free/day):**
```
extremity < 0.25 OR agreement < 0.63
```
Expected hit rate: **20–30 % of jobs**.

**Output schema** added to fusion vector as `p_vlm`; `vlm_invoked: bool` recorded; rationale feeds narrative (saves a second Gemini call).

**Why it's the single biggest accuracy lever (after reverse search):** VLMs catch *semantic* impossibilities (warped hands, impossible reflections, text gibberish, anatomy errors, lighting inconsistencies) that no forensic or learned model sees. On the uncertain 20–30 % slice it routinely flips verdicts toward correct.

**Fallback:** if `GEMINI_API_KEY` missing or rate-limited → gate effectively returns to 4-tier COEF; VLM signal dropped from fusion vector.

### 3.6 Cross-modal multiplicative bonus (NEW v1.3)

When **≥3 independent tiers** agree (e.g., `p_retrieval>0.7` AND `p_reverse>0.7` AND `p_vlm>0.7`), apply a super-additive confidence bonus capped at +0.10:

```python
agreement_count = sum(1 for p in [p_retrieval, p_reverse, p_vlm, fused_p_tier1] if (p > 0.7 or p < 0.3))
if agreement_count >= 3:
    confidence_bonus = min(0.10, 0.03 * (agreement_count - 2))
    confidence = min(1.0, confidence + confidence_bonus)
```

This captures the intuition that three independent observers agreeing is genuinely stronger evidence than the linear sum of their scores.

---

## 4. Content-Type-Aware Abstention (NEW v1.3)

A single global `ABSTAIN_HIGH=0.75` fails because content types have wildly different priors:
- A selfie missing EXIF is normal (phones strip EXIF when shared)
- A landscape photo missing EXIF is suspicious
- A meme/screenshot has no useful EXIF expectations
- A document scan has different forensics entirely

### 4.1 Routing
At preprocess, CLIP zero-shot classifies the upload into **one of 6 content types**:
1. `selfie_portrait` — single human face, center-framed
2. `landscape_scene` — outdoor scene, no dominant face
3. `object_product` — single object, product photography style
4. `meme_screenshot` — text overlay, low-res, compression artifacts
5. `document_scan` — flat, text-heavy, scan/photo of paper
6. `artwork_illustration` — drawn/illustrated style (not photorealistic)

### 4.2 Type-specific thresholds (initial defaults, auto-tuned from refDB)
```python
TYPE_THRESHOLDS = {
    \"selfie_portrait\":      {\"high\": 0.78, \"low\": 0.22, \"agree\": 0.55},
    \"landscape_scene\":      {\"high\": 0.72, \"low\": 0.28, \"agree\": 0.55},
    \"object_product\":       {\"high\": 0.75, \"low\": 0.25, \"agree\": 0.55},
    \"meme_screenshot\":      {\"high\": 0.82, \"low\": 0.18, \"agree\": 0.50},  # noisy by nature
    \"document_scan\":        {\"high\": 0.80, \"low\": 0.20, \"agree\": 0.55},
    \"artwork_illustration\": {\"high\": 0.85, \"low\": 0.15, \"agree\": 0.50},  # AI art common
}
```
After 200+ labelled uploads (auto-accumulated via refDB + corrections), per-type thresholds are re-tuned to hit target precision via `scripts/tune_thresholds.py`.

---

## 5. Image Modality Pipeline (FULL DETAIL — first-finish core)

### 5.1 Preprocessing
1. Decode → RGB tensor; record original EXIF + SHA256
2. Resize to 256×256 for detectors; keep original for metadata, VLM, reverse search
3. Face detection: RetinaFace on `mac_full`/`cuda_full`; cv2 Haar fallback on `cloud_lite`
4. **Content-type classification** via CLIP zero-shot (6-way softmax) → routes abstention thresholds
5. **TTA** — 3 views per learned detector: original, h-flip, JPEG-recompress(q=85). Mean of scores.
6. **Patch voting** (`mac_full`/`cuda_full` only) — for images >512², run on 4 corner patches + center; aggregate mean+max.

### 5.2 Tier-0 Provenance Gate (image)
```
1. C2PA validate (c2pa-python)
2. SD invisible watermark (invisible-watermark)
3. SynthID detect (guarded import)
4. Meta IM watermark (guarded import)
```
Short-circuit on hit.

### 5.3 Tier-1 signals (image)

| # | Name | Source | `cloud_lite` | `mac_full` | `cuda_full` | Output |
|---|---|---|---|---|---|---|
| 1 | `img.prithiv` | `prithivMLmods/deepfake-detector-model-v1` (Apache-2.0) | ✓ ONNX-INT8 | ✓ fp32 MPS | ✓ fp16 CUDA | p_fake + logits |
| 2 | `img.freq` | NumPy/SciPy FFT radial + DCT skew/kurtosis | ✓ | ✓ | ✓ | p_fake + radial PNG |
| 3 | `img.clip0` | `openai/clip-vit-base-patch32` zero-shot | ✓ | ✓ | ✓ | p_fake (cosine softmax) |
| 4 | `img.meta` | exifread + c2pa partial-anomaly scoring | ✓ | ✓ | ✓ | flag + table |
| 5 | **`img.compression`** (NEW v1.3) | PNG chunk fingerprint + double-JPEG detection + codec signature | ✓ | ✓ | ✓ | p_fake + fingerprint dict |
| 6 | `img.npr` | `chuangchuangtan/NPR-DeepfakeDetection` (MIT) | — | ✓ | ✓ | p_fake + GradCAM |
| 7 | `img.ufd` | `Yuheng-Li/UniversalFakeDetect` (Apache-2.0) | — | ✓ | ✓ | p_fake + attention |
| 8 | `img.dire` | DIRE — ADM reconstruction error | — | ✓ (CPU) | ✓ (CUDA) | p_fake + residual PNG |

### 5.4 Compression / container forensics (`img.compression`) — NEW v1.3

A training-free, deterministic signal that catches generator-specific encoder fingerprints.

**PNG checks:**
- `bit_depth`, `color_type`, `compression_method`, `filter_method`, `interlace_method`, presence of `tEXt`/`zTXt` chunks
- SDXL/MJ/Flux often emit PNGs with `bit_depth=8`, `color_type=2`, no `tEXt`, IDAT zlib-level=9 → score +0.4
- Camera-saved or screenshot PNGs frequently have specific chunk ordering (e.g., `pHYs` chunk present) → score -0.3
- ICC profile presence/absence vs. content type

**JPEG checks:**
- Double-JPEG detection via DCT histogram periodicity (when AI-saved-as-JPEG: typically single-quality, no double-JPEG signature)
- Quantization table fingerprint (camera quant tables differ from Pillow defaults)
- Markers: presence of EXIF (`APP1`), JFIF (`APP0`), Photoshop IRB (`APP13`)
- Camera-real JPEGs almost always have EXIF + camera-specific quant tables

**WebP / AVIF / HEIC:** container metadata sanity, encoder string parsing.

**Output:** `p_fake` ∈ [0,1] + a `fingerprint` dict shown in metadata panel.

**Why this matters:** Many modern generators leave deterministic file-format signatures that no model in the original v1.2 plan extracts. Catches ~5–10 % additional cases for free.

### 5.5 Tier-2 retrieval (`img.retrieval`)
- Embedder: CLIP-B/32 (shared with `img.clip0` — same model loaded once)
- Index: `storage/refdb/image_real.index` + `image_ai.index` (FAISS, IndexFlatIP)
- Hard-negatives index: `image_real_hard.index` + `image_ai_hard.index` (smaller, queried jointly)
- k=15 across union of indexes; distance-weighted score (§3.3)
- Returns top-5 neighbors (id, label, distance, source thumbnail) for XAI panel

### 5.6 Tier-2.5 reverse search (`img.reverse`)
- SerpAPI `google_reverse_image` → JSON parse
- Cache 24 h on SHA256 of upload
- Output: `p_fake` ∈ {0.05, 0.10, 0.15, 0.50 (none), 0.92, 0.95} per §3.4 table + top-5 hit URLs/dates for XAI

### 5.7 Tier-3 VLM (`img.vlm`)
- Gemini 3 Flash vision via `emergentintegrations`
- Prompt: Appendix B
- Output: `{p_ai: float, defects: [str], rationale: str}` strict JSON

### 5.8 Fusion input vector (image)

```
cloud_lite (always present):
  [p_prithiv, p_freq, p_clip0, p_meta, p_compression, p_retrieval,
   face_present, exif_missing, c2pa_missing, content_type_idx,
   p_reverse?, p_vlm?]

mac_full / cuda_full (above + when applicable):
  + [p_npr, p_ufd, p_dire, patch_disagreement, tta_std]
```

Missing signals (e.g., `p_vlm` when not invoked) → fusion model trained with **mean-imputation** on a padded fixed-width vector. Selector at runtime knows which slots to impute.

### 5.9 XAI artifacts produced per image job
- `heatmap.png` — GradCAM overlay (NPR backbone on mac/cuda; CLIP attention on cloud)
- `fft.png` — FFT radial profile RadarChart data + reference bands
- `retrieval_neighbors.json` — top-5 with thumbnail URLs
- `reverse_hits.json` — top-5 SerpAPI hits with date + domain + URL
- `vlm_rationale.json` — Gemini bulleted defects (when invoked)
- `compression_fingerprint.json` — full PNG/JPEG/WebP forensic dict
- `narrative.txt` — final 3–5 sentence narrative
- `report.json` — full result payload

---

## 6. Cold-Start Calibration on refDB (NEW v1.3)

The original plan shipped with `temperature = 1.0` and an \"Uncalibrated\" badge until 100 user labels accumulate. **This is unnecessary.** We have the reference DB — use it.

### 6.1 At install (one-time, in `build_reference_db.py`)
1. Build refDB (1500 real + 1500 AI image embeddings + corresponding raw signal scores via running every Tier-1 detector on each sample)
2. **Hold out 20 %** as a calibration fold
3. Fit per-signal **Platt scaling** on the 80 % train fold:
   ```python
   # For each signal i:
   A_i, B_i = LogisticRegression().fit(s_i.reshape(-1,1), y).coef_, intercept_
   p_i = sigmoid(A_i * s_i + B_i)
   ```
4. Save to `backend/fusion/platt.json`
5. Evaluate per-signal AUROC + ECE on held-out 20 % → save to `calibration/report.md`

### 6.2 At runtime
- Result schema reports `\"calibration\": \"platt_refdb\"` (was `\"cold_start\"`)
- Once `n_user_labels ≥ 100`, **re-fit Platt on (refDB ∪ user-labels)** weighted by recency → `\"calibration\": \"platt_blended\"`
- Once `n_user_labels ≥ 500`, switch to isotonic regression per signal → `\"calibration\": \"isotonic\"`

### 6.3 Live health metric
`GET /api/health` returns:
```json
{
  \"calibration\": \"platt_refdb\",
  \"ece_refdb_holdout\": 0.043,
  \"auroc_refdb_holdout\": 0.91,
  \"refdb_size\": {\"image_real\": 1500, \"image_ai\": 1500},
  \"fusion_mode\": \"uniform\",
  \"n_user_labels\": 0
}
```
If `ece_refdb_holdout > 0.10` (calibration drift), surface a warning in `/health` and UI.

---

## 7. Adaptive Fusion (locked from v1.2, with refDB warm start)

### 7.1 Fusion selector
```python
def pick_fusion_model(n_user_labels: int) -> str:
    if n_user_labels < 100: return \"uniform\"       # cold start (calibrated via refDB)
    if n_user_labels < 500: return \"lr_l2\"
    return \"gbdt\"
```

### 7.2 Modes

| Mode | Algorithm | When | Per-signal calibration |
|---|---|---|---|
| `uniform` | Equal weights + small bias for `metadata_missing` | n < 100 | Platt-refDB (§6) |
| `lr_l2` | L2 LR (`C=0.3`) | 100 ≤ n < 500 | Platt-blended |
| `gbdt` | LightGBM (num_leaves=15, n_estimators=100, min_child_samples=10) | n ≥ 500 | Isotonic |

### 7.3 Weight files (hot-reload on file-mtime change)
- `backend/fusion/weights_uniform.json` (ships with repo)
- `backend/fusion/platt.json` (generated by refDB build)
- `backend/fusion/weights_lr.json` (generated by `run_calibration.py`)
- `backend/fusion/weights_gbdt.txt` (LightGBM)
- `backend/fusion/iso_*.pkl` (isotonic per signal)

### 7.4 Confidence formula (locked)
```
agreement = 1 − 2 · std([p_1, …, p_k])
extremity = abs(2 · p_final − 1)
confidence = sqrt(agreement · extremity)
+ cross-modal multiplicative bonus (§3.6)
```

---

## 8. Abstention Logic (content-type-aware)

```python
thr = TYPE_THRESHOLDS[content_type]  # §4.2
if p_final >= thr[\"high\"] and agreement >= thr[\"agree\"]:
    verdict = \"AI-GENERATED\"
elif p_final <= thr[\"low\"] and agreement >= thr[\"agree\"]:
    verdict = \"REAL\"
else:
    verdict = \"INCONCLUSIVE\"
```

**Provenance Tier-0 bypasses abstention** — if the gate fires, verdict is fixed.

**INCONCLUSIVE UX copy:** *\"Forensic signals conflict on this sample. See per-signal evidence below — at least one strong indicator disagrees with the majority.\"*

---

## 9. Developer / Debug Mode (NEW v1.3)

Toggle in UI header (`Ctrl/Cmd + D` or button). When enabled, the result page reveals:

| Panel | Content |
|---|---|
| **Raw signal table** | Every signal: raw score `s_i`, calibrated `p_i`, fusion weight `w_i`, contribution `w_i*p_i` |
| **Per-stage durations** | Same as `result.durations_ms` — table view |
| **Threshold overrides** | Live sliders for `ABSTAIN_HIGH`, `ABSTAIN_LOW`, `ABSTAIN_AGREE` — re-renders verdict client-side without re-running detectors |
| **Fusion vector dump** | The full input vector + zero-imputed mask |
| **VLM gate state** | Why VLM was/wasn't invoked (which threshold tripped) |
| **Reverse-search raw** | Full SerpAPI response (parsed) |
| **refDB neighbors raw** | All k=15 with distances (not just top-5) |

**Why this matters for you specifically:** your stated pain is \"false predictions most of the time.\" This panel lets you inspect *exactly* which signal is wrong on each failure case — within minutes — and tune thresholds or disable individual detectors without redeploying. Closes the iteration loop tighter than anything else.

Backend support: a `?debug=1` query param on `/result` returns extra fields under `result.debug = {...}`. Otherwise, omitted.

---

## 10. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│   FRONTEND (React 19 + craco, Tailwind, shadcn/ui, Recharts)│
│   /   /job/:id   /about                                     │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTPS (REACT_APP_BACKEND_URL)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│   API (FastAPI async, all routes /api/*)                    │
│   POST /analyze   GET /jobs/{id}   GET /jobs/{id}/result    │
│   GET /jobs/{id}/assets/{name}    GET /history              │
│   GET /health   GET /modalities   GET /profile              │
│   GET /refdb/stats   POST /jobs/{id}/correct                │
└──────────────────────┬──────────────────────────────────────┘
                       │
            ┌──────────┼─────────────┐
            ▼          ▼             ▼
       ┌─────────┐ ┌──────────┐ ┌──────────────┐
       │ MongoDB │ │ Job      │ │ Detector     │
       │ jobs,   │ │ Runner   │ │ Registry     │
       │ results,│ │ (FastAPI │ │ (device-aware│
       │ labels  │ │  BG)     │ │  + LRU + SHA)│
       └─────────┘ └────┬─────┘ └──────┬───────┘
                        │              │
              ┌─────────┼──────────────┘
              ▼         ▼          ▼          ▼          ▼
        ┌─────────┐┌────────┐ ┌────────┐┌─────────┐┌─────────┐
        │ Tier-0  ││ Tier-1 │ │ Tier-2 ││ Tier-2.5││ Tier-3  │
        │ Provena ││ Detect │ │ Retri- ││ Reverse ││ VLM     │
        │ -nce    ││ ors    │ │ eval   ││ Search  ││ Judge   │
        └────┬────┘└───┬────┘ └───┬────┘└────┬────┘└────┬────┘
             │ hit?    │          │          │ (gated)  │ (gated)
             │ short-  │          │          │          │
             │ circuit │          │          │          │
             └─────────►├◄────────┴──────────┴──────────┘
                       ▼
              ┌──────────────────────┐
              │ Platt / Temperature  │
              │ per-signal calib     │
              └──────────┬───────────┘
                        ▼
              ┌──────────────────────┐
              │ Adaptive Fusion      │
              │ (uniform/LR/GBDT)    │
              └──────────┬───────────┘
                        ▼
              ┌──────────────────────┐
              │ Content-type Gate    │
              │ (6 type thresholds)  │
              └──────────┬───────────┘
                        ▼
              ┌──────────────────────┐
              │ XAI Renderer +       │
              │ Gemini Narrator      │
              └──────────────────────┘
```

**Process model:** single FastAPI process. Uploads under `/app/backend/storage/jobs/{job_id}/`. Job state in MongoDB. Models loaded lazily, LRU-evicted when VRAM/RAM pressure exceeds profile budget. No Celery/Redis in MVP.

**Device auto-detection at startup:**
```python
def detect_profile() -> str:
    forced = os.getenv(\"DETECTOR_PROFILE\", \"auto\")
    if forced != \"auto\": return forced
    if torch.cuda.is_available(): return \"cuda_full\"
    if torch.backends.mps.is_available(): return \"mac_full\"
    return \"cloud_lite\"
```

---

## 11. Model Registry & Pinning

All weights cached under `/app/backend/storage/models/` (env `HF_HOME`). First-run download: `cloud_lite` ≈ 0.9 GB; `mac_full`/`cuda_full` ≈ 4.5 GB.

`backend/detectors/registry.py` keeps a frozen table with SHA pins, profile gating, and device preference. Schema:

```python
@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    sha: str                  # commit hash for pinning (filled after registry verify)
    license: str
    size_mb: int
    profile_in: tuple[str, ...]
    device_pref: dict[str, str]   # profile -> device
    fallback_repo: str | None = None

MODELS: dict[str, ModelSpec] = {
    \"img.prithiv\":   ModelSpec(\"img.prithiv\",  \"prithivMLmods/deepfake-detector-model-v1\", \"\", \"Apache-2.0\", 350,  (\"cloud\",\"mac\",\"cuda\"), {\"cloud\":\"cpu\",\"mac\":\"mps\",\"cuda\":\"cuda\"}),
    \"img.clip0\":     ModelSpec(\"img.clip0\",    \"openai/clip-vit-base-patch32\",             \"\", \"MIT\",        605,  (\"cloud\",\"mac\",\"cuda\"), {\"cloud\":\"cpu\",\"mac\":\"mps\",\"cuda\":\"cuda\"}),
    \"img.npr\":       ModelSpec(\"img.npr\",      \"chuangchuangtan/NPR-DeepfakeDetection\",    \"\", \"MIT\",        48,   (\"mac\",\"cuda\"),         {\"mac\":\"mps\",\"cuda\":\"cuda\"}),
    \"img.ufd\":       ModelSpec(\"img.ufd\",      \"Yuheng-Li/UniversalFakeDetect\",            \"\", \"Apache-2.0\", 1600, (\"mac\",\"cuda\"),         {\"mac\":\"mps\",\"cuda\":\"cuda\"}),
    \"img.dire\":      ModelSpec(\"img.dire\",     \"Zhendong-Wang/DIRE\",                       \"\", \"MIT\",        1100, (\"mac\",\"cuda\"),         {\"mac\":\"cpu\",\"cuda\":\"cuda\"}),
    \"face.retina\":   ModelSpec(\"face.retina\",  \"insightface/buffalo_l\",                    \"\", \"MIT\",        280,  (\"mac\",\"cuda\"),         {\"mac\":\"mps\",\"cuda\":\"cuda\"}),
    \"embed.clip\":    ModelSpec(\"embed.clip\",   \"openai/clip-vit-base-patch32\",             \"\", \"MIT\",        605,  (\"cloud\",\"mac\",\"cuda\"), {\"cloud\":\"cpu\",\"mac\":\"mps\",\"cuda\":\"cuda\"}),  # shared
    # Phase-1 follow-up:
    \"aud.w2v2df\":    ModelSpec(\"aud.w2v2df\",   \"garystafford/wav2vec2-deepfake-voice-detector\", \"\", \"MIT\", 360, (\"cloud\",\"mac\",\"cuda\"), {\"cloud\":\"cpu\",\"mac\":\"mps\",\"cuda\":\"cuda\"}),
    \"aud.aasist3\":   ModelSpec(\"aud.aasist3\",  \"MTUCI/AASIST3\",                             \"\", \"Apache-2.0\",1100,(\"mac\",\"cuda\"),         {\"mac\":\"cpu\",\"cuda\":\"cuda\"}, fallback_repo=\"nii-yamagishilab/aasist\"),
    \"aud.whisper\":   ModelSpec(\"aud.whisper\",  \"openai/whisper-tiny\",                       \"\", \"MIT\",       75,  (\"cloud\",\"mac\",\"cuda\"), {\"cloud\":\"cpu\",\"mac\":\"cpu\",\"cuda\":\"cpu\"}),
    \"embed.wavlm\":   ModelSpec(\"embed.wavlm\",  \"microsoft/wavlm-base-plus\",                 \"\", \"MIT\",       380, (\"cloud\",\"mac\",\"cuda\"), {\"cloud\":\"cpu\",\"mac\":\"mps\",\"cuda\":\"cuda\"}),
    \"embed.dino\":    ModelSpec(\"embed.dino\",   \"facebook/dinov2-base\",                      \"\", \"Apache-2.0\",350, (\"mac\",\"cuda\"),         {\"mac\":\"mps\",\"cuda\":\"cuda\"}),
    \"vid.syncnet\":   ModelSpec(\"vid.syncnet\",  \"lithiumice/syncnet\",                        \"\", \"MIT\",       56,  (\"mac\",\"cuda\"),         {\"mac\":\"mps\",\"cuda\":\"cuda\"}),
}
```

**Startup behaviour:** on first request only — verify repo+SHA reachable. If not, fall back; else **disable signal** (never crashes; logs structured warning).

### 11.1 Memory & VRAM strategy

#### `mac_full` (Apple M1 Max, 32 GB unified)
- `TORCH_DEVICE=mps`, **fp32 default** (fp16 on MPS often slower)
- Multiple models stay resident; no aggressive eviction
- Peak unified-memory budget: 12 GB ceiling
- DIRE forced CPU on Mac (3D-conv MPS fallback would be slower than CPU); enable via `ENABLE_DIRE_MPS=true` to experiment
- AASIST3 / RawNet2 on CPU
- Faster-whisper uses CTranslate2 with Metal accel internally

#### `cuda_full` (RTX 3050, 4 GB VRAM)
- `TORCH_DEVICE=cuda`, fp16 autocast everywhere
- Sequential model loading per modality; LRU evict + `torch.cuda.empty_cache()` between stages
- Max simultaneous resident ≤ 2 GB

| Stage | Resident models | Peak VRAM |
|---|---|---|
| Image ingress | RetinaFace + NPR fp16 | 300 MB |
| Image UFD pass | UFD fp16 (NPR evicted) | 850 MB |
| Image DIRE pass | DIRE fp16 (UFD evicted) | 1.1 GB |
| Image embed | CLIP fp16 | 600 MB |

Headroom always ≥ 2.5 GB on 3050.

#### `cloud_lite` (CPU container)
- `TORCH_DEVICE=cpu`
- ONNX-INT8 where available (CLIP exported via `optimum` at install time)
- All `mac_full`/`cuda_full`-only modules skipped via guarded imports
- **HF Inference API fallback (opt-in)** for UFD/DIRE when `HF_TOKEN` set
- Peak RAM ≈ 2 GB

### 11.2 Per-call timeout budget (AGENTS.md compliance)

| Operation | Hard timeout |
|---|---|
| MongoDB read/write | 100 ms |
| FAISS lookup | 50 ms |
| SerpAPI reverse search | 8 s |
| HF model download | 60 s (first call) |
| Single detector predict — image | 5 s |
| Gemini narrator call | 30 s |
| Gemini VLM tiebreaker call | 30 s |
| Full job — image (`cloud_lite`) | 30 s |
| Full job — image (`mac_full` / `cuda_full`) | 20 s |

Timeouts via `asyncio.wait_for`. On exceed → signal disabled for that job, others continue.

---

## 12. API Contract

Base: `{REACT_APP_BACKEND_URL}/api`

### POST `/analyze` (multipart)
**Request:** `file: <binary>`, optional `hints: {modality?: \"image\"|\"audio\"|\"video\"}`
**Response 202:**
```json
{ \"job_id\": \"uuid\", \"modality\": \"image\", \"status\": \"queued\", \"profile\": \"mac_full\" }
```

### GET `/jobs/{job_id}`
```json
{ \"job_id\": \"...\", \"modality\": \"image\", \"status\": \"queued|running|done|failed\",
  \"progress\": 0.65, \"stage\": \"tier1_clip0\",
  \"started_at\": \"...\", \"finished_at\": null }
```

### GET `/jobs/{job_id}/result?debug=0|1`
Only when `status == \"done\"`. Full payload (truncated example):
```json
{
  \"job_id\": \"...\",
  \"modality\": \"image\",
  \"profile\": \"mac_full\",
  \"calibration\": \"platt_refdb\",
  \"fusion_model\": \"uniform\",
  \"content_type\": \"selfie_portrait\",
  \"verdict\": \"AI-GENERATED\",
  \"p_ai_generated\": 0.86,
  \"confidence\": 0.78,
  \"agreement\": 0.83,
  \"extremity\": 0.72,
  \"cross_modal_bonus\": 0.06,
  \"abstained\": false,
  \"provenance\": { \"hit\": false, \"source\": \"none\" },
  \"vlm_invoked\": true,
  \"reverse_invoked\": true,
  \"signals\": [
    {\"name\":\"img.clip0\",      \"p_fake\":0.91, \"weight\":0.16, \"explanation\":\"...\"},
    {\"name\":\"img.freq\",       \"p_fake\":0.78, \"weight\":0.12, \"explanation\":\"...\"},
    {\"name\":\"img.prithiv\",    \"p_fake\":0.83, \"weight\":0.14, \"explanation\":\"...\"},
    {\"name\":\"img.meta\",       \"p_fake\":0.99, \"weight\":0.10, \"explanation\":\"...\"},
    {\"name\":\"img.compression\",\"p_fake\":0.88, \"weight\":0.10, \"explanation\":\"PNG bit_depth=8 color_type=2 no tEXt; matches SDXL default\"},
    {\"name\":\"img.retrieval\",  \"p_fake\":0.88, \"weight\":0.18, \"explanation\":\"12/15 nearest neighbors AI\"},
    {\"name\":\"img.reverse\",    \"p_fake\":0.92, \"weight\":0.10, \"explanation\":\"Top hit: civitai.com/posts/...\"},
    {\"name\":\"img.vlm\",        \"p_fake\":0.92, \"weight\":0.10, \"explanation\":\"Warped fingers; inconsistent shadows\"}
  ],
  \"retrieval\": {\"k\":15, \"neighbors\":[...]},
  \"reverse_search\": {\"hits\":[{\"url\":\"...\",\"domain\":\"civitai.com\",\"date\":\"2024-03\"}, ...]},
  \"xai\": {
    \"heatmap_url\": \"/api/jobs/{id}/assets/heatmap.png\",
    \"frequency_plot_url\": \"/api/jobs/{id}/assets/fft.png\",
    \"metadata\": {...},
    \"compression_fingerprint\": {...},
    \"narrative\": \"...\",
    \"narrative_source\": \"gemini|fallback_template\"
  },
  \"input\": {\"filename\":\"cat.png\",\"sha256\":\"...\",\"bytes\":238923,\"mime\":\"image/png\"},
  \"durations_ms\": {\"preprocess\":120,\"tier0\":40,\"prithiv\":640,\"clip0\":810,\"compression\":18,\"freq\":48,\"meta\":12,\"retrieval\":52,\"reverse\":1840,\"vlm\":820,\"fusion\":3,\"xai\":210},
  \"debug\": null  /* populated only if ?debug=1 */
}
```

### POST `/jobs/{job_id}/correct`
**Request:** `{ \"user_label\": \"ai\" | \"real\" }`
**Action:** writes `labels` doc, appends upload's embedding + label to `refdb/image_*_hard.npy`, rebuilds hard-negatives FAISS index, hot-reloads in registry.
**Response:** `{ \"ok\": true, \"refdb_hard_size\": 23 }`

### GET `/jobs/{job_id}/assets/{name}`
Serves cached PNG/JSON. Filename whitelist enforced.

### GET `/jobs/{job_id}/report.json`
Full JSON download.

### GET `/history?limit=20`
Last N jobs.

### GET `/health`
```json
{
  \"status\": \"ok\",
  \"profile\": \"mac_full\",
  \"signals_loaded\": [\"img.prithiv\",\"img.clip0\",\"img.freq\",\"img.meta\",\"img.compression\",\"img.npr\",\"img.ufd\",\"img.dire\",\"img.retrieval\"],
  \"db_ok\": true,
  \"gemini_ok\": true,
  \"serpapi_ok\": true,
  \"refdb_loaded\": true,
  \"refdb_size\": {\"image_real\":1500,\"image_ai\":1500,\"image_real_hard\":12,\"image_ai_hard\":4},
  \"fusion_mode\": \"uniform\",
  \"calibration\": \"platt_refdb\",
  \"ece_refdb_holdout\": 0.043,
  \"auroc_refdb_holdout\": 0.91,
  \"n_user_labels\": 0,
  \"uptime_s\": 1834
}
```

### GET `/profile`, `/modalities`, `/refdb/stats`, `/refdb/thumb/{id}.jpg`
Standard surfaces.

### Error envelope (single shape)
```json
{ \"error\": \"UPLOAD_TOO_LARGE\", \"message\": \"...\", \"request_id\": \"uuid\" }
```
Codes: `UNSUPPORTED_MIME`, `UPLOAD_TOO_LARGE`, `CORRUPT_MEDIA`, `MODEL_LOAD_FAILED`, `OOM`, `RATE_LIMITED`, `REFDB_MISSING`, `SERPAPI_QUOTA`, `INTERNAL`. Status 400/413/415/422/429/500.

---

## 13. MongoDB Schemas

Database: `${DB_NAME}`. **All `_id` excluded from API responses. UUIDs everywhere.**

### `jobs` collection
```json
{
  \"_id\": \"<uuid>\", \"created_at\": \"ISO-8601 UTC\", \"updated_at\": \"ISO-8601 UTC\",
  \"status\": \"queued|running|done|failed\", \"stage\": \"string\", \"progress\": 0.0,
  \"modality\": \"image|audio|video\", \"profile\": \"cloud_lite|mac_full|cuda_full\",
  \"input\": {\"filename\":\"...\", \"sha256\":\"...\", \"bytes\":0, \"mime\":\"...\", \"path\":\"storage/jobs/{id}/original.ext\"},
  \"error\": null
}
```

### `results` collection
Same shape as `/result` response (without nested `input`).

### `labels` collection
```json
{ \"_id\": \"<uuid>\", \"job_id\": \"<uuid>\", \"user_label\": \"ai|real\", \"submitted_at\": \"...\", \"consumed\": false }
```

### `serpapi_cache` collection
```json
{ \"_id\": \"<sha256 of upload>\", \"response\": {...}, \"fetched_at\": \"...\", \"ttl_until\": \"...\" }
```

**Indexes:** `jobs.created_at desc`, `results.job_id unique`, `jobs.status`, `labels.consumed`, `serpapi_cache.ttl_until`.

---

## 14. Frontend Spec

### 14.1 Theme — Dark forensic / \"Control Room\"
- Background: `#0A0A0A` page, `#121212` surface, `#1A1A1A` hover, `#27272A` borders
- Text: `#FFFFFF` primary, `#A1A1AA` secondary, `#71717A` muted
- Brand: `#06B6D4` cyan
- Verdict: `#EF4444` AI-GENERATED, `#10B981` REAL, `#F59E0B` INCONCLUSIVE
- Provenance badge: `#10B981` filled when fired
- VLM-invoked badge / Reverse-search-invoked badge: cyan pills
- Fonts: **IBM Plex Sans** (headings), **Inter** (body), **JetBrains Mono** (data) — Google Fonts CDN
- Icons: `@phosphor-icons/react` (Duotone/Regular)
- **NO purple gradients. NO emoji. NO \"AI slop\".**

### 14.2 Routes
| Route | Purpose |
|---|---|
| `/` | Upload zone + recent history + how-it-works strip |
| `/job/:id` | Live progress (while running) → Control Room result grid (when done) |
| `/about` | Plain-English COEF explainer, 5 tiers, non-goals, calibration + DB status |

### 14.3 Result-page component tree (image first-finish, image scope)
```
JobPage
├── ProgressSteps (while running — 5-tier visualisation)
└── ResultView (when done)
    ├── VerdictCard (+ ProvenanceBadge + VLMBadge + ReverseSearchBadge + ContentTypeBadge)
    ├── ConfidenceAgreementBars
    ├── NarrativePanel
    ├── SignalBarChart (Recharts horizontal, contribution bars)
    ├── HeatmapPanel (image GradCAM)
    ├── FrequencyPanel (Recharts RadarChart)
    ├── MetadataTable (EXIF + C2PA + SHA256 + bytes + mime)
    ├── CompressionFingerprintPanel  // NEW v1.3
    ├── RetrievalNeighborsPanel (5 thumbs + distances + label)
    ├── ReverseSearchPanel (5 hits w/ domain + date + link)  // NEW v1.3
    ├── VLMRationalePanel (when invoked)
    ├── DownloadActions
    └── DeveloperPanel (toggle Ctrl/Cmd+D)  // NEW v1.3
        ├── RawSignalTable
        ├── ThresholdSliders
        ├── DurationsTable
        ├── FusionVectorDump
        └── GateStateDump
```

### 14.4 data-testid coverage (mandatory, kebab-case, every interactive + critical element)
`media-upload-dropzone`, `analyze-submit-btn`, `verdict-card-container`, `confidence-progress-bar`, `agreement-progress-bar`, `signal-bar-<name>`, `narrative-text`, `download-report-btn`, `history-item-<id>`, `chart-fft-radial`, `chart-signal-bars`, `metadata-technical-table`, `profile-badge`, `provenance-badge`, `vlm-invoked-badge`, `reverse-search-badge`, `content-type-badge`, `job-status-pill`, `retrieval-neighbor-<n>`, `reverse-hit-<n>`, `vlm-rationale-list`, `compression-fingerprint-panel`, `dev-mode-toggle`, `dev-raw-signal-row-<name>`, `threshold-slider-<name>`, `correct-verdict-ai-btn`, `correct-verdict-real-btn`.

### 14.5 Motion
- Hover: `transition-colors duration-200`
- Card entrance: `animate-in fade-in slide-in-from-bottom-2 duration-300`
- Progress: `transition-all duration-500 ease-out`

---

## 15. File / Folder Structure (AGENTS.md naming compliance)

```
/app
├── backend/
│   ├── server.py
│   ├── config.py                 # Pydantic Settings, env validation
│   ├── deps.py
│   ├── routes/
│   │   ├── analyze.py
│   │   ├── jobs.py
│   │   ├── history.py
│   │   ├── refdb.py
│   │   ├── correct.py            # NEW: user-corrected verdicts
│   │   └── health.py
│   ├── schemas/
│   │   ├── jobs.py
│   │   └── results.py
│   ├── services/
│   │   ├── runner.py             # 5-tier orchestrator
│   │   ├── storage.py
│   │   ├── router.py             # MIME + magic-bytes modality detection
│   │   └── device.py             # auto-detects cuda|mps|cpu → profile
│   ├── provenance/
│   │   ├── c2pa_check.py
│   │   ├── synthid_check.py
│   │   ├── sd_watermark.py
│   │   └── meta_watermark.py
│   ├── detectors/
│   │   ├── base.py
│   │   ├── registry.py           # SHA-pinned + device-pref + LRU evict
│   │   ├── tta.py
│   │   ├── content_type.py       # NEW v1.3 — CLIP zero-shot 6-way
│   │   └── image/
│   │       ├── prithiv.py
│   │       ├── frequency.py
│   │       ├── clip0.py
│   │       ├── meta.py
│   │       ├── compression.py    # NEW v1.3 — PNG/JPEG/WebP forensics
│   │       ├── npr.py
│   │       ├── ufd.py
│   │       └── dire.py
│   ├── retrieval/
│   │   ├── embedder.py           # CLIP / DINOv2 / WavLM wrappers
│   │   ├── index.py              # FAISS load/query/dedup
│   │   ├── hard_negatives.py     # NEW v1.3 — append + reindex
│   │   └── build_db.py
│   ├── reverse_search/           # NEW v1.3
│   │   ├── serpapi_client.py
│   │   ├── interpreter.py        # parse hits → p_reverse
│   │   └── cache.py              # Mongo-backed 24h cache
│   ├── vlm/
│   │   ├── judge.py
│   │   └── prompts.py
│   ├── fusion/
│   │   ├── calibrate.py          # Platt + temperature + isotonic
│   │   ├── fuse.py               # adaptive: uniform | lr | gbdt
│   │   ├── selector.py
│   │   ├── crossmodal_bonus.py   # NEW v1.3
│   │   ├── weights_uniform.json
│   │   ├── platt.json            # generated by refDB build
│   │   ├── weights_lr.json
│   │   └── weights_gbdt.txt
│   ├── abstention/               # NEW v1.3
│   │   └── gate.py               # content-type-aware thresholds
│   ├── xai/
│   │   ├── heatmap.py
│   │   ├── plots.py
│   │   ├── narrator.py           # Gemini + fallback
│   │   ├── prompts.py
│   │   └── fallback_templates.py
│   ├── db/
│   │   ├── mongo.py
│   │   └── repos.py
│   ├── utils/
│   │   ├── logs.py               # structured JSON logging
│   │   ├── errors.py             # custom exceptions + envelope
│   │   ├── retry.py              # exponential backoff helper
│   │   └── timing.py             # per-stage timer ctx
│   ├── scripts/
│   │   ├── build_reference_db.py
│   │   ├── run_calibration.py
│   │   ├── tune_thresholds.py    # NEW v1.3 — content-type-aware tuning
│   │   ├── verify_registry.py
│   │   └── license_audit.py
│   ├── calibration/
│   │   ├── samples/              # bundled tiny eval set
│   │   └── report.md             # generated
│   ├── storage/
│   │   ├── models/               # HF cache (HF_HOME)
│   │   ├── refdb/
│   │   │   ├── image_real.index   image_real.npy   image_real_labels.json   image_real_sources.json
│   │   │   ├── image_ai.index     image_ai.npy     image_ai_labels.json     image_ai_sources.json
│   │   │   ├── image_real_hard.npy   image_ai_hard.npy   (hard-negatives, NEW v1.3)
│   │   │   └── thumbs/
│   │   ├── cache/
│   │   │   └── serpapi/          # 24h SHA256-keyed JSON
│   │   └── jobs/{job_id}/
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── fixtures/
│   ├── requirements.txt
│   └── .env
│
├── frontend/
│   ├── package.json
│   ├── tailwind.config.js
│   ├── craco.config.js
│   ├── .env
│   └── src/
│       ├── App.js
│       ├── index.css
│       ├── lib/
│       │   ├── api.js
│       │   ├── format.js
│       │   └── devmode.js            # NEW v1.3 — local-storage flag
│       ├── components/
│       │   ├── ui/                   # shadcn
│       │   ├── DropZone.jsx
│       │   ├── VerdictCard.jsx
│       │   ├── ProvenanceBadge.jsx
│       │   ├── VLMBadge.jsx
│       │   ├── ReverseSearchBadge.jsx    # NEW v1.3
│       │   ├── ContentTypeBadge.jsx      # NEW v1.3
│       │   ├── ConfidenceAgreementBars.jsx
│       │   ├── NarrativePanel.jsx
│       │   ├── SignalBarChart.jsx
│       │   ├── HeatmapPanel.jsx
│       │   ├── FrequencyPanel.jsx
│       │   ├── MetadataTable.jsx
│       │   ├── CompressionFingerprintPanel.jsx   # NEW v1.3
│       │   ├── RetrievalNeighborsPanel.jsx
│       │   ├── ReverseSearchPanel.jsx            # NEW v1.3
│       │   ├── VLMRationalePanel.jsx
│       │   ├── CorrectVerdictBar.jsx             # NEW v1.3
│       │   ├── DeveloperPanel.jsx                # NEW v1.3
│       │   ├── ProgressSteps.jsx
│       │   └── HistoryList.jsx
│       ├── pages/
│       │   ├── UploadPage.jsx
│       │   ├── JobPage.jsx
│       │   └── AboutPage.jsx
│       └── styles/
│           └── globals.css
│
├── memory/
│   ├── Masterplan.md             # this file
│   ├── PRD.md                    # updated on every finish
│   └── test_credentials.md       # no auth in MVP; placeholder
│
├── test_reports/
├── design_guidelines.json
└── README.md
```

---

## 16. Dependencies

### 16.1 Backend (additions on top of base template)
```
# Core ML / forensics
torch>=2.2
torchvision
torchaudio
transformers>=4.45
huggingface_hub
safetensors
accelerate

scikit-learn          # Platt / isotonic / LR
lightgbm              # GBDT fusion when n>=500
faiss-cpu             # retrieval ANN

opencv-python-headless
Pillow
imageio
pywavelets
librosa               # Phase 1 follow-up
soundfile             # Phase 1 follow-up
ffmpeg-python         # Phase 1 follow-up
matplotlib

# Forensics & provenance
exifread
c2pa                  # Tier-0 C2PA validate
invisible-watermark   # SD watermark detect
synthid-text          # guarded import

# Optional CPU acceleration
onnxruntime
optimum               # ONNX export for CLIP on cloud_lite

# Reverse search & Gemini
requests              # SerpAPI direct call
emergentintegrations==0.1.0   # Gemini wrapper

# Scraping pipeline (used by build_reference_db only)
beautifulsoup4
yt-dlp                # Phase 1 follow-up (video sources)

# Quality
ruff                  # AGENTS.md lint
mypy                  # AGENTS.md type checks
pytest
pytest-cov
pytest-asyncio
httpx                 # async client for tests
```

**Install pattern (mandatory):** `pip install <pkg>` then `pip freeze > /app/backend/requirements.txt` (never rewrite requirements by hand).

**Guarded imports:** `mac_full`/`cuda_full`-only libs (insightface, mediapipe, scenedetect, etc.) imported inside detector modules with `try/except ImportError` → signal silently skipped on `cloud_lite`.

### 16.2 Frontend additions
```
@phosphor-icons/react       # via: yarn add @phosphor-icons/react
# (recharts, sonner, lucide-react already present)
```

---

## 17. Environment Variables

`/app/backend/.env` (additive; never delete existing keys):
```
# Protected (do not modify)
MONGO_URL=\"mongodb://localhost:27017\"
DB_NAME=\"test_database\"
CORS_ORIGINS=\"*\"

# Detector control
GEMINI_API_KEY=
SERPAPI_KEY=
HF_TOKEN=                       # optional, enables HF Inference API fallback
DETECTOR_PROFILE=auto           # auto | cloud_lite | mac_full | cuda_full
TORCH_DEVICE=auto
HF_HOME=/app/backend/storage/models
ENABLE_DIRE_MPS=false
ENABLE_VLM_TIEBREAKER=true
ENABLE_REVERSE_SEARCH=true

# Limits & gates
MAX_UPLOAD_MB=200
VIDEO_MAX_SECONDS=120            # Phase 1 follow-up
VLM_EXTREMITY_THRESHOLD=0.25
VLM_AGREEMENT_THRESHOLD=0.63
REVERSE_EXTREMITY_THRESHOLD=0.30
REVERSE_AGREEMENT_THRESHOLD=0.70

# Calibration / abstention defaults (per-content-type overrides in code)
ABSTAIN_HIGH=0.75
ABSTAIN_LOW=0.25
ABSTAIN_AGREE=0.55

# Misc
WDS_SOCKET_PORT=443
ENABLE_HEALTH_CHECK=false
```

**Security.**
- `GEMINI_API_KEY`, `SERPAPI_KEY`, `HF_TOKEN` backend-only; never logged; never returned in any API response.
- CORS locked to `REACT_APP_BACKEND_URL` origin in production.
- Uploads served only via UUID + whitelist route, never raw path.
- Filename sanitization on upload (UUID + sniffed ext).

---

## 18. Error Handling, Resilience, Observability

### 18.1 Error envelope
Single shape across all endpoints (§12). Status codes per code.

### 18.2 Validation (boundary only — AGENTS.md principle)
- MIME sniff via `python-magic` + extension cross-check (reject mismatch → `UNSUPPORTED_MIME`)
- Max upload `MAX_UPLOAD_MB` (default 200 MB) → `UPLOAD_TOO_LARGE` (413)
- Filename sanitization (no path traversal; replaced with UUID)

### 18.3 Retry & resilience
- HF model download: 3 retries, exponential backoff (1s/2s/4s), jittered
- Gemini call: 2 retries on 429, then fall through to template / drop signal
- SerpAPI call: 2 retries on 429/5xx, then drop signal; on 402 (quota), surface `SERPAPI_QUOTA` to debug panel only
- MongoDB ops: motor built-in retries; surface `INTERNAL` after 1 retry
- FAISS index load: fail-fast with `REFDB_MISSING` (suggests running build script)

### 18.4 Structured logging (one line per event, JSON)
Fields: `ts`, `level`, `request_id`, `job_id`, `route`, `event`, `dur_ms`, `signal_name`, `status`, `error_code`, `profile`. Sink: stdout (supervisor captures).

### 18.5 Per-stage timings
Recorded in `results.durations_ms` per signal — surfaced in `/result` and `/health`.

### 18.6 Health surfaces
- `GET /health` — full diagnostic block (§12)
- `GET /profile` — current profile + signal list + calibration status + gate config
- `GET /modalities` — supported + per-modality enabled signals
- `GET /refdb/stats` — counts per modality + hard-neg counts + build date + AUROC of retrieval-alone + ECE

---

## 19. Testing Strategy (AGENTS.md ≥80 % on critical modules)

### 19.1 Unit tests (`pytest`)
Target ≥80 % coverage on `detectors/`, `fusion/`, `retrieval/`, `provenance/`, `reverse_search/`, `abstention/`.

- Each detector `predict()` on 1 real + 1 fake fixture → score bounds asserted
- Fusion math: deterministic at fixed inputs
- Adaptive selector: returns expected mode for n ∈ {0, 50, 100, 499, 500, 5000}
- Content-type-aware gate: truth table per content type
- Pydantic schema round-trips
- Platt scaling: identity at A=1,B=0; monotonic otherwise
- Retrieval: SHA dedup excludes self-match; FAISS query returns k results; hard-negative append + query works
- Provenance: synthetic C2PA + SD-watermark fixtures detected; clean photo not flagged
- VLM gate: invoked iff thresholds tripped
- Reverse-search gate: invoked iff thresholds tripped; cache HIT skips API call
- Compression forensics: PNG fingerprint fixtures (SDXL/MJ/Flux exports vs camera-JPEG) → expected scores

### 19.2 Integration tests
- `POST /analyze` with small PNG → completes within budget → valid result schema
- MongoDB jobs/results repo round-trips
- Asset route Content-Type correctness + path-traversal rejection
- Error envelope for all 4xx/5xx paths
- Tier-0 short-circuit path produces correct response shape with `provenance.hit=true`
- VLM gate disabled when key missing → pipeline still completes
- SerpAPI disabled when key missing → pipeline still completes
- Correction endpoint appends to hard-neg index and survives restart

### 19.3 E2E via `testing_agent_v3` (after backend integration tests pass)
- Upload real photo with EXIF → REAL (high agreement)
- Upload AI image with no metadata → AI-GENERATED
- Upload ambiguous → INCONCLUSIVE
- Upload C2PA-signed → REAL with provenance badge
- Upload SD-watermarked → AI-GENERATED with provenance badge
- All XAI assets accessible; retrieval thumbnails render; reverse-search hits render
- VLM rationale renders when invoked
- Download JSON report parses
- History list renders previous jobs
- All `data-testid` selectors hit
- Developer mode toggle reveals raw signal table + threshold sliders work

### 19.4 Calibration evaluation
After `run_calibration.py`, `calibration/report.md` records per-modality:
- AUROC, ECE, abstention rate, fusion mode used, sample count
- **AUROC of retrieval signal alone** (sanity-check it's contributing)
- **AUROC of reverse-search signal alone on uncertain slice**
- **AUROC of VLM signal alone on uncertain slice**
- **AUROC of compression forensics alone**
Committed to repo.

### 19.5 License audit
`python -m backend.scripts.license_audit` enumerates all model SHAs + licenses, retrieval DB sources + licenses. Fails CI if any unknown / restrictive license bundled.

---

## 20. Milestones (M0→M3 first-finish; M4+ outlined)

| ID | Deliverable | Exit criteria |
|---|---|---|
| **M0** | Scaffold + device-detect + skeleton | FastAPI skeleton, mongo wiring, React shell, upload + job stubs, `/api/profile` returns auto-detected profile, `/api/health` green, lint passes (ruff + mypy), pytest skeleton green. |
| **M1** | Tier-1 image `cloud_lite` + cold-start calibration | `prithiv` + `freq` + `clip0` + `meta` + `compression` + content-type router + TTA + uniform fusion + GradCAM + narrative + abstention. AI img → AI on fixtures; real photo → REAL. Platt-on-refDB calibration loaded. ECE in `/health`. |
| **M2** | Tier-0 Provenance Gate | C2PA + SD-watermark + (SynthID guarded) short-circuit verified on signed/watermarked fixtures. Bypasses abstention; UI renders Provenance badge. |
| **M3** | Tier-2 Retrieval + Tier-2.5 Reverse Search + Tier-3 VLM + Developer mode + Correction endpoint | `build_reference_db.py` runs end-to-end on permissive sources; FAISS image_real/image_ai indexes load; retrieval AUROC-alone ≥ 0.75 on held-out. SerpAPI integrated + cached + gated. Gemini VLM gated + rationale rendered. Hard-negative append works. Developer panel toggles + threshold sliders work. **`testing_agent_v3` full image E2E suite passes.** **→ FIRST FINISH** |
| M4 | Audio modality | w2v2df + spectral + prosody + WavLM retrieval + spectrogram XAI |
| M5 | Video modality | frames + faces + img_ens + flicker + audio + syncnet + blink + identity + DINOv2 retrieval + frame timeline |
| M6 | `mac_full` + `cuda_full` profile validation | npr + ufd + (dire) wired and unit-tested with device-pref; RTX 3050 VRAM stage table holds; latency budgets met |
| M7 | Active learning + adaptive fusion auto-promote | LR/GBDT auto-promotion verified at n=100, n=500; hot-reload on file-mtime |
| M8 | Phase 1.5 — text detection (Binoculars) | Phase 1.5 |
| M9 | Polish + docker compose + README | `docker compose up` runs `cloud_lite` on a fresh machine |

---

## 21. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| HF model repo / SHA disappears | M | M | SHA pin + fallback repo; signal disabled gracefully |
| Gemini rate-limit / missing key | H | L | Narrator → rule-based; VLM → drop from fusion |
| **SerpAPI quota (100/mo free) exhausted** | M | M | **24h cache + uncertainty gate (~15-25 % invocation); on quota: drop signal with structured warning** |
| RTX 3050 OOM | M | M | Sequential load + fp16 + LRU evict |
| MPS op fallback to CPU silently slows pipeline | M | M | DIRE on Mac defaults CPU; benchmarked; About page documents |
| `cloud_lite` CPU too slow on video | M | M | 8-frame cap + 180 s hard timeout + clear UX (Phase 1 follow-up) |
| Retrieval DB build script fails on a source | M | L | Per-source try/except; logged in build_report; pipeline tolerates partial DB |
| Reference DB AUROC lower than expected | M | M | Documented in `calibration/report.md`; system works; admin can expand DB |
| LAION-CLAP non-commercial license | (resolved) | — | **v1.3 default = WavLM-base-plus (MIT)**; CLAP opt-in |
| **Reverse-search misleading on stock-photo sites** | M | M | **Domain whitelist + date-floor logic; only triggers strong prior on reputable news + creation-date < known-generator-date** |
| User confuses INCONCLUSIVE with bug | M | M | About page explainer; INCONCLUSIVE UX copy explicit |
| VLM hallucinates a forensic claim | M | M | Prompt restricts to visible defects + uncertainty caveats; only 1 signal among many |
| Adversarial uploads (targeted) | L | — | Explicit non-goal, documented |
| Developer-mode threshold override leaks via shared URL | L | L | Toggle is client-side localStorage only; never persisted server-side |

---

## 22. Definition of Done — Phase 1 First-Finish (M0→M3)

- [ ] All `/api/*` endpoints in §12 work
- [ ] **Image modality** produces verdicts + XAI in `cloud_lite` on bundled fixtures
- [ ] Tier-0 short-circuit verified on C2PA + SD-watermark fixtures
- [ ] Tier-2 reference DB built and AUROC-alone ≥ 0.75 per modality on held-out
- [ ] Tier-2.5 reverse search: gated + cached + parses real SerpAPI response; falls back when key absent
- [ ] Tier-3 VLM gate verified: invoked iff thresholds tripped; gracefully absent without key
- [ ] Compression forensics signal contributes; PNG/JPEG fixture scores match expected ranges
- [ ] Content-type router: 6 types classified; type-specific thresholds applied
- [ ] Cold-start calibration: Platt-on-refDB loaded; `ece_refdb_holdout` in `/health` < 0.10
- [ ] Cross-modal multiplicative bonus applied when ≥3 tiers agree; capped at +0.10
- [ ] Hard-negative append + reindex on `/jobs/{id}/correct` works
- [ ] Adaptive fusion: uniform mode works at n<100; LR mode unit-tested via fixture data
- [ ] Developer panel: raw signal table, threshold sliders (client-side re-render), durations table
- [ ] `testing_agent_v3` full image E2E run passes
- [ ] data-testid audit: every interactive + critical element tagged (kebab-case)
- [ ] README with setup, profile-switching, reverse-DB build instructions, SerpAPI/Gemini key acquisition
- [ ] License audit script passes; `licenses.txt` committed
- [ ] No hardcoded secrets; `.env.example` documented
- [ ] AGENTS.md naming compliance — all files short + professional
- [ ] AGENTS.md type-safety — type hints + Pydantic + mypy clean
- [ ] AGENTS.md logging — structured JSON one-line-per-event
- [ ] AGENTS.md timeouts — every external call has explicit timeout
- [ ] AGENTS.md error handling — single error envelope shape
- [ ] Coverage ≥ 80 % on `detectors/`, `fusion/`, `retrieval/`, `provenance/`, `reverse_search/`, `abstention/`
- [ ] Dark forensic theme locked, no purple gradients, fonts loaded, Phosphor icons
- [ ] All three verdict states visually distinct
- [ ] Rule-based narrator fallback verified by removing `GEMINI_API_KEY`
- [ ] Reverse-search fallback verified by removing `SERPAPI_KEY`
- [ ] PRD.md created at `/app/memory/PRD.md`

---

## 23. Mapping to AGENTS.md Industry Standards

| AGENTS.md principle | Where implemented in v1.3 |
|---|---|
| PEP8 + clean naming | §15 file tree (short names: `npr.py`, `dire.py`, `judge.py`, `fuse.py`, etc.); ruff + mypy in CI |
| Modular design / SRP | Per-tier folders (`provenance/`, `retrieval/`, `reverse_search/`, `vlm/`, `fusion/`, `abstention/`, `xai/`); detector base class in `detectors/base.py`; one class per file |
| Async patterns | FastAPI async routes; motor async Mongo; `asyncio.wait_for` for timeouts |
| Logging | §18.4 structured JSON one-line; `utils/logs.py` |
| Timeouts | §11.2 + §18.3 — every external call has explicit `asyncio.wait_for` |
| Circuit breaker | Detector registry SHA-verify fail → disable signal gracefully (effectively a breaker); SerpAPI quota fail → drop signal |
| Real AI integrations (no mocks) | Gemini (real key) + SerpAPI (real key) — confirmed by user; mocks only used in unit tests |
| Async + DB pool | motor + connection from `db/mongo.py` |
| Caching | `storage/cache/serpapi/` + `serpapi_cache` Mongo TTL collection; HF model cache; FAISS in-memory |
| Unit + integration + E2E | §19 (≥80 %); testing_agent_v3 for E2E |
| Type safety | Type hints on all signatures + Pydantic schemas + mypy |
| Security: input validation | §18.2 boundary only (MIME sniff, size cap, sanitize filename) |
| Security: secrets via env | §17 — never logged, never returned |
| Security: rate limiting | (Phase 1.5 — currently single-user local app) |
| Distributed tracing | request_id propagated in all logs; per-stage `dur_ms` |
| Metrics | `/api/health` exposes ECE, AUROC, durations, refDB stats |
| Health endpoints | `/api/health`, `/api/profile`, `/api/modalities`, `/api/refdb/stats` |
| Horizontal scaling | Stateless FastAPI workers + shared Mongo + shared FAISS read-only index (Phase 2 multi-worker) |
| Graceful degradation | Every external dep has a fallback path (Gemini → template, SerpAPI → drop, HF repo → fallback_repo → disable signal) |
| Retry w/ exponential backoff | §18.3; `utils/retry.py` |
| Idempotency | SHA256-keyed SerpAPI cache; reuploads return identical results |
| REST principles | §12 — proper status codes, error envelope, JSON |
| Versioning | API base `/api`; v2 path reserved |
| Data validation (Pydantic) | `schemas/jobs.py`, `schemas/results.py` |
| Migrations | (single Mongo collection; no schema migration needed in Phase 1) |
| AI/ML standards | Model versioning via SHA-pinned registry; fallback strategies for AI failures; prompt engineering in `vlm/prompts.py` + `xai/prompts.py`; token usage minimized via gating |
| ADRs | This document IS the ADR — every \"(NEW v1.3)\" entry is a decision record |

---

## 24. Appendix A — Gemini Narrator Prompt

```
SYSTEM: You are a forensic media-authenticity analyst. Explain evidence plainly
for non-experts. Never exceed 5 sentences. No hype. State uncertainty when
signals disagree. Never claim 100 % certainty.

USER: Given these detector signals for an {modality} file, write a 3–5 sentence
explanation of why the system reached verdict {verdict} with confidence {conf}%
and agreement {agree}%. Reference the 2–3 strongest signals by name. If verdict
is INCONCLUSIVE, explicitly state evidence is mixed and name which signals
disagree.
Content type: {content_type}
Provenance: {provenance_summary}
Reverse search: {reverse_summary}
Signals (JSON): {signals_json}
```

## 25. Appendix B — VLM Tiebreaker Prompt

```
SYSTEM: You are a forensic image analyst. Examine the supplied image and
report ONLY visually verifiable defects that suggest AI generation
(warped anatomy, inconsistent shadows, impossible reflections, text gibberish,
texture artifacts, semantic impossibilities). Do not speculate beyond visible
evidence. If the image looks plausibly authentic, say so.

USER: Rate this image from 0.0 (clearly real photograph) to 1.0 (clearly
AI-generated). Then list up to 5 bullet points of specific visual defects
you observed, each with a brief location (\"upper-left\", \"hand region\", etc.).
Return STRICT JSON, no prose outside JSON:
{\"p_ai\": <float 0..1>, \"defects\": [<str>, ...], \"rationale\": \"<2 sentences>\"}
```

## 26. Appendix C — Rule-based Narrator Fallback Templates

```
AI-GENERATED:
\"This {modality} was flagged as AI-generated with {conf}% confidence
and {agree}% detector agreement. The strongest evidence comes from
{signal_1_name} ({signal_1_explanation}) and {signal_2_name}
({signal_2_explanation}).\"

REAL:
\"This {modality} appears authentic with {conf}% confidence and
{agree}% detector agreement. {signal_1_name} ({signal_1_explanation})
and {signal_2_name} ({signal_2_explanation}) both support a real origin.\"

INCONCLUSIVE:
\"The forensic signals conflict on this {modality}. {signal_1_name}
suggests AI-generation while {signal_2_name} suggests authenticity.
Manual review recommended; per-signal evidence is shown below.\"

PROVENANCE_HIT_AI:
\"This {modality} contains an AI-watermark ({provenance_source}) which
matched with very high confidence. Embedded watermarks are placed by
the generator itself, so this verdict is decisive.\"

PROVENANCE_HIT_REAL:
\"This {modality} has a valid C2PA content credential signed by
{c2pa_producer}. Content credentials are cryptographically verified
provenance metadata.\"
```

## 27. Appendix D — Reverse-search Interpreter Rules (Tier-2.5)

```python
def interpret_serpapi(response: dict, upload_meta: dict) -> dict:
    hits = response.get(\"image_results\", []) + response.get(\"visual_matches\", [])
    if not hits:
        return {\"p_fake\": None, \"reason\": \"no_hits\", \"top_hits\": []}

    # Domain priors
    NEWS = {\"reuters.com\",\"apnews.com\",\"bbc.co.uk\",\"bbc.com\",\"nytimes.com\",\"washingtonpost.com\",\"theguardian.com\",\"cnn.com\"}
    AI_GALLERIES = {\"civitai.com\",\"lexica.art\",\"openart.ai\",\"midjourney.com\",\"prompthero.com\"}
    AI_SOCIAL = {\"reddit.com/r/stablediffusion\",\"reddit.com/r/midjourney\",\"reddit.com/r/aiart\"}
    STOCK = {\"gettyimages.com\",\"shutterstock.com\",\"istockphoto.com\",\"alamy.com\"}

    earliest = min((h.get(\"date\") for h in hits if h.get(\"date\")), default=None)
    domains = [h.get(\"source\",\"\").lower() for h in hits[:10]]

    if any(d in AI_GALLERIES for d in domains) or any(s in d for d in domains for s in AI_SOCIAL):
        return {\"p_fake\": 0.93, \"reason\": \"ai_gallery_hit\", \"top_hits\": hits[:5]}
    if earliest and earliest < \"2022-01\" and any(d in NEWS for d in domains):
        return {\"p_fake\": 0.07, \"reason\": \"pre_ai_era_news\", \"top_hits\": hits[:5]}
    if any(d in NEWS for d in domains):
        return {\"p_fake\": 0.12, \"reason\": \"news_domain\", \"top_hits\": hits[:5]}
    if any(d in STOCK for d in domains):
        return {\"p_fake\": 0.18, \"reason\": \"stock_agency\", \"top_hits\": hits[:5]}
    return {\"p_fake\": None, \"reason\": \"no_strong_prior\", \"top_hits\": hits[:5]}
```

## 28. Appendix E — Operational Commands

```bash
# Verify all model SHAs reachable (run in CI)
python -m backend.scripts.verify_registry

# Build the reference DB (one-time at install / when refreshing)
python -m backend.scripts.build_reference_db --modalities image

# Run cold-start calibration on refDB (called automatically by build_reference_db)
python -m backend.scripts.run_calibration --modality image --source refdb

# After enough user labels accumulate:
python -m backend.scripts.run_calibration --modality image --source mixed --min-labels 100

# Tune content-type-aware thresholds from accumulated labels
python -m backend.scripts.tune_thresholds --modality image

# License audit
python -m backend.scripts.license_audit
```

---

## 29. What Changed from v1.2 — At a Glance

| Area | v1.2 | v1.3 |
|---|---|---|
| First-finish scope | All 3 modalities | **Image-only (M0→M3) for fast validation** |
| Tier 2.5 reverse search | Not present | **NEW — SerpAPI gated by uncertainty** |
| Compression forensics | Not present | **NEW — PNG/JPEG/WebP fingerprint signal** |
| Content-type abstention | Single global threshold | **6 type-specific thresholds via CLIP zero-shot** |
| Cold-start calibration | \"Uncalibrated\" badge until 100 user labels | **Platt-on-refDB ships day 0** |
| Cross-modal fusion | Linear sum | **Multiplicative bonus when ≥3 tiers agree** |
| Hard-negative memory | Not present | **NEW — corrected verdicts append to refDB instantly** |
| Developer mode UI | Not present | **NEW — raw signal table + live threshold sliders** |
| ECE in /health | Not exposed | **NEW — live calibration drift metric** |
| Audio embedder default | LAION-CLAP (CC-BY-NC) | **WavLM-base-plus (MIT)**; CLAP opt-in |
| HF Inference API fallback | Not mentioned | **Opt-in for `cloud_lite` heavy models** |
| Headline KPI | \"Improve OOD AUROC\" | **\"≥95 % accuracy on non-abstained, tunable abstention rate\"** |
| AGENTS.md mapping | Implicit | **Explicit per-principle table in §23** |

---

## 30. Approval Gate

This v1.3 document is **plan-only**. No code is written yet.

**Before any implementation begins**, please confirm:
1. ✅ M0→M3 image-only first-finish is the right starting point
2. ✅ SerpAPI key + Gemini key will be placed in `/app/backend/.env` by the user
3. ✅ The 7 new v1.3 capabilities (reverse search, cold-start calibration on refDB, content-type abstention, compression forensics, cross-modal bonus, hard-negative memory, developer mode) are approved
4. ✅ The AGENTS.md mapping in §23 captures all standards you care about

**Once approved, the implementation order will be:**
M0 → M1 → M2 → M3 → first finish + `testing_agent_v3` E2E run → user review → M4 (audio) → M5 (video) → ...

**End of Master Plan v1.3.** Single source of truth. Ready for implementation on your approval.
"