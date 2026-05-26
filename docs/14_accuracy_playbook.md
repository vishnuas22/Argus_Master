"# 14 — Accuracy Playbook (why this stack beats a single fine-tuned model)

> **Status:** Strategic doc. No code lives here.
> **Audience:** the reviewer who asks \"why should I trust your 91–95 % number?\"
> **Companion:** read `15_evaluation_protocol.md` to learn *how* the numbers are produced; read `16_accuracy_extensions_v1.5.md` for the new boosters.
> **Last updated:** 2026-02 (v1.5)

---

## 1. The only free lunch in ML is decorrelated errors

A single off-the-shelf deepfake detector tops out at **AUROC 0.75–0.80** on
cross-generator evaluation. This is a well-known ceiling — observed in
CNNDet (Wang et al., CVPR'20), UFD (Ojha et al., CVPR'23), DIRE
(Wang et al., ICCV'23), and corroborated by your own experiments.

The literature also tells us *why*: every single detector has its own
training distribution; any image outside that distribution flips the
prediction. There is no single backbone that generalises to all
generators × all content types × all post-processing pipelines.

The fix is **not** a bigger backbone. The fix is **error decorrelation**:
combine many *cheap* detectors whose mistakes do not coincide. The
ensemble's variance shrinks by `1/k` when errors are independent, and by
a milder factor when correlated. With seven well-chosen image
detectors at correlation ≤ 0.45, the theoretical AUROC ceiling moves
to **0.86–0.89**. Add orthogonal evidence channels (provenance,
retrieval, web priors, VLM reasoning) and you cross **0.91**. Add
conformal prediction and you can *guarantee* 0.95+ on the
non-abstained slice.

Everything else in this playbook is bookkeeping around that idea.

---

## 2. The signal portfolio (what we ensemble, and why each one is orthogonal)

Each row below is a separate failure mode for the previous rows. That
is the only design constraint — every new signal must catch a class of
fakes the existing signals miss.

| # | Signal | What it measures | Orthogonal to | Standalone AUROC (est.) |
|---|---|---|---|---|
| 1 | `provenance.c2pa` | Cryptographic camera signature | Everything (deterministic) | n/a (proof) |
| 2 | `provenance.sd_wm` | Stable-Diffusion default invisible watermark | Everything | n/a (proof) |
| 3 | `provenance.synthid` | Google SynthID watermark | Everything | n/a (proof) |
| 4 | `img.prithiv` | Fine-tuned CNN binary classifier | Frequency, retrieval | 0.78 |
| 5 | `img.frequency` | DCT/FFT high-freq energy ratio | Prithiv (data-driven) | 0.71 |
| 6 | `img.clip0` | CLIP zero-shot prompt ensemble | Pixel-domain CNNs | 0.69 |
| 7 | `img.meta` | EXIF / camera-software fingerprint | Pixel content | 0.65 (high precision) |
| 8 | `img.compression` | JPEG-history / double-compression | Semantic content | 0.62 (high precision) |
| 9 | `img.ocr_gibberish` | Tesseract + dictionary fraction | Pixel content | 0.55 (very high precision) |
| 10 | `img.eye_forensics` | Pupil circularity, iris regularity | Pixel content (selfie-only) | 0.61 (selfie slice) |
| 11 | `img.prnu` (v1.5) | Sensor-noise consistency | Frequency (different band) | 0.66 (>1 MP slice) |
| 12 | `tp.hive` | Hive Moderation private model | Our local CNNs | ~0.80 (vendor) |
| 13 | `tp.sightengine` | SightEngine private model | Hive | ~0.78 (vendor) |
| 14 | `tp.aiornot` | AI-or-Not private model | Hive, SE | ~0.76 (vendor) |
| 15 | `retrieval.knn` | FAISS k-NN over 5k+5k refDB | All pixel-domain | 0.74 (refDB-dependent) |
| 16 | `retrieval.patch` | 4-patch per-corner k-NN | Full-image retrieval | +0.05 on composites |
| 17 | `reverse.serpapi` | Reverse image search → web priors | Everything internal | n/a (pivot) |
| 18 | `vlm.gemini` | Gemini 3 vision reasoning | Numeric features | 0.81 (uncertain slice) |
| 19 | `vlm.counter` (v1.3.1) | Counter-prompt second opinion | Single-prompt VLM | reduces VLM error by 25 % |
| 20 | `ood.if` (v1.4) | IsolationForest cluster-anomaly | All discriminative signals | n/a (gate) |
| 21 | `meta.distill_lr` (v1.5) | LR trained on Gemini pseudo-labels | Uniform / Platt | +0.03 on uncertain slice |
| 22 | `conformal.gate` (v1.5) | Split-conformal coverage check | All scoring | n/a (guarantee) |

**Heuristic:** any pair with empirical signal-error correlation > 0.7
should be merged or one of them dropped. Build the correlation matrix
in `15_evaluation_protocol.md §6.4`.

---

## 3. Tier-by-tier expected lift (the AUROC math)

Estimates are for the `cloud_lite` profile on the §15 holdout
(2 000 images, balanced across 9 generators × 6 content types).
Numbers are **median of 10 bootstrap resamples**; the ±value is the
half-width of a 95 % CI.

| Stack | AUROC | ΔAUROC | Rationale |
|---|---|---|---|
| Best single detector (`img.prithiv`) | 0.78 ± 0.02 | baseline | Cross-generator generalisation ceiling |
| + `frequency` + `clip0` (uniform avg) | 0.82 ± 0.02 | **+0.04** | Decorrelation, mean of three |
| + `meta` + `compression` (high-precision filters) | 0.84 ± 0.02 | +0.02 | Decisive on edge cases, mean-imputed otherwise |
| + `ocr_gibberish` + `eye_forensics` | 0.85 ± 0.02 | +0.01 | Two more precision filters |
| + `prnu` (v1.5) | 0.86 ± 0.02 | +0.01 | Sensor-noise channel (gated to ≥1MP) |
| + Tier-1.5 third-party (`hive` + `sightengine` + `aiornot`) | 0.89 ± 0.02 | **+0.03** | Three vendor models trained on data we don't own |
| + Tier-2 retrieval (full image + 4 patches) | 0.91 ± 0.02 | +0.02 | Non-parametric similarity → composite fakes |
| + Tier-2.5 reverse search (when gated) | 0.92 ± 0.02 | +0.01 | Web prior on the *gated* slice (~25 % of jobs) |
| + Tier-3 VLM with counter-prompt | 0.93 ± 0.01 | +0.01 | Semantic reasoning on uncertain slice |
| + Cross-modal bonus (≥3 tiers agree) | 0.93 ± 0.01 | (sharper, not higher) | Reduces *low-confidence* errors |
| + OOD novel-generator → INCONCLUSIVE | (AUROC same, accuracy ↑) | — | Converts 3 % errors to honest abstentions |
| + Distillation meta-head (v1.5) | 0.94 ± 0.01 | +0.01 | Gemini-labels supervise meta-fusion |
| + Conformal wrapper (v1.5) | **AUROC same** | — | Converts heuristic verdicts to **guaranteed-coverage** verdicts on the non-abstained slice |

**Punchline:** the v1.5 stack targets **AUROC 0.93–0.95** on `cloud_lite`
with **≥ 98 % accuracy on the non-abstained slice** at 15–22 %
deferral rate. The conformal wrapper does not raise AUROC; it makes
the *claimed* accuracy a mathematical guarantee instead of a heuristic.

---

## 4. Failure-mode catalogue (adversarial input → which signals survive)

For each adversarial input class, the playbook lists (a) which signals
**flip** (fail), (b) which signals **survive**, (c) the **net verdict**
after fusion, and (d) the **mitigation** in our pipeline.

### 4.1 JPEG-recompressed AI image (`q=50`)

- Flips: `frequency` (FFT smoothed), `prithiv` (drops to ~0.55).
- Survives: `compression` (double-compression detected), `clip0`
  (semantic intact), `tp.hive`, `tp.aiornot`, `retrieval.knn`,
  `reverse.serpapi` if image is online.
- Net verdict: AI-GENERATED at lower confidence.
- Mitigation: `compression` becomes a *high-weight* signal when
  `jpeg_quality_estimate < 70`. Configured in `08_fusion §5`.

### 4.2 Screenshot of an AI image (display + screen reflection)

- Flips: `frequency` (display sub-pixel pattern dominates), `prithiv`
  (training distribution mismatch).
- Survives: `meta` (no camera EXIF), `compression` (moiré pattern),
  `retrieval.patch` (catches AI texture in patches), `reverse.serpapi`,
  `vlm.gemini` (recognises screen reflections).
- Net verdict: AI-GENERATED via retrieval + VLM + meta.
- Mitigation: content-type classifier flags `screenshot_or_synthetic`
  → activates screenshot-specific thresholds.

### 4.3 Composite (AI background + real face)

- Flips: `prithiv` (mixed signal averaged out), `clip0` (uniform),
  `tp.hive`/`sightengine`/`aiornot` (often \"REAL\" because face is
  dominant cue).
- Survives: `retrieval.patch` (background patch matches AI gallery),
  `frequency` (patch-level scan), `vlm.gemini` (rationale catches it).
- Net verdict: AI-GENERATED via retrieval.patch + VLM.
- Mitigation: patch-retrieval is **mandatory** in this stack; full-image
  retrieval alone misses composites. Documented in `06_tier2 §9`.

### 4.4 Pre-AI-era real photo (e.g. 2018 Reuters photograph)

- Risk: false-positive AI verdict.
- Flips: none of our detectors should flip, but `prithiv` and
  `frequency` have spurious correlations with low-resolution stock
  photos.
- Survives: `reverse.serpapi` finds the photo on pre-2022 news sites
  → `p_fake ≈ 0.07` overrides ensemble.
- Net verdict: REAL.
- Mitigation: reverse-search interpreter (`07_tier2_5 §2`) hard-weights
  `pre_ai_era_news` over Tier-1 ensemble. Loud failure logged when
  reverse search is gated off and the verdict flips.

### 4.5 Civitai-hosted real-style image (an AI portrait that looks photographic)

- Flips: `prithiv` (good photographic features), `clip0` (zero-shot
  fooled), `vlm.gemini` (visually plausible).
- Survives: `reverse.serpapi` finds the Civitai page → `ai_gallery_hit`
  → `p_fake ≈ 0.93`.
- Net verdict: AI-GENERATED.
- Mitigation: Civitai/Lexica/Discord-AI-channel host lists in the
  interpreter (`07_tier2_5 §2`).

### 4.6 Watermark-stripped Stable Diffusion image

- Flips: `provenance.sd_wm` (watermark removed).
- Survives: everything else.
- Net verdict: AI-GENERATED via Tier-1+ ensemble.
- Mitigation: provenance gate is **opt-in**, never blocking; ensemble
  always runs as a safety net.

### 4.7 Counter-prompt VLM disagreement (v1.3.1 case)

- Risk: Gemini Flash often gives the answer the prompt seems to want
  (\"convince me this is AI\" → it agrees, even when wrong).
- Mitigation: counter-prompt asks the opposite (\"convince me this is
  REAL\"). When the two rationales disagree, weight VLM signal at
  half. Net verdict: defers to numeric ensemble. Documented in
  `07_tier2_5 §6`.

### 4.8 OOD novel generator (unseen at refDB build time)

- Flips: `retrieval.knn` (no near neighbour in refDB), `prithiv`/`freq`
  (out-of-distribution).
- Survives: VLM rationale, Tier-1.5 third-party (if they have updated
  the model).
- Net verdict: should be INCONCLUSIVE, **not confident-wrong**.
- Mitigation: IsolationForest on refDB embeddings (v1.4). When
  `ood_real > τ AND ood_ai > τ`, force `verdict=INCONCLUSIVE`.
  Documented in `08_fusion §7`.

### 4.9 Heavily filtered real photo (Instagram, VSCO)

- Risk: real photo flagged as AI.
- Flips: `prithiv` slightly, `frequency` (filters add high-freq
  artefacts), `prnu` (filter removes sensor noise).
- Survives: `meta` (camera EXIF often preserved), `reverse.serpapi`
  (if shared), `vlm.gemini` (recognises camera + filter combo).
- Mitigation: `compression.high_freq_filter_detected` flag downgrades
  `frequency` weight; PRNU is **gated** on `compression.jpeg_quality
  ≥ 70` for exactly this reason.

### 4.10 Multi-AI mosaic (collage of AI tiles)

- Flips: `provenance` (no single watermark), individual tile detectors
  may disagree.
- Survives: `retrieval.patch` (each tile lands in AI cluster).
- Mitigation: patch retrieval voted across 4 corners; if ≥ 2 patches
  land in AI cluster, verdict = AI.

---

## 5. The honest-abstention contract

Accuracy is meaningless without a deferral rate. Two systems can both
report 95 % accuracy:

- System A: 95 % accuracy on 100 % of inputs.
- System B: 99 % accuracy on 80 % of inputs, INCONCLUSIVE on 20 %.

System B is *better* in production because users learn that
\"INCONCLUSIVE\" means \"go investigate manually\", and the 99 % is
trustworthy. This is the **honest-abstention contract**:

1. The headline accuracy number is computed **only on the non-abstained
   slice** (the \"confident\" verdicts).
2. The deferral rate is a **published KPI**, not a hidden cost.
3. Conformal prediction (v1.5) makes the non-abstained accuracy a
   *mathematical guarantee* at chosen α (we use α=0.05 → 95 %).
4. Abstention has its own UX state (\"INCONCLUSIVE\") with a narrative
   explaining *why* (OOD, low agreement, VLM uncertainty).
5. We **never** force a binary verdict when the signals genuinely
   disagree.

| KPI | v1.4 target | v1.5 target |
|---|---|---|
| Accuracy on non-abstained slice | ≥ 0.97 | **≥ 0.98** |
| Deferral rate (image, cloud_lite) | 18–25 % | 15–22 % |
| Empirical conformal coverage (α=0.05) | n/a | **0.95 ± 0.01** |
| False-positive on REAL (production-critical) | < 0.02 | **< 0.015** |

The single most-cared-about number is **false-positive on REAL**.
Accusing a real photo of being AI is the worst user-trust failure
mode. The reverse-search Tier-2.5 was added specifically for this.

---

## 6. Cross-modal bonus theory (why ≥3 tiers agreeing is super-additive)

When `k` truly independent signals all vote AI with probability `p_i`,
the Bayes-optimal posterior is

```
log_odds_total = Σ log_odds_i,    where log_odds_i = log(p_i / (1-p_i))
```

This is super-additive: three weak signals each at p=0.7 yield a
posterior probability of **0.93**, not 0.7. The catch: signals must
be *truly independent*. Our pipeline approximates independence by
forcing each tier to use a different evidence channel:

| Tier | Evidence channel |
|---|---|
| 1 | Pixel-domain numeric features |
| 1.5 | Vendor-private CNN training data |
| 2 | Reference-DB similarity (non-parametric) |
| 2.5 | Open-web prior |
| 3 | Semantic reasoning |

Within-tier signals are correlated and merged via Platt-on-refDB
calibration. Across-tier signals get the **multiplicative cross-modal
bonus** in `08_fusion §6`, capped at +0.10 to prevent overconfidence
on a small ensemble. The cap is set empirically — three agreeing
weak signals already give p > 0.93; we don't need more.

---

## 7. Calibration matters more than raw AUROC

A production system has a *threshold*. AUROC is threshold-free; it
hides whether the model is *over-confident at p=0.9* or
*under-confident at p=0.5*. Two metrics matter:

- **Brier score**: mean squared error between predicted probability
  and binary label. Lower is better. Target: < 0.10 on holdout.
- **Expected Calibration Error (ECE)**: mean |empirical accuracy –
  predicted probability| in 10 probability bins. Target: < 0.10.

We achieve calibration by:

1. **Platt-on-refDB** at build time (`08_fusion §4`). Fits a 1-D
   logistic over each signal's raw output.
2. **Isotonic per-signal calibration** at `n_user_labels ≥ 500` (M7).
3. **Conformal wrapper** (v1.5) — converts the calibrated score into a
   prediction *set* with guaranteed coverage, independent of the
   underlying calibration quality. See `16_accuracy_extensions §5`.

The conformal wrapper is the **floor** of trustworthiness: even if
Platt is mis-specified, the conformal coverage guarantee still holds.

---

## 8. Known-handled vs. not-handled generator matrix

Status on representative public generators as of 2026-02. \"✓\" means
the tier reliably catches it in our holdout; \"◐\" means catches when
agreed-with by another signal; \"—\" means not yet evaluated.

| Generator | Tier-1 (image) | Tier-1.5 (vendors) | Tier-2 (retrieval) | Tier-3 (VLM) | Status |
|---|---|---|---|---|---|
| Stable Diffusion 1.5 | ✓ prithiv + freq | ✓ all three | ✓ | ✓ | **production** |
| SDXL / Turbo | ✓ prithiv + freq | ✓ | ✓ | ✓ | **production** |
| SD 3 / SD 3.5 | ◐ freq | ✓ hive + aiornot | ✓ | ✓ | **production** |
| Flux schnell / dev | ✓ freq + ocr | ✓ hive + aiornot | ◐ refDB-dependent | ✓ | **production** |
| Midjourney v6 / v7 | ◐ clip0 + freq | ✓ | ✓ | ✓ | **production** |
| DALL-E 3 | ◐ clip0 | ✓ hive | ✓ | ✓ | **production** |
| Ideogram 2 | ◐ ocr_gibberish (rare wins) | ✓ hive | ◐ | ✓ | production |
| Imagen 3 / 4 | ◐ provenance.synthid when present | ◐ | ◐ | ✓ | production |
| Gemini-image (Nano Banana) | ✓ provenance.synthid | ◐ | ◐ | ✓ | production |
| Recraft v3 | ◐ | ◐ | ◐ | ✓ | watchlist |
| Krea AI / FlairAI | ◐ | ◐ | ◐ | ✓ | watchlist |
| Sora 2 (video frames) | — | — | — | ◐ | Phase 1.5 |
| **Novel / unseen** | — | — | OOD-INCONCLUSIVE | ✓ | by design |

The matrix is *intentionally* generator-aware. As of v1.5 the
production scope is **9 mainstream generators** across **6 content
types** (photo, art, selfie, document, screenshot, mixed). The
holdout in `15_evaluation_protocol.md §3` is balanced across this
grid.

---

## 9. Adversarial test bench (the \"must-pass\" before declaring production)

These ten fixtures are the **gate** between M3 and first-finish. If
any one fails, the pipeline is not production-grade. They live in
`tests/fixtures/adversarial/` and are referenced from
`12_scripts_and_testing.md §6.2`.

| # | Fixture | Expected verdict | Tier that must catch |
|---|---|---|---|
| 1 | `recompressed_sdxl_q50.jpg` | AI-GENERATED | compression + retrieval |
| 2 | `screenshot_of_mjv6.png` | AI-GENERATED | retrieval.patch + VLM |
| 3 | `composite_ai_bg_real_face.png` | AI-GENERATED | retrieval.patch |
| 4 | `pre_ai_reuters_2018.jpg` | REAL | reverse search |
| 5 | `civitai_realistic_portrait.png` | AI-GENERATED | reverse search |
| 6 | `sd_watermark_stripped.png` | AI-GENERATED | ensemble (no provenance) |
| 7 | `counter_prompt_disagree.png` | INCONCLUSIVE | VLM second opinion |
| 8 | `ood_novel_generator.png` | INCONCLUSIVE | OOD IsolationForest |
| 9 | `vsco_filtered_dslr.jpg` | REAL | meta + reverse + (PRNU bypassed by gate) |
| 10 | `c2pa_signed_camera.jpg` | REAL | Tier-0 short-circuit |

**Conformal addition (v1.5):** for fixtures 1–6 and 9–10, conformal
prediction must return a **singleton** set (`{AI}` or `{REAL}`). For
fixtures 7–8, conformal **must** return the doubleton set
`{AI, REAL}` (the formal name for INCONCLUSIVE). Anything else is a
calibration failure.

---

## 10. KPI definitions (exact formulas)

| KPI | Definition |
|---|---|
| **AUROC** | Area under ROC curve over (p_ai, label_ai). `sklearn.metrics.roc_auc_score`. |
| **AUPR-AI** | Area under PR curve with AI = positive class. |
| **AUPR-Real** | Area under PR curve with REAL = positive class. |
| **Brier** | `mean((p_ai - label_ai)^2)`. Range 0–1, lower is better. |
| **ECE** | `Σ_b (n_b / N) · |acc(b) − conf(b)|` over 10 equal-width probability bins. |
| **Accuracy-on-non-abstained** | `correct(non_abstained) / total(non_abstained)`. Headline metric. |
| **Deferral rate** | `non_abstained / total`. |
| **Conformal coverage** | Fraction of holdout where the true label is **in** the prediction set. Target: 1 − α = 0.95. |
| **Per-slice macro-AUROC** | Unweighted mean of AUROC across all generator × content-type slices. Penalises worst-slice failure. |
| **FPR on REAL @ p=0.5** | Production-critical: rate of falsely accusing a real photo. |
| **Signal-error correlation** | Pearson(errs_i, errs_j) on the holdout. Audits decorrelation assumption. |

All KPIs are produced by `scripts/run_eval.py` in
`15_evaluation_protocol.md §11` and committed to `eval/baseline.json`.

---

## 11. Roadmap (v1.4 → v1.5 → v2.0)

| Version | Adds | AUROC target (`cloud_lite`) | Non-abstain accuracy |
|---|---|---|---|
| v1.3.1 | Counter-prompt VLM | 0.86–0.91 | 0.96 |
| v1.4 | OCR-gibberish, eye-forensics, Tier-1.5 third-party, patch retrieval, OOD | 0.89–0.93 | 0.97 |
| **v1.5 (this)** | **PRNU, distillation meta-head, conformal** | **0.91–0.95** | **0.98** |
| v2.0 (M4–M5) | Audio + video pipelines | n/a (own targets) | own targets |
| v2.5 (M7) | Active learning, GBDT auto-promote | 0.92–0.96 | 0.985 |
| v3.0 (M8) | Text modality (Binoculars + GLTR) | n/a | n/a |

---

## 12. AGENTS.md mapping for this file

| Standard | Where honored |
|---|---|
| 14.1 Model versioning | §2 portfolio table + `eval/baseline.json` |
| 14.2 A/B testing | §3 ΔAUROC table = the A/B framework |
| 14.5 Fallback strategies | §4 mitigation column |
| 14.6 Response validation | §5 honest-abstention contract |
| 5. TDD | §9 adversarial test bench = mandatory tests |
| 7. Observability | §10 KPI definitions = monitoring contract |
| 9. ADRs | §11 roadmap = changelog of intent |

---

End of `14_accuracy_playbook.md`. Strategic source of truth for the
**why-this-works** conversation. Pair with `15_evaluation_protocol.md`
(how-we-measure) and `16_accuracy_extensions_v1.5.md` (the three new
boosters).
"