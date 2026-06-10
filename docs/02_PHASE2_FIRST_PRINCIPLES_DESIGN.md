# PHASE 2 — First-Principles Design

> Rule for this phase: no existing deepfake architecture is assumed correct. Every component must be derived from axioms, and every component must answer: why it exists, what problem it solves, its failure modes, its computational cost, and its long-term robustness.

---

## 2.1 Axioms

Derived from Phase 1's causal analysis; everything else follows from these.

- **A1 — Authenticity is causal history, not appearance.** The question is "what process produced this file?" Pixels are evidence about that process, never proof. → The system is an *estimator of process from evidence*, not a classifier of pixels.
- **A2 — The fake class is open; the real class is (more) stationary.** → Model realness; treat fake-trained components as perishable plugins with expiry dates.
- **A3 — Evidence quality is a per-image latent variable.** A thumbnail and a RAW are different *measurement instruments*, not just different inputs. → Estimate evidence-carrying capacity *first*; condition everything on it.
- **A4 — No single evidence stream is sufficient or always available.** → Heterogeneous panel; graceful degradation; fusion that knows what's missing.
- **A5 — Abstention is a first-class output.** Cost asymmetries and OOD inputs make "I don't know, and here's why" the correct answer for a meaningful fraction of traffic. → Formal abstention (conformal), not a softmax threshold.
- **A6 — Every claim must be checkable.** A verdict that cannot survive cross-examination is worthless in journalism, courts, and T&S escalation. → Evidence must be named, localized, and independently verifiable wherever possible.
- **A7 — The adversary reads your paper.** → Prefer evidence that is expensive for the adversary to fake (physics, provenance, external context) over evidence that is cheap to wash out (high-frequency statistics).

### Assumptions of existing architectures — challenged

| Standard assumption | Verdict | Replacement |
|---|---|---|
| "Detection = binary classification" | **Rejected** (A1, L3) | Multi-hypothesis process estimation: {camera-original, AI-generated, manipulated/composite, indeterminate} |
| "More fake training data → better generalization" | **Rejected** (L1) | More *real* data diversity → better realness model; fakes used mainly for fusion calibration |
| "One end-to-end network is optimal" | **Rejected** (F1, F5) | A panel of small, independent, named evidence modules |
| "Confidence = softmax output" | **Rejected** (F6) | Separated evidence/reliability/confidence + conformal sets |
| "Process every image the same way" | **Rejected** (A3) | Degradation triage gates the pipeline |
| "Fine-tune a foundation model on fake/real" | **Rejected** (budget + L2 + CLIP's 42% laundering collapse) | Frozen DINOv2 features + lightweight heads on top |
| "Localization heatmap = explanation" | **Partially rejected** | Heatmaps are *one artifact type* inside a structured verdict, not the explanation |

---

## 2.2 The Derived Architecture: Three Tiers + a Fast Path

```
Tier 0  Provenance fast-path     (microseconds–ms)
Tier 1  Degradation triage       (ms)            → degradation state d
Tier 2  Evidence panel           (ms–seconds)    → {e, r(d), c} per module
Tier 3  The Court                (ms)            → verdict, abstention, explanation
```

The tiers are ordered by **cost and certainty**: cheap/near-certain evidence first, expensive/uncertain evidence later, judgment last. This is the structure of an actual forensic investigation, which is not a coincidence — forensics solved "open-set process estimation under variable evidence quality" decades before deep learning met it.

---

### Tier 0 — Provenance Fast-Path

- **Why it exists:** cryptographic provenance is the only evidence class fully orthogonal to generator progress (A7). When a valid C2PA manifest chains to a trusted capture device, the authenticity question is largely *answered*, at ~zero compute.
- **Problem it solves:** wasted compute and unnecessary uncertainty on the (growing, 2026) fraction of signed media; also detects the inverse signal — a manifest that *fails* validation is strong evidence of tampering.
- **Failure modes:** (1) absence is weak evidence — most legitimate images are unsigned in 2026, so absence must contribute ~nothing to the verdict; (2) stripped credentials; (3) known spec/implementation gaps: revoked-certificate handling, unverifiable media after cert expiry (2026 analyses); (4) "signed AI" — OpenAI signs its generations, so a valid manifest can *prove* AI origin (this is a feature: it resolves the hypothesis, just not toward "camera-original"); (5) trust-list governance: who decides which signers are trusted?
- **Cost:** milliseconds (signature verification, `c2pa-python`).
- **Long-term robustness:** **the best in the system** — improves as adoption grows; immune to generator progress. The fast-path's *share of traffic* grows monotonically with ecosystem adoption.

### Tier 1 — Degradation Triage

- **Why it exists:** A3. Before asking "is it authentic?", ask "how much can this file still testify?" Without this, the system is confidently wrong on laundered inputs — failure mode F4/F7, the dominant production killer.
- **Problem it solves:** silent blindness. Outputs a **degradation state vector** `d` = (estimated JPEG quality + recompression generations, resampling/resize factor, screenshot likelihood, denoising/filtering estimate, effective resolution, color-subsampling profile). Every Tier-2 module's reliability is then computed as `r_m(d)` — a calibrated function learned offline (Phase 4).
- **How:** quantization-table inspection + DCT-histogram double-quantization analysis (JPEG generations); spectral resampling-peak detection (resize); UI-furniture/aspect-ratio/quant-fingerprint heuristics + a small classifier (screenshot); local-noise-floor statistics (denoising).
- **Failure modes:** (1) novel pipelines (new platform codecs, AVIF/HEIC re-encodes) misestimated → mitigated by treating `d` itself with uncertainty bands and by continuous re-calibration against a maintained laundering simulator; (2) adversary deliberately launders to *force* low reliability → but this drives the system toward **abstention, not error** — degrading an adversary's attack from "fool the system" to "make the system say 'insufficient evidence'", which is an enormous strategic improvement; (3) pristine synthetic images claiming to be camera-original PNGs — handled because *implausible cleanliness is itself evidence* (a "smartphone photo" with no JPEG history and no sensor noise is suspicious).
- **Cost:** tens of milliseconds, pure CPU (DCT statistics, FFT, small CNN ≤1M params).
- **Long-term robustness:** high — it measures *transmission damage*, which is governed by codecs and platforms, not by generators. Codec churn is slow and trackable.

### Tier 2 — The Evidence Panel

Nine modules, deliberately heterogeneous across the evidence families of Phase 3. Common contract (full schema in Phase 4): each emits `{evidence_score ∈ [-1,+1], reliability_score ∈ [0,1], confidence_score ∈ [0,1], artifacts[]}` where artifacts are named, localized, checkable findings.

Selection principle: **maximize evidence diversity per FLOP, prefer Tier S/A longevity (Phase 1.4), require explainability (A6).**

| Module | Evidence family | Why this one | Failure modes | Cost (CPU) | Longevity |
|---|---|---|---|---|---|
| **A. Metadata & container** | Provenance-adjacent | EXIF/XMP/quant-tables/thumbnail consistency; software-history strings; "born-digital" markers. Cheap, explainable, surprisingly discriminative for sloppy fakes | Trivially forgeable by sophisticated adversary → contributes asymmetrically (anomalies are strong; cleanliness is weak) | ~ms | Medium (forgeable but cheap to keep) |
| **B. Compression-history** | Degradation forensics | ELA + JPEG-ghost + double-quantization maps localize *spliced/inpainted regions* with different compression history | Whole-image generations have uniform history → blind to them (other modules cover); heavy recompression washes ghosts | ~10–50 ms | Medium-high for manipulation; n/a for full generation |
| **C. Spectral probe** | Frequency statistics | FFT/DCT radial-spectrum + peak analysis for upsampler fingerprints; SpAN-style power calibration. Training-free → cannot overfit to generators | Each generator generation cleans its spectrum; resize laundering creates *its own* peaks (must be disambiguated using `d`) | ~10 ms | Medium, declining — kept because it is nearly free and currently strong |
| **D. Learned noise residual** | Sensor/pipeline statistics | TruFor-class Noiseprint++ residual + anomaly + **built-in confidence map** — the strongest open localizer for splices; residual *uniformity* also signals full generation | Compression suppresses residuals (reliability conditioning is essential — this is the module `r_m(d)` helps most); research-only license needs checking (Phase 10) | ~1–2 s | Medium |
| **E. Real-distribution probe** | Realness modeling | Frozen **DINOv2 ViT-B/14** embeddings + kNN/one-class scoring against a large *real-only* reference set. The A2 flagship: never saw a fake, cannot overfit to one. DINOv2 chosen over CLIP for its ~92% vs ~42% robustness under transformations | Real-distribution drift (new phone pipelines look "unreal"); semantic OOD (microscopy, art) scores as anomalous → must report *why* it's anomalous (semantic vs forensic neighborhood) | ~1–3 s CPU | **High** — degrades gracefully, never inverts |
| **F. Perturbation-sensitivity probe** | Training-free generative trace | RIGID/2026-style: measure representation sensitivity to structured (high-frequency/Fourier) perturbations in a frozen ViT; AI images sit in flatter/more curved regions of representation space than camera images. Zero training, one forward pass + one FFT | Signal partially high-frequency → laundering-sensitive (conditioned via `d`); thresholds need per-domain calibration | ~1–2 s CPU (shares backbone with E) | High among statistical methods (no generator supervision) |
| **G. Physics & geometry** | Physical consistency | Shadow-to-light-source line intersection; reflection vanishing-point consistency; projective-geometry checks. Low-frequency → **laundering-resistant**; exploits generators' lack of ray-tracing (Tier A longevity); supremely explainable (you can *draw the lines*) | Low recall (needs visible shadows/reflections); cluttered scenes; soft lighting; automation of classical manual techniques is genuinely hard → MVP scopes this narrowly (Phase 8) | ~0.5–2 s | **High** |
| **H. Semantic plausibility** | World-logic | Hand/finger topology, garbled typography (OCR + lexicon check), object-interaction sanity — the SemaFor class. Catches what statistics miss; extremely persuasive artifacts for human reviewers | Generators improve here visibly each year; risk of stereotyping "AI style" → keep evidence weights re-calibrated quarterly | ~1–2 s | Medium, declining but long-tailed |
| **I. Retrieval & context** *(optional at MVP)* | External context | Reverse-image/near-duplicate search: earliest appearance, crop ancestry, known-fake corpora match. Generators cannot fabricate *history* (Tier S) | Needs index infrastructure / third-party APIs; coverage gaps for genuinely new images | network-bound | **Highest** with provenance |

**Why a panel and not one network:** independence of failure (F1), named evidence (A6), per-module reliability conditioning (A3), perishability management (modules C/H can be re-weighted or retired without touching the rest — the architecture *expects* evidence to die, per A2/L2).

### Tier 3 — The Court

- **Why it exists:** nine noisy, partially contradictory, variably reliable opinions are not a verdict. Something must weigh them *per image*, knowing each witness's current reliability.
- **Problem it solves:** principled fusion + calibrated uncertainty + structured output. Design (defended fully in Phase 5): (1) **reliability gate** — modules with `r_m(d)` below floor are excluded and *listed as unavailable in the verdict*; (2) **calibrated stacking** — LightGBM over the feature vector `[e_m, r_m, c_m, d]`, trained on a few thousand fused examples; learns interactions like "spectral peak + no resize in `d` → strong; spectral peak + resize detected → discount"; (3) **isotonic calibration** → honest probabilities; (4) **conformal wrapper** → prediction *sets* over {camera-original, AI-generated, manipulated} with user-set risk α; set size > 1 → ABSTAIN with guaranteed coverage; (5) **verdict builder** → Phase 7 schema.
- **Failure modes:** fusion-set bias (the GBM is the one trained-on-fakes component → kept tiny, regularized, monotonicity-constrained where sensible, and retrained quarterly — *small and cheap to retrain* is the point); conformal exchangeability violated under distribution shift (mitigated: per-domain calibration sets + drift monitoring, Phase 6/10).
- **Cost:** ~ms.
- **Long-term robustness:** the *mechanism* is robust; its calibration is consumable and scheduled for refresh — explicitly budgeted, not hoped away.

---

## 2.3 End-to-End Properties

| Property | Achieved by |
|---|---|
| Total latency (CPU, all modules) | ~3–6 s/image; Tier 0/1 short-circuits cheaper cases; E+F share one DINOv2 forward pass |
| Hardware | Single commodity machine, no GPU required; GPU optional 5–10× speedup |
| Graceful degradation | Reliability gating: a thumbnail yields a verdict from G+H+A only — with wider, honest uncertainty |
| Adversarial posture | Cheapest attacks (laundering) → abstention, not error; expensive attacks (physics-consistent, provenance-faking) raise adversary cost by orders of magnitude (A7) |
| Maintainability | Modules are independently replaceable; fusion retraining is cheap by design |
