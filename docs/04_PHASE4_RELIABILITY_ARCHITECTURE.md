# PHASE 4 — Evidence Reliability Architecture

> The defining design problem: a forensic system must know *when its own evidence is untrustworthy*. This phase specifies the module contract, the reliability-estimation machinery (ARGUS's core novelty), degradation handling, and contradiction management.

---

## 4.1 The Module Output Contract

Every Tier-2 module emits exactly this structure:

```json
{
  "module_id": "spectral_probe",
  "version": "1.3.0",
  "evidence_score": -0.62,
  "reliability_score": 0.81,
  "confidence_score": 0.74,
  "verdict_direction": "synthetic",
  "artifacts": [
    {
      "type": "spectral_peak",
      "description": "Periodic energy peaks at f/4 and f/2 radial frequency, consistent with 4x upsampling; not explained by detected resize history",
      "location": null,
      "strength": 0.71,
      "visual": "artifacts/spectrum_polar_0042.png",
      "checkable_claim": "FFT radial power spectrum shows peaks at normalized frequencies 0.25, 0.5 exceeding calibrated real-image envelope by >3σ"
    }
  ],
  "unavailable_reason": null,
  "compute_ms": 14
}
```

### Why three separate scores — the semantics

The three numbers answer three *different questions* and must never be collapsed prematurely:

- **`evidence_score` ∈ [−1, +1] — "What does the evidence say?"**
  Signed: −1 = strongly indicates synthesis/manipulation, +1 = strongly indicates camera-original, 0 = neutral. This is the witness's testimony.

- **`reliability_score` ∈ [0, 1] — "Can this witness see clearly *for this image*?"**
  A property of the *measurement conditions*, not of the testimony. Computed primarily from the degradation state `d` (and module-specific availability checks: "no text found" → typography module reliability ≈ 0 with `unavailable_reason` set). Crucially, reliability is computable **even when the evidence score is garbage** — that is the point.

- **`confidence_score` ∈ [0, 1] — "How internally certain is the module of its own reading?"**
  A property of the *measurement itself*: margin from decision boundaries, agreement across image crops/scales, variance under test-time augmentation, kNN-neighborhood tightness (module E), per-pixel confidence-map aggregates (module D).

**Worked example of why the separation matters.** A pristine RAW with a borderline spectral reading: `e=−0.1, r=0.97, c=0.3` → "an excellent witness who genuinely isn't sure" → fusion treats as weak-but-honest signal. A 5×-recompressed thumbnail with a strong spectral reading: `e=−0.8, r=0.15, c=0.9` → "a confident witness who couldn't possibly have seen" (resize laundering manufactures spectral peaks) → fusion discounts to near-zero. **A single fused number cannot represent both cases; this distinction is exactly what classifiers lose (failure F7).**

---

## 4.2 Estimating Reliability: Degradation-Conditioned Calibration

This is the mechanism ARGUS contributes that the literature lacks (closest precedent: TruFor's per-pixel confidence map — but that is per-module and not conditioned on a measured laundering state).

### 4.2.1 The degradation state vector

Tier-1 triage outputs:

```
d = (q_jpeg,        # estimated last JPEG quality (quant-table + DCT-histogram)
     n_gen,         # estimated recompression generations (double-quantization analysis)
     s_resize,      # estimated resampling factor (spectral peak detection)
     p_screenshot,  # screenshot probability (UI/aspect/quant heuristics + small CNN)
     z_denoise,     # denoising/smoothing estimate (local noise-floor statistics)
     res_eff,       # effective resolution after detected upscaling
     chroma)        # chroma-subsampling profile
```

Each component carries its own uncertainty band (triage is itself a measurement).

### 4.2.2 Offline: learning reliability curves

For each module `m`, ARGUS learns `r_m(d)` = the probability that the module's evidence direction is correct, given degradation state `d`:

1. **Laundering simulator.** A deterministic, versioned pipeline of composable ops: JPEG(q ∈ {95..30}) × resize(0.25–1.0, bilinear/lanczos) × screenshot-simulation (render-at-scale + UI crop + re-encode) × platform presets measured from real round-trips (WhatsApp/Telegram/X/Instagram export profiles) × sharpening/denoising.
2. **Sweep.** Pass the labeled calibration corpus (reals + fakes; Phase 8 datasets) through the ladder; at each rung, record every module's evidence score against ground truth.
3. **Fit.** `r_m(d)` = a small monotonic-constrained GBM or binned isotonic table mapping `d` → empirical directional accuracy. Monotonicity (reliability non-increasing in degradation severity) is enforced where physically justified — this regularizes the small calibration set and makes the curves auditable.
4. **Ship** the curves as versioned lookup artifacts. They are tiny (KBs), fast (µs query), and inspectable — one can literally plot "Noiseprint reliability vs JPEG quality" and check it against forensic intuition.

### 4.2.3 Online: querying reliability

At inference: triage → `d̂` (with uncertainty) → each module queries `r_m(d̂)`, taking the **lower confidence bound** when `d̂` is itself uncertain (conservative by construction). Module-specific availability checks can only lower it further (no shadows found → module G reliability ~0).

### 4.2.4 Why this beats the alternatives

| Alternative | Why rejected |
|---|---|
| Train each module end-to-end on augmented (laundered) data | Helps but conflates robustness with awareness: the module gets *somewhat* better under degradation but still cannot *report* that it's blind. Also requires retraining big components — budget violation. ARGUS does both where cheap (augmented heads) but the reliability curve is the load-bearing mechanism |
| Single global "image quality" scalar | Different modules die at different rungs: spectral probe dies at resize, Noiseprint at recompression, physics at nothing. Reliability must be per-module, hence conditioned on the *vector* `d` |
| Let the fusion layer figure it out implicitly | Works only on degradation states seen in fusion training; an explicit, physically-structured `r_m(d)` extrapolates better, is auditable, and survives fusion retraining |

---

## 4.3 How Degraded Images Flow Through the System

Walkthrough — a screenshot of a WhatsApp-forwarded photo:

1. **Triage:** `q_jpeg≈62, n_gen≈3, s_resize≈0.4, p_screenshot=0.93, res_eff=540px`.
2. **Reliability collapse, selectively:** CFA-adjacent and residual evidence: `r_D ≈ 0.1`; spectral probe `r_C ≈ 0.2` (resize manufactures peaks); metadata `r_A ≈ 0.3` (stripped — but the *stripping pattern* feeds `d`); DINOv2 probe `r_E ≈ 0.7` (robust representations); physics `r_G ≈ 0.85`; semantics `r_H ≈ 0.9`.
3. **Gating:** modules below floor (default r < 0.25) are excluded and *listed in the verdict as unavailable, with reasons* — the verdict says "sensor-level evidence unavailable: image has been recompressed ≥3×", which is forensic honesty no current tool offers.
4. **Fusion** proceeds on the surviving panel; conformal sets widen automatically (calibration includes degraded strata); the verdict's **trust_score** (Phase 7) reflects thinner evidence.
5. **Outcome:** either a verdict carried by laundering-resistant evidence, or a *guaranteed-coverage abstention*: "insufficient evidence — image too degraded for reliable assessment; here is what little the surviving evidence suggests."

**Adversarial corollary (the strategic win):** an adversary who launders harder pushes ARGUS toward abstention, not toward a wrong "authentic" verdict. Laundering attacks downgrade from *exploits* to *denial-of-evidence*, which the system reports as exactly that — and aggressive laundering of a contested image is itself a flag a human analyst reads correctly.

---

## 4.4 Contradiction Handling

Contradictions among reliable witnesses are **information, not noise** — often the most diagnostic signal in the panel.

### 4.4.1 Detecting conflict

Conflict metric over gated modules: `K = Σ_{i<j} r_i·r_j·max(0, −e_i·e_j)` normalized to [0,1] — reliability-weighted directional disagreement (a Dempster-Shafer-flavored conflict mass used for *diagnosis*, not for fusion).

### 4.4.2 Interpreting conflict — the patterns are meaningful

| Pattern | Likely meaning | System response |
|---|---|---|
| Statistical modules say *synthetic*, physics/semantics say *clean* | High-quality full generation (statistics see it; physics has nothing to catch) — **or** statistical false positive on unusual real content (long-exposure, microscopy) | Check module E's neighborhood report: semantically-unusual-but-forensically-normal vs forensically-anomalous; route accordingly |
| Compression-history finds a localized anomaly; whole-image probes say *real* | **Local manipulation of a real photo** (splice/inpaint) — the verdict should be "manipulated", not "AI-generated" | Promote the `manipulated` hypothesis; attach localization maps |
| Provenance valid; statistical modules say *synthetic* | Signed AI content (e.g., OpenAI-signed) — consistent, not conflicting | Verdict: "AI-generated, *disclosed via provenance*" — highest-confidence case |
| Metadata claims camera X; residual/noise inconsistent with any camera pipeline | Metadata forgery | Raise `risk_score`; flag metadata as adversarially planted |
| High conflict, no recognizable pattern | Genuine ambiguity or novel attack | Conflict feeds the conformal nonconformity score → abstention; case queued for human review and corpus collection |

### 4.4.3 Architectural consequences

1. `K` and pattern-match features are **inputs to the fusion layer** (the GBM learns conflict-conditional behavior) and to the nonconformity score (high unexplained conflict → wider prediction sets → abstention).
2. The verdict schema (Phase 7) has a mandatory `contradictions[]` section — contradictions are *disclosed*, never averaged into silence. A fused 0.5 from {+0.9 reliable, −0.9 reliable} and a fused 0.5 from {everything ≈ 0.5} are entirely different epistemic states; the verdict must distinguish them (this is precisely what weighted averaging destroys, see Phase 5).
3. Persistent novel conflict patterns in production telemetry are the **drift alarm** (Phase 6): a new generator that defeats statistical modules while failing physics shows up first as a conflict-pattern shift.
