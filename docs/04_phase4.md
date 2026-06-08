# PHASE 4 — Evidence Reliability

> *The defining innovation of ARGUS. Every module is forced to report not just what it found, but how much it should be believed given this specific image.*

## 4.1 The module output contract

Every evidence module MUST emit a structured tuple — no bare scores allowed:

```json
{
  "module": "jpeg_double_quant",
  "evidence_score":    0.72,   // ∈ [0,1]  P(authentic) implied by THIS cue, 0.5 = neutral
  "reliability_score": 0.31,   // ∈ [0,1]  how much the image LET this module work
  "confidence_score":  0.66,   // ∈ [0,1]  how internally sure the module is in its reading
  "direction":        "authentic|synthetic|tamper|neutral",
  "localization":      <optional heatmap / bbox>,
  "rationale":        "single coherent JPEG grid; no double-quantization peaks",
  "features":         { ...raw measurements for audit... }
}
```

### The crucial distinction: reliability vs. confidence
These are independently necessary and constantly confused in the literature:

- **Reliability** is a property of the **evidence channel given the image**. *"Can this cue even be read here?"* A 480p screenshot has **zero PRNU reliability** no matter what the PRNU module computes — the physical signal is gone. Reliability is estimated largely from the **quality profile**, not from the module's own output.
- **Confidence** is a property of the **module's reading given that the channel works**. *"Given a readable image, how sure am I of this measurement?"* A clean image with an *ambiguous* lighting estimate is high-reliability, low-confidence.

> Example that breaks every incumbent system: a pristine RAW with perfectly consistent PRNU → reliability ≈ 1.0, confidence ≈ 0.9, evidence ≈ 0.85 (authentic). The *same scene* as a meme screenshot → reliability ≈ 0.05 (silence the cue), even if the module naively still outputs evidence 0.85. ARGUS will *ignore* the second PRNU reading because it knows the channel is dead. Incumbents would trust it.

## 4.2 How reliability should be estimated

Reliability is **not** guessed by the module from its own answer (that would be circular and overconfident). It is computed from *external* preconditions:

**(a) Quality-conditioned reliability (the dominant term).**
Each module declares the image properties its signal *physically depends on* and a **degradation transfer function** mapping the quality profile → maximum achievable reliability. Examples:
- PRNU/sensor noise: reliability ∝ surviving high-frequency energy; collapses past JPEG-Q≈85, downscale >1.5×, or any screenshot flag.
- CFA/demosaicing: requires near-native resolution and ≤1 JPEG generation.
- Metadata: reliability = 1 if a rich, internally consistent EXIF/XMP block exists; ≈0 if stripped.
- Frequency peaks: reliability decays with resampling/blur.
- Physics/semantic/retrieval: reliability is *high and laundering-robust* (geometry and identity survive compression) — this is *why* they are the backbone.

These transfer functions are **calibrated empirically**: take authentic images, apply a grid of degradations (JPEG-Q sweep, downscale, screenshot simulation, double-compression), and measure each module's *agreement with ground truth* as a function of the quality vector. The fitted curve **is** the reliability function. (This is a small, cheap, fully self-supervised calibration — no proprietary data.)

**(b) Self-consistency reliability.** Modules producing spatial maps (PRNU, CFA, JPEG, noise) estimate reliability from *internal spatial agreement* — a uniformly consistent map is more reliable than a noisy speckled one (SNR of the evidence map).

**(c) Out-of-distribution reliability (open-set term).** Each learned module ships with a model of its *own training distribution* (e.g., a density/energy score on its feature space, or a kNN distance to training features). If the current image's features are far outside that manifold, reliability is discounted — the module is extrapolating. This is what neutralizes Tier-C fingerprint modules on unseen generators (Phase 6).

**Final per-module reliability** = `min(a) · g(b) · g(c)` — a conservative product, because any one broken precondition should silence the channel. (Min/product, not average: a single dead precondition kills the cue.)

## 4.3 How degraded images affect reliability (worked behavior)

| Image condition | High-reliability modules | Silenced (low-reliability) modules | System behavior |
|---|---|---|---|
| Pristine RAW / single-JPEG, rich EXIF | PRNU, CFA, sensor noise, metadata, JPEG, physics, retrieval | — | Full evidence; high Trust |
| Single platform recompression | metadata(partial), JPEG, physics, semantic, retrieval, frequency | PRNU, fine CFA | Strong; moderate-high Trust |
| Screenshot / heavy downscale | physics, semantic, retrieval, typography | PRNU, CFA, sensor, frequency, metadata | Lean on Tier-A only; Trust drops; may abstain |
| Tiny laundered meme (5th gen) | retrieval, semantic, gross physics | almost everything else | Low Trust; system says "insufficient evidence" rather than guessing |

> The headline behavior: **as evidence quality drops, ARGUS does not get less accurate — it gets less *confident*, lowers Trust, narrows toward laundering-robust evidence, and finally abstains.** It degrades *gracefully and honestly*. Incumbent detectors degrade *silently and overconfidently*. This is the entire point.

## 4.4 How contradictions should be handled

Contradiction is **signal, not noise.** ARGUS builds a **contradiction graph** over modules:

1. **Detect:** two modules with high reliability pointing in opposite directions (e.g., metadata says "Canon EOS R5, consistent" → authentic, while JPEG-ghost localization shows a spliced region → tamper).
2. **Weight by reliability:** a contradiction between two *high-reliability* modules is alarming and *raises Risk*; a contradiction where one party is low-reliability is resolved by down-weighting the weak one.
3. **Resolve, don't average:** ARGUS does **not** average contradictory high-reliability evidence into a mushy 0.5. Instead it (a) surfaces the contradiction explicitly in the explanation, (b) routes to the *more reliable and more generator-agnostic* source as tie-breaker, and (c) increases epistemic uncertainty (lowers Trust) to reflect genuine ambiguity.
4. **Special case — "too clean" paradox:** a region with *perfect* statistics but *absent* expected sensor noise is itself a contradiction (real captures have noise). The reliability layer flags "implausible consistency" as a synthetic indicator.
5. **Contradiction → Risk coupling:** high-reliability, high-severity contradictions drive the **Risk** score even when the Authenticity posterior is near 0.5 — because "we have strong conflicting evidence" is exactly when a human should look.

**Contradiction taxonomy:**
- *Provenance vs. pixel:* valid C2PA but pixel modules scream synthetic → likely a legitimately edited/AI-disclosed asset (C2PA may *declare* AI) → resolve via manifest semantics.
- *Capture vs. tamper:* global PRNU consistent but local splice → localized manipulation of a real photo.
- *Internal-pixel contradictions:* lighting consistent but anatomy broken → partial synthesis / inpainting.
- *External vs. internal:* retrieval finds 2019 original but metadata claims 2024 → re-dated repost.

Each pattern maps to a different **human-readable verdict template** (Phase 7).

\newpage
