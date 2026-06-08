# ARGUS — MODULE OUTPUT CONTRACT: FORMAL SPEC + CONFORMANCE SUITE
**Document #5 of the validation set · Status: REQUIRED FOR GATE G1**
**Version:** 0.1

> Why this exists: Gate G1 of the roadmap is "all 7 modules emit valid `{evidence, reliability, confidence}` tuples on 1k images." That gate is uncheckable without a machine-enforceable schema. This document is the **frozen contract** every module must satisfy, plus the conformance tests that enforce it. It also encodes the one mandated upgrade the docs call for: **`reliability_var` (a Beta posterior)** so reliability is a random variable, not a deterministic scalar.

---

## 1. The frozen JSON Schema (Draft 2020-12)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "argus.module.output.v1",
  "type": "object",
  "additionalProperties": false,
  "required": ["module","tier","evidence_score","reliability_score",
               "reliability_alpha","reliability_beta","confidence_score",
               "direction","llr","llr_weighted","ood_score","rationale","features"],
  "properties": {
    "module":            {"type":"string","minLength":1},
    "tier":              {"enum":["A","B","C"]},
    "evidence_score":    {"type":"number","minimum":0,"maximum":1},
    "reliability_score": {"type":"number","minimum":0,"maximum":1},
    "reliability_alpha": {"type":"number","exclusiveMinimum":0},
    "reliability_beta":  {"type":"number","exclusiveMinimum":0},
    "confidence_score":  {"type":"number","minimum":0,"maximum":1},
    "direction":         {"enum":["authentic","synthetic","tamper","neutral"]},
    "llr":               {"type":"number"},
    "llr_weighted":      {"type":"number"},
    "ood_score":         {"type":"number","minimum":0,"maximum":1},
    "localization":      {"type":["object","null"]},
    "rationale":         {"type":"string","minLength":1},
    "features":          {"type":"object"}
  }
}
```

## 2. Field semantics (normative)
- **`evidence_score`** ∈ [0,1] — P(authentic) *implied by this cue alone*; **0.5 = neutral**. (Direction of >0.5 = leans authentic.)
- **`reliability_score`** ∈ [0,1] — posterior **mean** = `alpha/(alpha+beta)`. "How much the image *let* this module work." Estimated from the **external quality profile**, NOT from the module's own answer (anti-circularity rule §4).
- **`reliability_alpha`, `reliability_beta`** — parameters of the **Beta posterior** over reliability. Variance = `αβ/((α+β)²(α+β+1))` propagates into Trust. *(Closes the "reliability is a point estimate" flaw.)*
- **`confidence_score`** ∈ [0,1] — internal certainty *given a readable channel*. Independent of reliability (a clean image with an ambiguous reading is high-reliability / low-confidence).
- **`llr`** — calibrated log-likelihood ratio, **pre-reliability**.
- **`llr_weighted`** — the actual fusion contribution. **Invariant (R3):** `llr_weighted == reliability_score * llr` (the specified tempering rule; flagged as a *heuristic*, see KT-4).
- **`ood_score`** ∈ [0,1] — distance from the module's training manifold (0 for purely classical modules with no learned component; they declare it explicitly).
- **`direction`** — must be consistent with `evidence_score` per R4.

## 3. Cross-field invariants (machine-checked)
| ID | Invariant | Rationale |
|----|-----------|-----------|
| R1 | all probabilities ∈ [0,1]; `llr` finite | basic validity |
| R2 | `reliability_score ≈ alpha/(alpha+beta)` (±1e-6) | Beta mean consistency |
| R3 | `llr_weighted == reliability_score * llr` (±1e-9) | the fusion rule is exactly the tempering rule |
| R4 | `direction=="neutral"` ⇒ `abs(evidence_score-0.5) < 0.05` AND `abs(llr) < 0.1` | neutral means neutral |
| R5 | `direction=="authentic"` ⇒ `evidence_score ≥ 0.5`; `"synthetic"/"tamper"` ⇒ `evidence_score ≤ 0.5` | sign consistency |
| R6 | absence-of-signal ⇒ `reliability_score == 0` (NOT `evidence_score` extreme) | **absence = neutral, never "fake"** (EXIF/C2PA stripped) |
| R7 | classical (non-learned) module ⇒ `ood_score == 0` and declared | OOD only meaningful for learned modules |
| R8 | `features` is JSON-serializable & deterministic for a fixed input+seed | audit/reproducibility |

## 4. Anti-circularity rule (normative)
`reliability_score` **must not** be a function of `evidence_score` or `confidence_score`. It is computed from the **quality vector + spatial self-consistency + (1−ood)** only. The conformance suite enforces this by a **perturbation test**: holding the quality vector fixed and changing only the pixel content that drives `evidence_score`, `reliability_score` must not change beyond tolerance.

## 5. Per-module reliability-zeroing table (frozen preconditions)
| Module | `reliability_score → 0` when… |
|--------|-------------------------------|
| Metadata/EXIF/XMP | EXIF absent/stripped |
| C2PA | manifest absent |
| JPEG double-quant + grid | Q<60 OR downscale flag OR ≥2 JPEG generations |
| ELA | screenshot OR heavy recompression |
| FFT azimuthal | resample/blur OR (learned-variant) OOD on unknown generator |
| Noise-residual / too-clean | low-res OR screenshot (corroborator only, capped weight) |
| Quality Profiler | never (it *is* the gate source) |

## 6. Conformance suite (the executable G1 gate)
The suite is a pytest module run over a 1k-image fixture. **G1 passes iff all checks are green on all 7 modules over all 1k images.**

```
T1  schema_validation       : every output validates against argus.module.output.v1
T2  invariants_R1_R8         : all cross-field invariants hold per image
T3  anti_circularity         : §4 perturbation test passes (Δreliability < 1e-3)
T4  zeroing_preconditions    : §5 table — reliability is 0 exactly when precondition met
T5  determinism              : same input + seed → byte-identical output (excl. timestamps)
T6  range_coverage           : across the 1k set, each module exercises r∈{~0, mid, ~1}
                               (proves the reliability layer is not a constant)
T7  beta_consistency         : R2 within tolerance; variance finite & > 0 when 0<r<1
T8  neutral_on_absence       : stripped-EXIF / no-C2PA images → direction neutral, r=0
```

## 7. Failure handling (normative)
- A module that cannot run on an input must return a **valid tuple** with `reliability_score=0`, `direction="neutral"`, `rationale` stating why — **never** an exception, a missing field, or a fabricated extreme score.
- The fusion layer treats `reliability_score=0` as exactly zero LLR contribution (R3 ⇒ `llr_weighted=0`).

## 8. Versioning
- Schema `$id` carries the version (`v1`). Any field change ⇒ `v2` + a migration note; calibrators and the FAISS index are **invalidated** on version bump (recorded in Doc #6 determinism boundary).

---
*G1 sign-off: T1–T8 green on 1k images for all 7 modules, conformance report committed with the dataset manifest hash.*
