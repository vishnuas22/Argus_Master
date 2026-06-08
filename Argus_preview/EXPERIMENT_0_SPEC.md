# ARGUS — EXPERIMENT 0: THE MINIMUM DECISIVE TEST
**Independent forensic review · senior research-scientist / statistician sign-off**
**Status:** proposed confirmatory protocol · architecture assumed FROZEN · no new modules introduced
**Optimised for:** scientific rigor · low compute (CPU-only, single laptop) · reproducibility · fast feedback (hours, not weeks)

---

## 0. Framing — what is actually on trial

**Central thesis (verbatim):**
> "Reliability-aware evidence fusion provides more trustworthy authenticity assessment than detector-only systems."

This sentence contains exactly one novel, load-bearing object: the **reliability layer** — the per-module, per-image term `reliability_score ∈ [0,1]` that tempers each module's log-likelihood ratio via the frozen invariant **R3: `llr_weighted = reliability_score · llr`** (Module Contract §3). Everything else in ARGUS (the classical cues, isotonic calibration, log-odds pooling, threshold abstention) is prior art. **If the reliability layer is empirically null, the thesis is dead and ARGUS collapses to "a forensic ensemble" — exactly the pre-registered failure mode (Pre-reg §6).**

The repo's own "Kill Test C" already names the right comparison (A-full vs A-null vs CLIP-TS). Experiment 0 is **not** that 30-day study. It is the **smallest sufficient subset** of it, plus **one control the existing plan is missing** (see §5/§6 — the *generic-shrinkage* confound, which a skeptical reviewer will raise first and which the current pre-registration does not defeat).

**Design principle:** falsification-first. Experiment 0 is built to *kill* the thesis cheaply if it is false. It is powered to detect the **large** effect the thesis predicts under laundering, not the 0.02-ECE marginal effect (that is the full study's job). A null here is a legitimate, publishable, pre-committed outcome.

---

## 1. EXPERIMENT 0 SPECIFICATION

### 1.1 The single question
> Under social-media laundering, does the **per-module, per-image reliability weighting** improve calibration and selective risk **beyond what is explained by (a) using no reliability and (b) generic confidence-dampening**, when compared against a detector-only baseline?

### 1.2 Scope reductions vs the full kill test (and why each is safe)
| Full kill test (Doc set) | Experiment 0 | Justification |
|---|---|---|
| 7 modules | **4 modules** (Quality Profiler [gate], FFT azimuthal, JPEG double-quant+grid, Noise-residual/too-clean) | These are the modules whose reliability is *driven to ~0 by laundering* (Module Contract §5). They are the ones through which the reliability mechanism can act. EXIF/C2PA reliability-zeroing is trivial (absence→0) and adds no mechanistic information; ELA is a weak corroborator. 4 modules still exercises r∈{~0, mid, ~1} (T6). |
| ≥3 held-out generator families, ≥1000+1000 per family | **1 generator family, ~400 synthetic + ~400 authentic source images** | The reliability/laundering claim is **orthogonal to unseen-generator generalisation**. Laundering, not novel generators, is the driver of reliability heterogeneity. LOGO/open-set (H2) is a *separate* claim, tested by the one-afternoon secondary probe (§6.2), not required to validate/falsify the central reliability thesis. |
| Full laundering grid (5×JPEG × 3×downscale × screenshot × double-compress) | **Reduced grid: JPEG-Q ∈ {90, 60, 40} × downscale ∈ {1×, 2×} + screenshot-sim** | Spans pristine→severe. 3×2+1 = 7 conditions per source. Enough to produce a monotone severity axis (needed for H3) without combinatorial blow-up. |
| CLIP probe + deep ensemble + open-set detector | **CLIP-ViT-B/32 + LR probe + temperature scaling (CPU)** as the detector-only baseline; ensemble/open-set deferred | One strong, pre-registered, calibrated detector-only baseline is sufficient to test "more trustworthy than detector-only." |

### 1.3 Arms (frozen before any test result is read)
All arms receive **byte-identical inputs** and share identical calibrators/thresholds fit on the **calibration split only**.

| Arm | Definition | Role |
|---|---|---|
| **A-full** | 4 modules → per-module isotonic → reliability-tempered log-odds pool (`Σ rᵢ·llrᵢ`) → threshold abstention | The thesis |
| **A-null** | *Identical* to A-full but **all `reliability_score ≡ 1.0`** (`Σ llrᵢ`) | Isolates the sole novelty (internal validity) |
| **A-shrink** ⚠ | *Identical* to A-null but all LLRs multiplied by a **single global constant `s∈(0,1]`** chosen on the calibration split to match A-full's mean output sharpness (mean \|pooled-LLR\| equal to A-full's, per laundering bin) | **The missing control.** Distinguishes "reliability is informative" from "reliability just dampens confidence." See §5.3. |
| **CLIP-TS** | Frozen `open_clip` ViT-B/32 (`laion2b_s34b_b79k`) → LR probe → temperature scaling | The "detector-only system" of the thesis |
| **BEST-1** (optional, free) | The single best individual classical module by calibration on the calibration split, isotonic-calibrated | A second, zero-cost "detector-only" reference |

### 1.4 Pre-registered primary comparisons
- **P1 (sole-novelty):** `A-full` vs `A-null`
- **P2 (literal thesis):** `A-full` vs `CLIP-TS`
- **P3 (anti-confound, decisive):** `A-full` vs `A-shrink`

On two primary endpoints each: **laundered-ECE** and **hard-subset selective risk (SR_hard)**. Brier is the proper-scoring tie-breaker.

### 1.5 Freeze / lockbox protocol (cheap, mandatory)
1. Freeze §1.3–§1.4, §3, §4 → record a SHA-256 of this file in git.
2. Split sources into **calibration** (fit isotonic, temperature, abstention τ, and A-shrink's `s`) and **test** (never touched until frozen). Cluster by source image — all laundered children stay together (SAP §3, Label Spec §8).
3. Test images → write-once dir + SHA-256 manifest; analysis verifies hashes at runtime.
4. Single global RNG seed recorded in run config; `predictions.parquet` (one row per image per arm) is the only input to the analysis script.

---

## 2. DATASET REQUIREMENTS

**Total raw source images needed: ~800 (≈400 authentic + ≈400 synthetic). Everything else is generated by the laundering grid.** This is the minimum that keeps clustered-bootstrap CIs usable for a *large* effect; see §5.4 power note.

| Pool | Source | n (sources) | Label basis | Use |
|---|---|---|---|---|
| Authentic-gold | RAISE or Dresden (RAW / in-camera JPEG), **C0/C1 only** | ~250 | provenance-anchored (RAW) | calibration + test |
| Authentic-wild | Flickr-CC **pre-2021** | ~150 | temporal anchor <2021 | the "real but degraded" distribution |
| Synthetic-gold | **one** GenImage or DiffusionForensics family (e.g. SD-v1.x), **C5 full-synthesis only** | ~400 | known generation provenance | calibration + test |
| Suspicious-but-real (FP probe) | ~120 reused subset: ≥40 flagship computational-photography + ≥40 low-end phone + ≥40 non-Western faces/dress | provenance-anchored authentic | **false-positive / liability gate only — never fit** |

**Deliberate label simplification:** use only **C0/C1 (authentic gold)** and **C5 (full synthetic)**. Drop the C3 boundary class entirely from Experiment 0 (Label Spec §4 already forbids it from calibration; here we also drop it from test to remove the single largest label-noise source). This buys clean ground truth at near-zero cost and is explicitly *not* a metric swap — it is a documented scope narrowing.

**Integrity controls that still apply (each is cheap and gating):**
- **C1 contamination audit** — run the demoted CLIP probe over the authentic pools; manually review top-50; require estimated synthetic contamination < 2%.
- **C2 confound control (the FFHQ trap)** — synthetic family must **not** share its authentic source distribution with the authentic pools (no aligned-face vs aligned-GAN-face matchup). If unavoidable, report seen-generator numbers as confounded. For Experiment 0, prefer a **scene/object generator family** over faces to sidestep alignment confounds entirely.
- **C3 leakage** — no source image spans calibration/test; calibrators fit on calibration split only.
- **C4 lockbox + SHA-256 manifest.**

**Hard subset predicate (frozen, simplified):** `{no EXIF/C2PA} ∧ {JPEG-Q ≤ 60 OR screenshot flag}`. (Retrieval term dropped — no FAISS in Experiment 0.)

**Compute footprint:** ~800 sources × 8 variants (7 laundered + pristine) ≈ 6.4k images. Classical modules are numpy/opencv/scipy, CPU-only, ~50–300 ms/image → **single-core hours; trivially parallel.** CLIP-ViT-B/32 feature extraction on CPU ≈ 0.3–1 s/image → a few hours; GPU optional, not required. **No model training. No GPU mandatory.**

---

## 3. SUCCESS CRITERIA (pre-registered, FROZEN)

The thesis **survives Experiment 0** iff **ALL** of the following hold on the held-out test set, after **Holm–Bonferroni** across the primary CI family, using **paired BCa bootstrap (B=1000), source-clustered**:

**S1 — sole novelty is real (P1):**
`CI95(ECE_L[A-null] − ECE_L[A-full]).low > 0` **AND** `CI95(SR_hard[A-null] − SR_hard[A-full]).low > 0`.

**S2 — beats detector-only (P2):**
`CI95(ECE_L[CLIP-TS] − ECE_L[A-full]).low > 0` **AND** `CI95(SR_hard[CLIP-TS] − SR_hard[A-full]).low > 0`.

**S3 — not just generic shrinkage (P3, the decisive control):**
`CI95(ECE_L[A-shrink] − ECE_L[A-full]).low > 0` (A-full strictly better than sharpness-matched global dampening, on at least the calibration metric).

**S4 — effect-size floor (anti-trivial-win):**
point estimates clear **ΔECE ≥ 0.02** and **ΔSR_hard ≥ 0.03** for P1.

**S5 — honesty guardrails pass:**
(a) **FD_acc** (forced-decision accuracy at 100% coverage on the hard subset) for A-full is **not worse** than A-null — i.e. the selective-risk win is *not* bought by abstaining on everything hard;
(b) **Liability gate KT-7:** suspicious-but-real false-"manipulated" rate < 10% overall and no subgroup ≥ 2× the overall rate;
(c) **Anti-circularity T3:** reliability perturbation test passes (Δreliability < 1e-3 when only evidence-driving pixels change).

**Headline survival statement (what gets reported):**
> "Per-image reliability weighting reduced laundered-ECE by Δ (95% BCa CI […]) over identical fusion without reliability, over a sharpness-matched global-shrinkage control, and over a temperature-scaled CLIP detector, while not degrading forced-decision accuracy — on consumer hardware, fully reproducible from seed+manifest."

---

## 4. FAILURE CRITERIA (pre-registered, FROZEN — each is a legitimate publishable result)

| ID | Condition | Interpretation | Action |
|---|---|---|---|
| **F1** | `A-full` and `A-null` CIs **overlap** on either primary metric, OR ΔECE < 0.02 / ΔSR_hard < 0.03 | The reliability layer (sole novelty) is **empirically null** | **STOP. Publish negative.** ARGUS = forensic ensemble. |
| **F2** | `CLIP-TS` matches or beats `A-full` on laundered calibration within CIs | The differentiator vs detector-only is gone | STOP / relabel the contribution. |
| **F3 ⚠** | `A-shrink` matches `A-full` within CIs | Reliability adds **nothing beyond generic confidence-dampening**; the per-module/per-image structure is decorative | **Thesis hollow.** Demote "reliability-aware" to "temperature-scaled fusion." This is the most likely *quiet* failure and the existing plan would miss it. |
| **F4** | A-full's SR_hard win disappears once **FD_acc** is held equal (win was an abstention artifact) | "Selective risk" gain is gaming, not trust | Not supported. |
| **F5** | Liability gate fails (FP ≥ 10% or subgroup disparity ≥ 2×) | Deployment-blocking regardless of all else | Flag as blocking. |
| **F6** | Anti-circularity T3 fails | Reliability is secretly a function of evidence → the whole construct is circular | Construct invalid; halt and fix contract before any claim. |

---

## 5. STATISTICAL TESTS

### 5.1 Estimands (from SAP §1)
`ECE_L` = 15-bin **adaptive (equal-mass)** ECE on laundered images, on the declared-prior posterior (Label Spec §6 fixed eval prior = empirical test class balance). `Brier` = mean((p−y)²), the unbiased tie-breaker. `SR_hard(c)` = error among the `c`-fraction most-confident answered hard-subset cases. `FD_acc` = accuracy at 100% coverage on the hard subset.

### 5.2 Inference (from SAP §3)
- **Paired BCa bootstrap, B=1000**, resampling the **per-arm difference** on the **same images**.
- **Cluster unit = source image.** All laundered children resample together (prevents leakage-inflated significance — non-negotiable).
- Report **full 95% BCa CI** of every difference, not just p.
- **Multiple comparisons:** primary family = the CI tests in S1–S3; control family-wise error with **Holm–Bonferroni at α=0.05**. Everything else (per-condition sweeps, BEST-1, H3 slope) labeled **exploratory — not error-controlled**.

### 5.3 The mandatory anti-confound test (A-shrink) — why it is the scientific core
`llr_weighted = r·llr` (R3) is, by the repo's own admission (KT-4; critical-findings §3: "reliability-tempered LLR is heuristic, not calibrated"; H5 expects to *reject* "calibrated by construction"), **a shrinkage operator**. Any shrinkage of over-large LLRs improves calibration on hard/laundered data where the model *should* be uncertain. Therefore **A-full will almost certainly beat A-null — even if reliability carries zero information** — simply because multiplying by numbers in (0,1] reduces overconfidence.

A skeptical reviewer's first question: *"Is your reliability term doing anything a single global temperature wouldn't?"* **A-shrink answers it.** It is the cheapest possible falsifier of the thesis and it is **absent from the current pre-registration.** If A-full ≯ A-shrink, the "reliability-aware" branding is unsupported (F3). Adding A-shrink costs one scalar fit and one extra prediction column — essentially free.

### 5.4 Power note (honest)
With ~250 authentic + ~400 synthetic test sources and ρ≈0.7 paired correlation, Experiment 0 has **~80% power to detect ΔECE ≈ 0.04–0.05** (a *large* effect), not the full study's 0.02. **This is intentional:** Experiment 0's job is to detect the large laundering effect the thesis loudly predicts, or to return a null fast. A *borderline* result (CIs straddling the 0.02 floor) is reported as **"underpowered — escalate to the full ≥1000/class study,"** not as survival. State this up front so a near-miss is not over-read.

### 5.5 Supporting analyses
- Reliability **diagrams** with per-bin Wilson CIs accompany every ECE scalar (never a lone number).
- ECE computed **separately** on pristine vs laundered (pooling hides the entire claim — SAP §7).
- **Pre-vs-post isotonic ECE** reported (makes the H5 "calibrated by construction" falsification visible).
- Prior-sensitivity band: recompute posterior metrics at prior ∈ {0.1,0.3,0.5,0.7,0.9}.

---

## 6. ABLATION DESIGN

### 6.1 Primary ablation ladder (single axis: the reliability mechanism)
```
A-full     : Σ rᵢ·llrᵢ          (full per-module, per-image reliability)
   │  ← P1: tests reliability vs none
A-null     : Σ llrᵢ             (no reliability)
   │  ← P3: tests reliability-structure vs generic dampening
A-shrink   : Σ s·llrᵢ , s const  (global sharpness-matched shrinkage)
```
The decisive publishable number is **Δ(ECE_L, SR_hard) between A-full and A-null**, *conditioned on A-full also beating A-shrink.* Reporting A-full vs A-null **without** A-shrink is the gap in the current plan.

### 6.2 Two cheap structure-probes (exploratory, one afternoon each)
- **Reliability-shuffle:** permute each image's per-module reliability vector across modules (keep the multiset, destroy the assignment). If A-full ≈ A-shuffle, the *which-module-is-reliable* information is inert (a stronger, sharper version of the A-shrink test).
- **Per-condition decomposition:** plot Δ(A-full − A-null) vs laundering severity. The thesis predicts the gain **grows with severity** (reliability matters most when modules break). A flat or inverted curve is evidence against the mechanism even if the pooled CI clears 0.

### 6.3 Module-attribution (only if pooled win survives; correlation-aware, SAP §6)
Do **not** use naïve leave-one-out (correlated modules mask each other). Use **grouped knockout** of the correlated cluster + **Monte-Carlo Shapley (200 permutations)** over the 4 modules. Exploratory only.

### 6.4 Secondary kill (independent of everything above, optional)
The "durable physics backbone" (H2) is **not** part of the central reliability thesis and is **excluded from Experiment 0's go/no-go.** If desired as a one-afternoon add-on: one physics-aware generator family → physics-module AUROC; **AUROC upper-CI < 0.65 ⇒ durability claim falsified.** Reported separately; never gates the reliability result.

---

## 7. THREATS TO VALIDITY

### 7.1 Internal validity
- **T-INT-1 — Shrinkage confound (highest priority).** A-full beats A-null merely by dampening. **Mitigation: A-shrink + reliability-shuffle arms.** Without these, the entire result is uninterpretable.
- **T-INT-2 — Circularity.** If `reliability_score` is even weakly a function of `evidence_score`, reliability becomes a confidence proxy and the comparison is rigged. **Mitigation: T3 perturbation test as a hard gate (F6).**
- **T-INT-3 — Calibration leakage.** Fitting isotonic / temperature / τ / `s` on data that touches test inflates everything. **Mitigation: strict calibration/test source-cluster split; SHA-256 lockbox.**
- **T-INT-4 — Abstention gaming.** SR_hard win bought by abstaining on hard cases. **Mitigation: mandatory FD_acc at 100% coverage (S5a/F4).**
- **T-INT-5 — "Too-clean ⇄ self-silencing" collision** (critical-findings §3). Laundering both smooths the image (drives the "too-clean" evidence) *and* lowers reliability — shared cause. The reliability gain could be the too-clean module reading its own degradation. **Mitigation:** keep noise-residual as *corroborator, capped weight* (Module Contract §5); report Δ with and without the too-clean module.

### 7.2 External validity
- **T-EXT-1 — Single generator family.** Result may not transfer to other generators. *Accepted scope limit; the claim under test is the laundering/reliability mechanism, not cross-generator generalisation.* Escalate to multi-family in the full study.
- **T-EXT-2 — Simulated ≠ real laundering.** JPEG/downscale/screenshot-sim under-represent real platform round-trips. **Mitigation:** the datasheet's 200-image real-vs-sim spot check (KT/A2) bounds the gap; report it.
- **T-EXT-3 — Demographic/device skew** in authentic pools → biased reliability and FP. **Mitigation:** suspicious-but-real subgroup FP gate (S5b).

### 7.3 Construct & statistical validity
- **T-STAT-1 — Underpowering / over-reading a near-miss.** **Mitigation: §5.4 honesty rule** — borderline ⇒ "escalate," never "survive."
- **T-STAT-2 — ECE binning bias at small N.** **Mitigation:** Brier as confirmatory tie-breaker; Wilson per-bin CIs; adaptive equal-mass bins.
- **T-STAT-3 — Multiplicity.** Several arms × 2 metrics. **Mitigation: Holm–Bonferroni** on the primary family; everything else labelled exploratory.
- **T-STAT-4 — Label boundary noise.** **Mitigation:** C0/C1 vs C5 only; C3 excluded.

---

## 8. MINIMUM IMPLEMENTATION REQUIRED

**No new modules. No model training. No GPU. ~4 existing classical modules + the existing fusion/calibration code + an analysis script.** Estimated build: a few engineer-days on top of the frozen MVP code.

```
exp0/
├── modules/                 # REUSE existing frozen modules (no changes)
│   ├── quality_profiler.py  # produces the quality vector (gate source)
│   ├── fft_azimuthal.py
│   ├── jpeg_dq_grid.py
│   └── noise_residual.py    # too-clean, capped-weight corroborator
├── reliability.py           # existing fitted transfer functions r(quality) per module
├── fusion.py                # Σ rᵢ·llrᵢ  ; arm flag toggles r≡1 (A-null), s·llr (A-shrink), shuffle
├── calibrate.py             # per-module isotonic + CLIP temperature + abstention τ + A-shrink s  (CAL SPLIT ONLY)
├── launder.py               # JPEG-Q∈{90,60,40} × downscale∈{1,2} + screenshot-sim
├── clip_probe.py            # open_clip ViT-B/32 → LR probe → temperature scaling (CPU ok)
├── run_arms.py              # writes predictions.parquet : (image_id, source_id, arm, p, llr, abstain, y, condition, subgroup)
├── analyze.py               # BCa paired clustered bootstrap → ECE_L, Brier, SR_hard, FD_acc, CIs, Holm; emits tables+figs
├── conformance.py           # T1–T8 schema/invariant/anti-circularity (T3) gate
├── manifest.sha256          # lockbox hashes of test images
└── run_config.yaml          # single seed, split definition, frozen thresholds
```

**Single source of truth for analysis:** `predictions.parquet` (one row per image per arm). `analyze.py` is deterministic from it — any reviewer recomputes every CI. Release `predictions.parquet` + `manifest.sha256` + `run_config.yaml`; that is full reproducibility without redistributing pixels.

**Decisive minimal deliverable = one table:**

| Metric (laundered / hard subset) | A-full | A-null | A-shrink | CLIP-TS | Δ(null−full) [95% BCa] | Δ(shrink−full) | Δ(CLIP−full) |
|---|---|---|---|---|---|---|---|
| ECE_L | … | … | … | … | […] | […] | […] |
| SR_hard | … | … | … | … | […] | […] | […] |
| FD_acc@100% | … | … | … | … | (guardrail) | | |

---

## 9. WHAT WOULD CONVINCE A SKEPTICAL REVIEWER

A reviewer predisposed to dismiss ARGUS as "rebranded shrinkage" is convinced **only** by this exact constellation:

1. **A-full beats A-null** on laundered-ECE *and* SR_hard, ΔECE ≥ 0.02 / ΔSR_hard ≥ 0.03, non-overlapping source-clustered BCa CIs after Holm.
2. **A-full beats A-shrink** (and A-shuffle) — proving the gain comes from *which* module is reliable *on this image*, not from generic dampening. **This is the single most persuasive number**; it is the one the current plan omits.
3. **A-full beats CLIP-TS** on laundered calibration — the literal "better than detector-only" claim.
4. **The Δ(A-full − A-null) curve grows with laundering severity** — the mechanism behaves as theorised (reliability matters most when modules break), not as a fluke of pooled averaging.
5. **FD_acc not degraded** — the selective-risk win is honest, not abstention-gaming.
6. **Anti-circularity (T3) passes** and **liability gate passes** — the construct is sound and not silently harmful.
7. **Reproduced from the released `predictions.parquet` + seed**, on CPU, in hours.

The crisp sentence that would move a skeptic:
> "Sharpness-matched global shrinkage closed only X% of the A-null→A-full calibration gap; the remaining gain tracks per-image reliability assignment and grows monotonically with laundering severity — on held-out data, reproducible from seed."

---

## 10. WHAT WOULD INVALIDATE THE CENTRAL CLAIM

Any **one** of these falsifies "reliability-aware evidence fusion provides more trustworthy authenticity assessment than detector-only systems":

1. **A-full ≈ A-null** (overlapping CIs, or Δ below the 0.02/0.03 floor) → reliability layer is empirically null → **ARGUS = forensic ensemble; thesis dead** (F1). *Most direct kill.*
2. **A-full ≈ A-shrink** → the reliability term does nothing a single global temperature wouldn't → **"reliability-aware" is decorative** (F3). *Most likely quiet kill — and invisible to the current pre-registration.*
3. **A-full ≈ or < CLIP-TS** on laundered calibration → no advantage over a detector-only system → **the comparative claim fails** (F2).
4. **SR_hard win vanishes when FD_acc is equalised** → "more trustworthy" was abstention-gaming (F4).
5. **Anti-circularity T3 fails** → reliability is a function of evidence → the construct is circular and the whole comparison is invalid (F6).
6. **Liability gate fails** (suspicious-but-real FP ≥ 10% or subgroup disparity ≥ 2×) → the system is *less* trustworthy where it matters most, regardless of ECE (F5).

**Critical reviewer's bottom line:** the thesis is strong only if it survives the **A-shrink control**. The repo's own kill test (A-full vs A-null vs CLIP-TS) is necessary but **not sufficient** — it cannot distinguish an informative reliability signal from a generic confidence discount. Experiment 0's one substantive addition over the frozen plan is to close that hole at near-zero compute cost. If the team runs only one experiment before building V2, it must be this one, and it must include A-shrink.

---
*No architectural change proposed. No new module introduced. All arms, metrics, and decision rules are drawn from the frozen ARGUS validation set; the sole addition is the A-shrink / A-shuffle confound control mandated by standard calibration-study practice.*
