# ARGUS — IMPLEMENTATION READINESS REVIEW
**Panel:** Principal Software Engineer · ML Systems Architect · Digital Forensics Researcher · MLOps Lead · CVPR Reviewer
**Mandate:** Architecture is FROZEN. We do **not** critique the design or propose new forensic modules. We answer one question only: **Can a competent team start coding tomorrow and build the *same* system, reproducibly, from these documents?**
**Verdict (one line):** The *scientific scaffolding* (pre-registration, SAP, label spec, module contract) is unusually strong; the *buildable specification* is not. There is **no code, no dependency lock, no dataset, contradictory tech stacks, and the most load-bearing interfaces are undefined.** **NOT implementation-ready.**

---

## 0. The repository, factually
- It is **100% Markdown**. `PRD.md` confirms: *"DONE — Phase 1: written design document only … No app/services were created."*
- `EXPERIMENT_0_SPEC §8` ("a few engineer-days **on top of the frozen MVP code**") presupposes an MVP codebase that **does not exist in this repo.** Every "reuse existing module" instruction points at vapor.
- Two mutually contradictory engineering constitutions are committed (`AGENTS.md` = Python/enterprise; `AGENTS_FRONTEND.md` = Next.js/Node), neither matching the MVP spec (React+FastAPI+Mongo). See §2.

---

## 1. EVERY AMBIGUOUS REQUIREMENT
Items where the text under-determines the implementation. Citations are to repo files.

### 1.1 The quality vector — the single most load-bearing undefined object
`02_A_mvp_freeze.md` Module 0 and `04_phase4.md §4.2` make **every** reliability transfer function a function of "the quality vector," yet the quality vector has **no fixed schema** (field names, count, ranges, units, normalization). Each sub-feature is named but not specified:
- "estimated JPEG quality & generations" — **no quant-table→Q algorithm** (IJG tables? custom tables? which standard mapping?).
- "double-compression via DCT first-digit / Benford deviation" — **no statistic, no threshold.**
- "blockiness (8×8 grid energy)" — undefined among a dozen published blockiness metrics.
- "blur (variance of Laplacian)" — no kernel size, no normalization, no decision threshold.
- "noise floor (MAD of wavelet HH)" — **wavelet family, decomposition level, MAD scale constant all unspecified.**
- "screenshot heuristic (no EXIF + exact-multiple dims + uniform Q)" — "exact-multiple dims" of *what* (which screen-resolution table?); "uniform Q" tolerance unspecified.
→ Two engineers will emit different-dimensional, differently-scaled quality vectors, which silently changes **every** reliability value downstream.

### 1.2 Reliability representation is doubly-defined and never reconciled
- `04_phase4.md §4.2`: `reliability = min(a)·g(b)·g(c)` — but **`g()` is never defined**, and (a),(b),(c) have no formulas ("SNR of the evidence map", "fitted curve").
- `04_MODULE_CONTRACT_SPEC.md §2`: reliability is the **mean of a Beta posterior** `alpha/(alpha+beta)`, with `reliability_alpha/beta` required fields and invariant R2/T7.
- **Nowhere is the map from "fitted accuracy-vs-quality curve" → `(alpha, beta)` specified.** The Beta posterior is mandatory (drives Trust variance) yet has no estimator. The product-form and the Beta-form are two different reliability models with no bridge.

### 1.3 Transfer-function functional form is unspecified
`03_BC_roadmap_killtest.md` Gate G2 demands "monotone, cross-validated transfer functions" but never fixes the model class (isotonic? logistic? monotone spline? GP?). Fit-error pass threshold is "reported," not bounded. Two teams fit different curves and both "pass."

### 1.4 Trust, Risk, verdict bands — named as functions, never defined
`05_phase5.md §5.3` / `09_final_score.md` App. B:
- `Trust = f(mean_reliability, coverage, 1−contradiction, 1−OOD)` — **`f` undefined** (weights, form).
- `Risk = g(1−Authenticity, contradiction_severity, context_prior, harm_weight)` — **`g` and `harm_weight` undefined**; `01_critical_findings.md §6` itself states the harm model "does not exist in MVP."
- `07_phase7_8.md` shows `verdict_band: "LIKELY MANIPULATED — MEDIUM-HIGH CONFIDENCE"` and asserts bands "map deterministically from (authenticity, trust)" — **the cutoffs are never given.**

### 1.5 "mean_reliability", "evidence_coverage" — undefined aggregations
Used in abstention and Trust. Is the mean over all 7 modules (including r≈0 silenced ones) or only fired modules? Is coverage a count, a fraction, threshold-gated? Each choice materially changes abstention. Not specified anywhere.

### 1.6 Abstention thresholds tuned to an undefined objective
`01_PROBLEM_AND_LABEL_SPEC.md §7`: τ_cov, τ_rel "tuned on a calibration split." **Tuned to optimize what?** (target coverage? selective risk? Youden's J?) Unspecified ⇒ irreproducible thresholds.

### 1.7 LLR computation underspecified
`05_phase5.md`: `LLR = log[P(e|auth)/P(e|synth)]` via per-module isotonic. **No clipping bound** on calibrated p ⇒ `log(p/(1−p))` → ±∞ on confident modules. No defined isotonic training set/size per module.

### 1.8 context_prior value is circular / unpinned
`01_PROBLEM_AND_LABEL_SPEC.md §6` fixes the eval prior to "empirical class balance of the labeled test set" — but the test-set class balance is not stated, and using the test balance to set the prior that scores the test set is methodologically circular for any non-study use. The actual number is never committed.

### 1.9 evidence_score → direction sign convention left to the engineer
Invariant R5 requires `authentic ⇒ evidence ≥ 0.5`. But modules emit "inconsistency scores" (JPEG, ELA, FFT) where high = tamper. The inversion (raw feature → [0,1] P(authentic)) is **not given for any module** — each engineer picks a mapping.

---

## 2. EVERY UNDOCUMENTED / CONFLICTING DEPENDENCY
- **No `requirements.txt`, `pyproject.toml`, `package.json`, or lockfile exists.** `02_A_mvp_freeze.md §A.3` lists "pinned majors" using `>=` (e.g., `numpy>=1.26`, `Pillow>=10`) — **`>=` is not a pin.** numpy 2.x / OpenCV ABI breaks are unguarded. Not reproducible.
- **Tech-stack contradiction (blocking):**
  - `AGENTS_FRONTEND.md §3` (binding): **Next.js + TypeScript + D3 + NestJS/Fastify (Node) + Socket.IO + Celery + Redis.**
  - `PRD.md` / `02_A_mvp_freeze.md`: **React 18 + Vite + FastAPI (Python) + MongoDB + rq/celery.**
  - These specify **different backend languages.** Unresolvable without a decision.
- **Backend constitution vs MVP mismatch:** `AGENTS.md` mandates OAuth2/JWT, Redis cache, Prometheus/Grafana, OpenTelemetry, Kafka/RabbitMQ, Alembic migrations (implies SQL), Terraform/K8s — while the product is "no user accounts, single-page tool" on MongoDB. Auth, queue, DB, and observability stacks are all under-/over-specified relative to the build.
- **EXIF tool conflict:** `08_phase9_10.md` lists `exiftool` (Perl system binary; GPL/Artistic; subprocess); `02_A_mvp_freeze.md §A.3` lists `exifread`+`piexif` (pure Python). Different capabilities, different outputs. Pick one.
- **Native/young libs without build docs:** `c2pa-python>=0.5` (Rust native, ABI-churny), `faiss-cpu` (build/memory), `open_clip` weights `laion2b_s34b_b79k` (~600MB, no pinned hash/source).
- **VLM dependency unresolved:** `Qwen2-VL`/`LLaVA` (local weights, VRAM unspecified) **or** Gemini/Groq API (network + nondeterminism). "User provides key" — provider not fixed. Breaks the `T5` determinism + audit-reproducibility claims.
- **Conformal libs (`mapie`/`crepes`) listed then excluded** ("Frozen OUT of MVP") — inconsistent.
- **Queue undecided:** `rq`/`celery` ("or"); broker (Redis) not pinned in backend spec.
- **Python version conflict:** `python==3.11` (A.3) vs "Python 3.9+ syntax" (`AGENTS.md`).
- **No base OS/container image** despite Docker/IaC mandates. No CUDA/driver matrix for the optional GPU path.

---

## 3. EVERY MISSING DATASET SPECIFICATION
- **No data directory, no manifests, no file lists exist.** All SHA-256 "lockbox manifests" referenced (`03_DATASET_DATASHEET.md §C4`, `00_PRE_REGISTRATION.md §4`) are promised, not present.
- **Ambiguous source versions:** "RAISE-1k/8k" (which?); Dresden "subset" (which sensors/scenes?); "Flickr-CC pre-2021 (~5k)" (no query, no per-image license list, no download script).
- **Synthetic pools unscoped:** GenImage / DiffusionForensics "subsets" — which generators, how many, what split? `LOGO` holds out "≥2 families" but **never names which families** ⇒ not reproducible.
- **Pool 4 (suspicious-but-real, ~500) "to be built"** — does not exist; no sourcing/consent/annotation pipeline beyond "prefer CC."
- **Pool 5 (physics-aware generator) "to be sourced"** — no specific generator identified.
- **Degradation grid is specified three times with different values:**
  - `00_PRE_REGISTRATION §4` / `03_DATASET_DATASHEET`: JPEG-Q ∈ {95,85,75,60,40}; downscale ∈ {1.5,2,3}.
  - `EXPERIMENT_0_SPEC §1.2`: JPEG-Q ∈ {90,60,40}; downscale ∈ {1,2}.
  - → Which grid is canonical? "screenshot-sim" and "double-compress" have **no defined algorithm/parameters** in any file.
- **Class balance per pool unfixed** (yet it defines the eval prior, §1.8).
- **Annotation spec incomplete:** the C2/C3 boundary uses a 1.0% area threshold (`01_PROBLEM_AND_LABEL_SPEC §3`) but no mask format, annotation tool, annotator pool, or per-image budget; κ≥0.6 target without an annotator process.
- **Module-count inconsistency across docs:** "20+ orthogonal modules" (abstract) vs "7" (MVP freeze) vs "8" (Phase 8). Affects D_eff target and contract scope.
- **Licensing/redistribution** per pool is a "to record" TODO (`§Part IV`), not done.

---

## 4. EVERY MISSING / CONFLICTING EXPERIMENT DEFINITION
- **Two competing primary experiments:** `03_BC_roadmap_killtest.md` "Kill Test C" (arms: A-full / A-null / CLIP-TS; ≥3 families × ≥1000+1000; full grid; 7 modules) vs `EXPERIMENT_0_SPEC` (arms add **A-shrink, A-shuffle**; 1 family × ~400+400 sources; reduced grid; **4** modules). **Authority is not declared.** Pre-reg §3 (7 modules, no A-shrink) contradicts Experiment 0 (4 modules, A-shrink mandated).
- **A-shrink `s` is internally contradictory:** `EXPERIMENT_0 §1.3` calls it "a single global constant `s∈(0,1]`" but then "per laundering bin" — global vs per-bin is unresolved; the fit objective ("match mean |pooled-LLR|") needs a defined matching tolerance.
- **Gate thresholds partly missing:** G2 has no fit-error bound; G1's "1k-image fixture" is **never specified** (which images, where from); G4 "statistically significant gain" defers to the kill test but inherits the grid/arm ambiguity.
- **H6 (faithfulness)** has no concrete metric/protocol beyond "moves in predicted direction by >0."
- **CLIP-TS baseline is underspecified:** LR-probe hyperparameters, training split, temperature-fit set, OOD-gate (Mahalanobis) fit data — none given.
- **No committed seed / `run_config.yaml`** exists, though both are cited as the reproducibility root.
- **"Real-vs-sim 200-image spot check (KT/A2)"** is referenced as gating but never defined as a procedure.
- **Power analysis is hand-waved** (~80% power for ΔECE 0.04–0.05) with no reproducible computation or variance source for Experiment 0.

---

## 5. EVERY MODULE WHOSE BEHAVIOR IS NOT FULLY SPECIFIED
For each MVP module, the raw-feature→`evidence_score`/`confidence_score`/`direction`/`llr` mapping is the part that is missing. **`confidence_score` has no computation defined for any module.** Specifics:
- **0 Quality Profiler:** output schema undefined (§1.1). It is the gate source for all others ⇒ blocking.
- **1 Metadata/EXIF:** "satisfied constraints / total" names checks (camera↔lens↔resolution↔timestamp↔thumbnail↔maker-note) but each check's logic is undefined; "camera↔lens compatibility" requires a lens/mount reference DB that is not provided.
- **2 C2PA:** numeric `evidence_score` for valid/invalid/AI-asserting unspecified; trust-root/signer list undefined; partial-manifest behavior undefined.
- **3 JPEG DQ+grid:** "inconsistency score" + DQ-peak detection algorithm unspecified; the r→0 trigger depends on the undefined Q estimate (§1.1).
- **4 ELA:** "recompress Q=90, abs-diff, region-variance" → mapping to [0,1] and to a direction is undefined; ELA is qualitative by nature.
- **5 FFT azimuthal:** "peak/periodicity" → evidence undefined; contract says classical ⇒ `ood_score==0` (R7) yet text mentions a "learned-variant OOD" — is there a learned variant in MVP? Ambiguous.
- **6 Noise-residual/too-clean:** "residual energy + stationarity" → evidence undefined; "capped weight" cap value not given; collides with self-silencing (acknowledged in `01_critical_findings §3`) but no implementable separation rule.
- **All modules:** `localization` payload format (encoding, resolution, normalization, coordinate frame) undefined though modules 3/4/6 emit heatmaps.
- **Anti-circularity test T3 may be unimplementable as written:** it requires "changing only the pixel content that drives `evidence_score` while holding the quality vector fixed" — but the quality vector is computed *from pixels*, so any pixel change perturbs it. The operationalization is undefined.
- **Determinism invariant T5 ("byte-identical output for same input+seed")** is unattainable across BLAS/FFT threading and impossible with an API VLM; no tolerance is defined ⇒ the gate fails by construction.

---

## 6. PLACES WHERE TWO ENGINEERS WOULD DIVERGE (consolidated)
1. Quality-vector schema/normalization (§1.1).
2. `(alpha,beta)` derivation for reliability (§1.2).
3. `g()` in `min(a)·g(b)·g(c)` and the self-consistency/OOD sub-terms.
4. Transfer-function model class (§1.3).
5. Trust / Risk / verdict-band formulas and cutoffs (§1.4).
6. `mean_reliability` / `evidence_coverage` aggregation set (§1.5).
7. Abstention tuning objective (§1.6).
8. LLR clipping + per-module isotonic training set (§1.7).
9. evidence→direction sign mapping per module (§1.9).
10. **Backend language/stack** (Node vs Python), DB (Mongo vs SQL), queue (rq vs celery).
11. **Which degradation grid** (§3) and **which experiment spec** (§4).
12. exiftool vs exifread; VLM local vs API.
13. "screenshot-sim" / "double-compress" transforms (§3).

Each of the above produces a *different, non-comparable* system and *different numbers*.

---

## 7. MISSING ACCEPTANCE CRITERIA
- **No MVP-level "definition of done."** The only go/no-go (G4/kill test) gates *V2*, not MVP completion.
- **No latency budget** at module or end-to-end level (`AGENTS.md` gives DB/AI generic numbers; forensic-module budgets absent; reviews estimate 0.5–3 s but nothing is committed).
- **No Quality-Profiler accuracy tolerance** (e.g., JPEG-Q estimate error bound) — yet everything keys off it.
- **G2 transfer-function fit-error threshold** missing.
- **G1 fixture undefined** (the "1k images" are not enumerated) ⇒ G1 is not checkable.
- **Coverage >80% mandated** (`AGENTS.md`) with no defined testable surface or fixtures.
- **Narrative faithfulness** has no pass/fail metric ("faithful by construction" is asserted; `01_critical_findings` calls it an unsolved NLI problem).
- **No retrieval precision/recall, no UI/UX acceptance, no reproducibility tolerance** (bit-exact vs ε).
- **No error-rate/Daubert artifact** acceptance for the promised "frozen forensic mode."

---

## 8. HIDDEN ENGINEERING RISKS (implementation/reproducibility-specific)
*(Distinct from the scientific risks already enumerated by the repo's own red-team and independent reviews.)*
- **R-1 (blocking):** Experiment 0 and the gates assume a **pre-existing MVP codebase that is absent.** The "few engineer-days" estimate is anchored to nothing; realistic effort is weeks-to-months to reach G1.
- **R-2:** `>=` dependency ranges + native libs (faiss, c2pa, opencv) → **non-reproducible builds**; numpy-2 ABI break is a live hazard.
- **R-3:** **Determinism is over-promised** (T5 byte-identical) and physically unattainable with FFT/BLAS threading and API VLMs; the audit-log "reproducible verdict" claim breaks the moment the VLM module is on.
- **R-4:** **exiftool/c2pa-python packaging** in a container (Perl runtime; Rust native build; licensing) is unscoped.
- **R-5:** **No CI/quality gates exist** despite mandates for mypy/pylint/black/coverage — no configs, no pipeline.
- **R-6:** **Retrieval corpus + temporally-grounded index do not exist publicly** (the repo admits this) — the FAISS module has no data to index; open_clip version bumps silently invalidate any index built.
- **R-7:** **Concurrency model undefined** ("parallel modules" vs Python GIL + heavy CV/VLM) — no worker/async design, no memory budget for co-resident CLIP+FAISS+VLM on "commodity hardware."
- **R-8:** **Secrets handling** for the LLM key is unspecified (env/vault), though `AGENTS.md` demands secrets management.
- **R-9:** **Dataset licensing/redistribution** (RAISE/Dresden/Flickr/GenImage + GPL exiftool bundling) is an unaddressed legal/operational risk for any release.
- **R-10:** **MongoDB audit schema undefined**; "store full evidence JSON + versions + timestamps" with reproducibility but no schema, indexing, or retention design.

---

## 9. IMPLEMENTATION READINESS SCORE

### **38 / 100 — NOT implementation-ready.**

| Dimension | Weight | Score | Rationale |
|---|:--:|:--:|---|
| Requirement / interface clarity | 18 | 6 | Core interfaces (quality vector, reliability Beta, Trust/Risk, bands) undefined |
| Dependency & environment reproducibility | 14 | 3 | No lockfile; `>=` pins; contradictory stacks; native libs unscoped |
| Dataset readiness | 16 | 4 | Pools partly "to be built"; no manifests; conflicting grids; LOGO families unnamed |
| Experiment definition consistency | 12 | 6 | Strong intent, but two conflicting specs; missing seed/config/thresholds |
| Module behavioral completeness | 14 | 5 | evidence/confidence/direction mappings + localization formats missing |
| Acceptance criteria & test fixtures | 10 | 3 | No MVP DoD, no latency budget, undefined G1 fixture, no faithfulness metric |
| Repro/CI/MLOps infrastructure | 8 | 1 | None present; determinism over-promised |
| Architectural & scientific documentation (frozen) | 8 | 8 | Genuinely excellent; pre-reg/SAP/label-spec/contract are best-in-class |
| **Total** | **100** | **38** | |

**Why not lower:** the validation set (pre-registration, statistical analysis plan, label spec, module contract, datasheet) is far above industry norm and removes most *methodological* ambiguity — a real asset. **Why not higher:** none of that is *buildable as written* — there is no code, no environment, no data, the two most load-bearing interfaces (quality vector, reliability→Beta) are undefined, two stacks contradict, and two experiments contradict. A second team handed this repo would build a *different, non-comparable* system.

---

## 10. PRE-CODING CHECKLIST (must be closed before a line is written)

**A. Decisions / freezes (1–2 days, mostly choices)**
- [ ] **A1** Pick ONE stack: backend language (Python-FastAPI per MVP **or** Node per frontend constitution), DB (MongoDB), queue (rq **or** celery), and retire the contradictory constitution. Record an ADR.
- [ ] **A2** Declare the **canonical experiment** (Kill Test C **or** Experiment 0) and the **canonical degradation grid**; reconcile Pre-reg §3 module count with it.
- [ ] **A3** Choose exiftool **xor** exifread/piexif; choose VLM-local **xor** API (and accept the determinism consequence); confirm conformal IN/OUT of MVP.

**B. Interface specifications (the blocking gaps)**
- [ ] **B1** Freeze the **quality-vector schema**: every field, type, range, normalization, and the exact algorithm for Q-estimation, double-compression, blockiness, blur, noise-floor, screenshot heuristic (with thresholds).
- [ ] **B2** Specify the **reliability pipeline end-to-end**: `g()`, the (a)/(b)/(c) formulas, **and the `(quality_vector → alpha, beta)` estimator** so R2/T7 are satisfiable. Reconcile product-form vs Beta-form.
- [ ] **B3** Define **Trust**, **Risk** (incl. `harm_weight` source or its removal for MVP), and the **verdict-band cutoffs** as concrete formulas.
- [ ] **B4** Define `mean_reliability`/`evidence_coverage` aggregation sets, the **abstention tuning objective**, and **LLR clipping bounds**.
- [ ] **B5** For **each module**: the raw-feature→`evidence_score`→`direction` mapping, the `confidence_score` formula, and the `localization` payload format.
- [ ] **B6** Re-specify or relax **T3 (anti-circularity)** and **T5 (determinism)** into *implementable* tests with tolerances.

**C. Dependencies / environment**
- [ ] **C1** Produce a real **lockfile** (`requirements.txt` via `pip freeze` / `poetry.lock`) with exact versions and weight hashes (open_clip), plus a base container image and (optional) CUDA matrix.
- [ ] **C2** Containerize and smoke-test the **fragile natives** (c2pa-python, exiftool, faiss) before any module work.
- [ ] **C3** Stand up **CI** (black/mypy/pylint/pytest+coverage) and a secrets mechanism for the LLM key.

**D. Data**
- [ ] **D1** Acquire/cut and **manifest (SHA-256)** every pool; pin exact RAISE/Dresden/Flickr/GenImage/DiffusionForensics versions; **name the held-out LOGO families**; fix per-pool class balance and the eval prior value.
- [ ] **D2** Define the **screenshot-sim and double-compress algorithms** with parameters.
- [ ] **D3** Build **Pool 4 (suspicious-but-real)** and source **Pool 5 (physics-aware)**; complete licensing/consent records.
- [ ] **D4** Specify the **G1 1k-image fixture** (which images), and the per-module isotonic **calibration training split**.

**E. Acceptance & reproducibility**
- [ ] **E1** Write the **MVP Definition of Done** (functional + metric gates), a **latency budget** per module and end-to-end, and the **Quality-Profiler accuracy tolerance**.
- [ ] **E2** Set **G2 fit-error threshold**, the reproducibility tolerance (bit-exact vs ε), and a concrete **faithfulness metric** (or downgrade the narrative to clearly non-scoring).
- [ ] **E3** Commit the **frozen `run_config.yaml`** (seed, splits, thresholds) and the **MongoDB audit schema**.

**Gate to begin coding:** A1–A3, B1–B5, C1, D1, D4, E1, E3 closed. (B6/C2/C3/D2/D3/E2 may run in parallel with early classical-module work but block G1/G2 sign-off.)

---
*Scope note: per mandate, this review treats the architecture as frozen and proposes no new forensic modules. Every item above is an implementation-readiness, reproducibility, or scientific-validity-of-execution gap, not an architectural critique.*
