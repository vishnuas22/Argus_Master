# ARGUS MVP — Master Execution Prompt

> Copy-paste this prompt (in full, or milestone-by-milestone) to drive the build. It is engineered to minimize errors: every instruction binds the implementation to the design docs, enforces contracts before features, and gates each milestone with tests before the next begins.

---

## THE PROMPT

```
You are building the ARGUS MVP — an image authenticity assessment platform.

══════════════════════════════════════════════════════════════
SOURCE OF TRUTH (read before writing any code)
══════════════════════════════════════════════════════════════
The complete design lives in /app/docs/. It is binding, not advisory:
- 02_PHASE2_FIRST_PRINCIPLES_DESIGN.md  → the 3-tier architecture (Tier 0 provenance → Tier 1 degradation triage → Tier 2 evidence panel → Tier 3 court)
- 04_PHASE4_RELIABILITY_ARCHITECTURE.md → the MODULE OUTPUT CONTRACT (section 4.1 JSON) — every evidence module MUST emit exactly this schema
- 05_PHASE5_EVIDENCE_FUSION.md          → fusion = reliability gate → calibrated scoring → verdict (weighted-voting fallback is MANDATORY)
- 07_PHASE7_EXPLAINABILITY.md           → the verdict.json schema (section 7.1) — the API's response contract
- 08_PHASE8_MVP_3_MONTHS.md             → exact modules, libraries, and scope cuts
- 10_PHASE10_HOSTILE_REVIEW_AND_REDESIGN.md → known failure modes; do not reintroduce them

If code and docs conflict, the docs win. If a doc is ambiguous, choose the
simplest interpretation that preserves the module contract, and record the
decision in /app/memory/DECISIONS.md.

══════════════════════════════════════════════════════════════
NON-NEGOTIABLE ENGINEERING STANDARDS
══════════════════════════════════════════════════════════════
1. CONTRACT FIRST. Before any module logic, implement in /app/backend:
   - pydantic models: ModuleOutput, Artifact, DegradationState, Verdict
     (field names and ranges exactly as docs 4.1 and 7.1)
   - an abstract EvidenceModule base class:
       run(image, degradation_state) -> ModuleOutput
     with built-in: timing, exception capture (a crashing module returns
     unavailable_reason="internal_error", NEVER crashes the pipeline),
     and reliability lookup r_m(d).
   Every module is a separate file in backend/modules/, registered in a
   plugin registry. No module imports another module.

2. FAIL-CLOSED, NEVER FAIL-SILENT. A module that cannot assess (no shadows
   found, no text found, image too degraded) returns reliability ≈ 0 with
   unavailable_reason set. The fusion layer lists it under
   unavailable_evidence. The pipeline NEVER hides a dead module.

3. THREE SCORES, NEVER COLLAPSED. evidence_score ∈ [-1,1],
   reliability_score ∈ [0,1], confidence_score ∈ [0,1] stay separate until
   the fusion layer. Any code that pre-multiplies them inside a module is a
   contract violation.

4. EVERY ARTIFACT IS CHECKABLE. Each artifact carries a checkable_claim
   string and, where applicable, a saved visual (PNG overlay) under
   /app/backend/artifacts/{verdict_id}/. No claim without a basis a human
   can verify.

5. DETERMINISM & VERSIONING. Same image in → same verdict out. Every verdict
   embeds module_versions and pipeline version. Seed all randomness.

6. ENVIRONMENT RULES. Backend: FastAPI on 0.0.0.0:8001, all routes prefixed
   /api, MongoDB via MONGO_URL env only (store verdicts, never images
   beyond processing). Frontend: React, calls REACT_APP_BACKEND_URL only.
   Every interactive/info element gets a kebab-case data-testid.

7. CPU-ONLY BUDGET. Total panel ≤ 10 s per image on CPU. DINOv2 loads once
   (singleton service), shared by the realness probe and perturbation probe.

══════════════════════════════════════════════════════════════
BUILD ORDER — MILESTONES WITH HARD GATES
Do not start milestone N+1 until milestone N passes its gate.
══════════════════════════════════════════════════════════════

M1 — SKELETON & CONTRACT
- pydantic schemas, EvidenceModule base, plugin registry, pipeline runner
- POST /api/assess (multipart image) → runs registered modules → returns a
  valid Verdict JSON even with ZERO real modules (use one stub module)
- GET /api/verdicts/{id}, GET /api/verdicts (history from MongoDB)
GATE: curl an image through /api/assess; response validates against the
Verdict schema; stub module crash test → pipeline survives, reports
unavailable_reason.

M2 — TIER 1 TRIAGE + CHEAP MODULES (A, B, C)
- DegradationEstimator: JPEG quality (quant tables via Pillow), double-JPEG
  heuristic, resize detection (FFT residual peaks), screenshot heuristic
  → DegradationState d
- Module A metadata: pyexiftool dump + ~15 rule checks (software strings,
  missing-camera-metadata-on-claimed-photo, date logic, thumbnail mismatch)
- Module B compression history: ELA (recompress q90, amplified diff PNG) +
  JPEG-ghost sweep; outputs heatmap artifact
- Module C spectral probe: FFT radial power spectrum vs envelope; saves
  polar spectrum plot; MUST discount its own evidence when d shows resize
  (the docs' resize-disambiguation rule)
- Reliability curves v0: hand-authored monotonic lookup tables per module
  over d (documented in code as lc-v0; replaced by calibrated curves later)
GATE: 3 test images (pristine real JPEG, AI-generated PNG, recompressed
real) produce directionally sensible, DIFFERENT module outputs; the
recompressed image must show DROPPED reliabilities, not changed evidence.

M3 — DINOv2 PROBES (E, F)
- Singleton DINOv2 ViT-B/14 (timm, CPU) embedding service
- Module E realness probe: kNN distance percentile against a bundled
  reference set (start small: 2–5k diverse real-image embeddings,
  precomputed and shipped as a .npy — document the small-N caveat in the
  module's confidence_score); artifact = nearest-neighbor distances
- Module F perturbation-sensitivity probe: cosine-similarity drop under
  structured high-frequency noise + blur variant (RIGID-style), thresholds
  calibrated on the same reference set
GATE: E and F separate a 20-real/20-AI smoke-test set with AUROC > 0.8
combined; latency for full panel still ≤ 10 s.

M4 — THE COURT (FUSION + VERDICT)
- Reliability gate (floor 0.25, gated modules → unavailable_evidence)
- Fusion v0 = reliability-weighted voting (the mandated fallback) over
  surviving modules → calibrated to [0,1] via logistic squashing
- Conflict metric K (docs 4.4.1) + contradictions[] builder using the
  pattern table in docs 4.4.2
- trust_score and risk_score per docs 7.2 semantics
- Template-based explanation generator: slot-filled ONLY from verdict
  fields (docs 7.3 — no free text generation)
- Upgrade path stub: fusion interface accepts a pluggable judge so
  LightGBM + conformal can replace voting later WITHOUT API changes
GATE: end-to-end verdict on the smoke-test set; hand-audit 5 verdicts —
every sentence in the explanation must trace to a JSON field; the
laundered image must yield lower trust_score with the SAME image content.

M5 — ANALYST UI
- React: upload zone → progress per tier → verdict dashboard
- Three score dials (authenticity, trust, risk) with the docs' semantics
  shown on hover; conformal/abstention badge
- Evidence cards sorted by ranking: direction, the three scores, expandable
  checkable_claim, artifact image viewer (ELA heatmap, spectrum plot,
  NN panel) as toggleable overlays on the original image
- contradictions[] and unavailable_evidence[] rendered as first-class
  sections (never hidden); verdict JSON download button
- History page from /api/verdicts
GATE: full user journey via screenshot test: upload AI image → see
synthetic verdict with ≥3 evidence cards and 1 viewable artifact overlay;
upload real photo → authentic-leaning verdict; data-testids on all
interactive elements.

M6 — HARDENING & REGRESSION SUITE
- /app/tests/regression_set/: ≥10 reals, ≥10 AI, ≥5 laundered variants
  with expected verdict directions in a manifest.json
- pytest suite: schema validation, module crash isolation, determinism
  (same image twice → identical verdict), laundering monotonicity
  (more laundering → trust_score never increases)
- Run full testing agent pass (backend + frontend)
GATE: regression suite green; confident-wrong count on regression set = 0
(abstentions allowed, wrong-with-high-trust not).

══════════════════════════════════════════════════════════════
CONTINUOUS-BUILD RULES (apply every session after MVP)
══════════════════════════════════════════════════════════════
- Any new evidence module = new file implementing EvidenceModule + entry in
  registry + 3 regression images + reliability table. Nothing else changes.
- Any change to schemas requires bumping schema_version and updating BOTH
  docs 4.1/7.1 and the pydantic models in the same commit.
- After every feature: run the regression suite BEFORE declaring done.
- Never optimize a module past its gate criteria while other milestones
  are incomplete — breadth before depth, per docs 08 scope discipline.
- Log every deliberate deviation from /app/docs in /app/memory/DECISIONS.md
  with one-line rationale.
```

---

## How to use it

| Situation | What to paste |
|---|---|
| Kick off the build | The whole prompt above (one message) |
| Resume after a break | "Continue ARGUS build per /app/docs/11_MVP_EXECUTION_PROMPT.md — we completed M{n}, gate passed. Start M{n+1}." |
| Add a feature later | "Per the CONTINUOUS-BUILD RULES in /app/docs/11_MVP_EXECUTION_PROMPT.md, add module: {name}. Follow the module contract." |
| Something broke | "Regression suite failing on {case}. Debug per FAIL-CLOSED rule — find which module violated its contract before touching fusion." |

## Why this prompt reduces errors

1. **Contract-first ordering** — the #1 source of multi-module build errors is schema drift; M1 freezes the contracts before any forensic logic exists.
2. **Hard gates** — each milestone has a falsifiable exit test, so errors are caught within the milestone that caused them, never three milestones later.
3. **Crash isolation by design** — a buggy module degrades to `unavailable`, never takes down the pipeline; debugging is localized by construction.
4. **Determinism + regression set** — every later change is checked against frozen expected behavior, enabling continuous building without silent regressions.
5. **Docs-as-law clause** — prevents the implementer (human or AI) from "improving" the architecture mid-build, which is how reliability semantics get silently collapsed.
