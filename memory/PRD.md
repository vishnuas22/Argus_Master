# ARGUS — Product Requirements & Memory

## Original Problem Statement
Design a next-generation image authenticity assessment platform ("ARGUS") via a 10-phase deep-research design exercise: problem deconstruction, first-principles architecture, evidence taxonomy (22 sources scored), reliability-aware module design, fusion strategy comparison, unknown-generator resistance, production XAI, 3-month MVP plan, research-contribution analysis, and hostile self-review with final /100 score. User explicitly asked for deep research + better-method suggestions BEFORE documentation.

## User Choices
- Deliverable: comprehensive research/design document ONLY (no app build this session)
- Research findings presented first → user approved approach
- Depth: deep & exhaustive
- Format: Markdown files in repo
- (If MVP app ever built: real working forensic modules on CPU — metadata/EXIF, ELA/JPEG ghost, FFT, noise residual, DINOv2 probe)

## What's Been Implemented (June 2026)
- Deep web research (7 searches): training-free detectors (SpAN, RIGID-class, ZED), DINOv2 vs CLIP robustness (92% vs 42% under transforms), TruFor/Noiseprint++, conformal prediction, C2PA 2026 status (spec v2.4, camera adoption), fusion literature (FRAME, MoE, Dempster-Shafer), physics forensics (Light2Lie NDSS 2026)
- 12 markdown docs created in /app/docs/:
  - 00_EXECUTIVE_SUMMARY.md (thesis, ASCII architecture, decision table)
  - 01–10 phase docs (all 10 phases, deep)
  - REFERENCES.md
- Core architecture: 3-tier "degradation-aware evidence court" — Tier 0 C2PA fast-path, Tier 1 degradation triage → state vector d, Tier 2 nine-module evidence panel with {evidence, reliability=r_m(d), confidence} contract, Tier 3 reliability-gated LightGBM stacking + isotonic + conformal abstention + verdict schema
- Key novelty claimed: degradation-conditioned reliability calibration r_m(d); conformal authenticity verdicts; evidence-court XAI schema
- Final hostile-review architecture score: 86/100 (itemized losses)

## Prioritized Backlog
- P0: (if user wants) Build ARGUS MVP web app per Phase 8 spec — FastAPI + React, modules A/B/C (metadata, ELA/ghost, FFT) are pure CPU and buildable here; DINOv2 probe via timm CPU
- P1: Interactive document site / export to PDF
- P2: LaunderBench simulator prototype; retrieval module design detail

## Notes
- No code/app exists yet; /app is still the scaffold. Docs live in /app/docs/.
