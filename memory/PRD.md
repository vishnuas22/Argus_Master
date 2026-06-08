# PRD — ARGUS Authenticity Assessment Platform

## Original problem statement
Design a next-generation image authenticity assessment platform (ARGUS) that reframes deepfake "detection" into reliability-aware, explainable, uncertainty-quantified evidence reasoning that generalizes to unseen generators and survives social-media laundering. Delivered as a 10-phase research + architecture document.

## Current deliverable status (June 2026)
- **DONE — Phase 1: written design document only** (user choice: "Just a written document (Markdown/PDF), no app", "focus only on thorough documentation now before building").
- Outputs:
  - `/app/docs/ARGUS/ARGUS_Design_Document.md` (combined, ~12k words)
  - `/app/docs/ARGUS/ARGUS_Design_Document.pdf` (34 pages, dark forensic-console theme)
  - Per-phase source markdown files `00..09`
  - `/app/docs/ARGUS/build_pdf.py` (regenerates the styled PDF)
- Covers all 10 phases: problem deconstruction, first-principles design, evidence taxonomy + scorecard, reliability contract, fusion strategy, unknown-generator resistance, XAI, 3-month MVP, research/IP, hostile review + redesign, final score 84/100.

## User-confirmed preferences for future build (NOT yet built)
- Visual style: dark "forensic lab / security console" aesthetic.
- LLM for verdict narration: Gemini / Groq (user will provide key).
- No user accounts; single-page tool.
- Depth v1 when built: classical forensics first, accurate results before ML.

## Backlog / Next action items (P0 → P2)
- **P0:** Build functional ARGUS MVP demo app (upload image → run real classical forensic modules: metadata/EXIF, C2PA, ELA, JPEG-grid/double-quant, FFT/frequency, noise residual; output Authenticity/Trust/Risk + evidence ranking + grounded explanation). Dark forensic console UI.
- **P1:** Add CLIP+FAISS retrieval/near-dup module; reliability layer with degradation transfer functions; Bayesian+LightGBM fusion + conformal abstention.
- **P1:** Integrate Gemini/Groq grounded narrative (user-provided key) for human-readable verdict.
- **P2:** VLM physics/semantic module (geometrically verified), OOD-gated learned probe, LOGO + laundering evaluation harness.

## Notes
- No app/services were created in this iteration; deliverable is documentation only. No backend/frontend changes; nothing to test via testing agent.
