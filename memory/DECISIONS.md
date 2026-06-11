# ARGUS — Deliberate deviations & ambiguity resolutions (docs-as-law log)

| Date | Decision | Rationale |
|---|---|---|
| 2026-06-10 | Tier 0 C2PA provenance deferred | Not in the M1–M6 milestone list of docs 11; metadata module covers XMP `trainedAlgorithmicMedia` markers meanwhile. |
| 2026-06-10 | `verdict_id = "argus-" + sha256[:16]` instead of date+counter | Determinism standard #5: same image in → byte-identical forensic content out; re-assessment overwrites the same verdict. |
| 2026-06-10 | Determinism scope excludes timing fields (`compute_ms`, `total_compute_ms`, `created_at`) | Wall-clock timing is inherently non-deterministic; all forensic content (scores, artifacts, explanation) is exact-deterministic. |
| 2026-06-10 | `shap_contribution` in v0 = exact additive contribution of the voting judge (r·e/Σr) | Field name fixed by docs 7.1 contract; SHAP proper arrives with the LightGBM judge; voting contribution is the exact analogue. |
| 2026-06-10 | `likelihood_ratio` v0 = exp(3.6·\|e\|·r), heuristic | Voting fallback has no likelihood model; mapping reproduces docs 7.1 worked example (e=-0.78, r=0.74 → LR≈8). Replaced when calibrated fusion lands. |
| 2026-06-10 | Conformal-stub = APS-style cumulative-probability set (α=0.10) | M4 mandates only the voting fallback + upgrade-path stub; MAPIE split-conformal slots into the same `conformal` field later. |
| 2026-06-10 | DINOv2 embedded at 224px (not 518px default) | CPU ≤10s panel budget (standard #7); reference set and thresholds calibrated at the same size, so consistent. |
| 2026-06-10 | Reference set ~2.5k COCO val2017 embeddings (small-N) | Docs M3 allows 2–5k; small-N caveat capped into module E confidence (≤0.65) and stated in its checkable_claim. |
| 2026-06-10 | AI test corpus generated via gpt-image-1 (Emergent LLM key), COCO-caption-style prompts | No public AI-image dataset bundled; caption-matched prompts follow docs 10 S4 de-confounding guidance. |
| 2026-06-10 | Resize factor estimate maps strongest residual peak f → 1/f (upsampling assumption) | Gallagher peak→factor inversion is ambiguous for downscales; v0 reports presence + rough factor; reliability curves only consume \|factor−1\|. |
| 2026-06-10 | Recompression generations via JPEG-ghost minima (not DCT-histogram DQ) | Docs M2 says "double-JPEG heuristic"; ghost minima are the simplest interpretation preserving the contract; jpegio DQ analysis is a post-MVP upgrade. |
| 2026-06-10 | Frontend per-tier progress is client-side staged animation | /api/assess is a single synchronous call at MVP; no streaming endpoint in scope. |
