"# 15 — Evaluation Protocol (how the numbers are produced)

> **Status:** Implementation-ready. The \"how-we-measure\" companion to `14_accuracy_playbook.md` (the why) and `17_evaluation_and_benchmarks.md` (the GoldenEval bench).
> **Supersedes:** nothing — fills the gap between docs 14 and 17. Doc 14 §3, §6.4, §10 and doc 16 §1.7, §3.1, §5, §6 all reference this file by number; until now it did not exist.
> **Last updated:** 2026-02 (v1.5)
> **Prereq for:** every AUROC / ECE / coverage / FP-on-REAL number cited in 14, 16, 17.

---

## 0. Why this file exists

Doc 14 §3 quotes lifts like \"+0.04 AUROC ± 0.02 (median of 10 bootstrap resamples)\" without naming a sourced protocol. Doc 16 §1.7 / §5 / §6 cite slice-AUROC deltas with the same wave-of-the-hand. Doc 17 quotes \"macro AUROC ≥ 0.85 cloud_lite\" as a gate but never says *which set, which splits, which bootstrap, which seed, which signals are gated on/off* the numbers come from. Without a frozen protocol, every cited number is an aspiration and cannot be challenged by a reviewer.

This document is that protocol.

---

## 1. Non-negotiable rules

1. **Holdout disjointness.** Every set used to compute a published metric must be SHA-disjoint and pHash-disjoint and CLIP-near-duplicate-disjoint from every set used to fit any model parameter (Platt scaler, distillation head, OOD IsolationForest, conformal quantile, content-type router, PRNU logistic). Enforced by `20_refdb_eval_disjointness.md`. CI fails the merge if violated.
2. **Frozen seed.** All bootstrap resamples use `numpy.random.default_rng(seed=20260201)`. The seed is part of the protocol; changing it requires a doc-version bump (`15.x`) and a re-run of every quoted number.
3. **Frozen pipeline.** Every published metric is computed against a pipeline configuration snapshot (`eval/pipeline_configs/<ts>.json`) that records: enabled detectors, model SHAs, fusion mode, calibration source, gate thresholds, env variables. Re-running with a drifted pipeline produces a *new* report — old reports are not silently overwritten.
4. **Public-only data.** Every sample used in a published metric is downloadable by an external reviewer via the manifests in `17 §3`. No private holdouts.
5. **Stratified bootstrap, not k-fold.** GoldenEval is too small for k-fold to give a tight CI; stratified bootstrap on a frozen row-CSV produces a tighter and reproducible 95 % CI.

---

## 2. The two evaluation sets

| Set | Purpose | Size (image, M3 first-finish) | Lifecycle |
|---|---|---|---|
| **refDB-holdout-fold** | Per-signal Platt fit AUROC, content-type router accuracy, OOD IsolationForest threshold tuning, distillation-head CV AUROC | 20 % of refDB = 1 000 real + 1 000 AI | Re-fit on every refDB rebuild |
| **GoldenEval v1** | All **publishable** KPIs (overall AUROC, per-slice AUROC, ECE, coverage, FP-on-REAL, adversarial bench) | 850 real + 900 AI = 1 750 | Frozen; bump to v2 changes manifest hash |

`refDB-holdout-fold` shares its source distribution with refDB-train and therefore reports **optimistically** — it is fine for model-internal hyper-parameter selection (which Platt slope, which OOD τ) and **forbidden** as a source for any externally-quoted number.

GoldenEval is the only set where the headline KPI may be quoted in:
- The README banner
- The `Masterplan §1.5` revised-targets table
- Doc 14 §3, doc 16 §5
- The `/api/health` `eval_mini` field

Anything else is **internal** and must be annotated `(internal, refDB-holdout)` in the report.

---

## 3. Bootstrap protocol

### 3.1 Why bootstrap and not k-fold

GoldenEval is fixed at 1 750 samples and stratified across 16 slugs × 2 labels. K-fold on 1 750 leaves ~ 175 samples per fold → fold-level AUROC variance is so high that fold-mean ± half-width is meaningless. Stratified bootstrap with replacement on the **frozen p_ai array** gives a tighter, reproducible 95 % CI at zero extra pipeline runtime.

### 3.2 Exact procedure

```
seed = 20260201
B    = 1000                              # bootstrap resamples
For b in 1..B:
    Sample N rows from GoldenEval WITH REPLACEMENT, stratified by (label, slug)
    Compute the metric on resample b
median = np.median(samples)              # point estimate (PUBLISHED)
ci95_lo, ci95_hi = np.percentile(samples, [2.5, 97.5])
half_width = (ci95_hi - ci95_lo) / 2     # the \"± value\" in published tables
```

`B = 1000`, not 10 as the original `14 §3` footer hinted. Wall-clock on the cached `p_ai` array is ~ 3 s for AUROC, ~ 8 s for ECE.

### 3.3 What is the \"resample\"?

The pipeline does **not** re-run 1 000 times. The pipeline runs **once** per GoldenEval sample (via `run_goldeneval.py` in `17 §4`), producing a JSONL \"row-CSV\" of
`(sample_id, slug, label, content_type, p_ai, abstained, conformal_set, signals[...])`.

The bootstrap operates purely on that JSONL — it samples row indices with replacement and recomputes each metric. Cheap and reproducible — the same JSONL input must produce byte-identical bootstrap JSON output.

### 3.4 Implementation blueprint — `backend/scripts/eval/bootstrap_metrics.py`

```python
# file: /app/backend/scripts/eval/bootstrap_metrics.py
\"\"\"Bootstrap point estimates + 95 % CIs for every published KPI.

Consumes the row-JSONL emitted by run_goldeneval.py; emits a JSON + Markdown report.
Reproducible: same input → byte-identical output.\"\"\"
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

log = logging.getLogger(\"eval.bootstrap\")

SEED = 20260201
B = 1000


def _resample(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    \"\"\"Stratified bootstrap by (label, slug).\"\"\"
    by_strat: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        by_strat.setdefault((r[\"label\"], r[\"slug\"]), []).append(i)
    idx: list[int] = []
    for indices in by_strat.values():
        idx.extend(rng.choice(indices, size=len(indices), replace=True).tolist())
    return [rows[i] for i in idx]


def _auroc(rs: list[dict]) -> float | None:
    y = np.array([r[\"label\"] for r in rs]); p = np.array([r[\"p_ai\"] for r in rs])
    if len(set(y.tolist())) < 2: return None
    return float(roc_auc_score(y, p))


def _aupr(rs: list[dict], pos_class: int) -> float | None:
    y = np.array([r[\"label\"] for r in rs])
    p = np.array([r[\"p_ai\"] for r in rs])
    if pos_class == 0:
        y, p = 1 - y, 1 - p
    if len(set(y.tolist())) < 2: return None
    return float(average_precision_score(y, p))


def _brier(rs: list[dict]) -> float:
    y = np.array([r[\"label\"] for r in rs], dtype=float)
    p = np.array([r[\"p_ai\"] for r in rs], dtype=float)
    return float(np.mean((p - y) ** 2))


def _ece(rs: list[dict], bins: int = 15) -> float:
    p = np.array([r[\"p_ai\"] for r in rs]); y = np.array([r[\"label\"] for r in rs])
    edges = np.linspace(0, 1, bins + 1); n = len(p); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0: continue
        e += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(e)


def _coverage(rs: list[dict]) -> float | None:
    if not rs: return None
    return float(np.mean([
        (\"ai\" if r[\"label\"] == 1 else \"real\") in r[\"conformal_set\"]
        for r in rs
    ]))


def _accuracy_non_abstained(rs: list[dict], thr: float = 0.5) -> float | None:
    keep = [r for r in rs if not r[\"abstained\"]]
    if not keep: return None
    correct = sum(
        1 for r in keep
        if (r[\"p_ai\"] >= thr) == (r[\"label\"] == 1)
    )
    return float(correct / len(keep))


def _deferral_rate(rs: list[dict]) -> float:
    return float(np.mean([r[\"abstained\"] for r in rs]))


def _fpr_on_real_at_p50(rs: list[dict], thr: float = 0.5) -> float | None:
    real_kept = [r for r in rs if r[\"label\"] == 0 and not r[\"abstained\"]]
    if not real_kept: return None
    return float(sum(1 for r in real_kept if r[\"p_ai\"] >= thr) / len(real_kept))


METRICS = {
    \"auroc\":                    _auroc,
    \"aupr_ai\":                  lambda rs: _aupr(rs, pos_class=1),
    \"aupr_real\":                lambda rs: _aupr(rs, pos_class=0),
    \"brier\":                    _brier,
    \"ece_15\":                   _ece,
    \"coverage\":                 _coverage,
    \"accuracy_non_abstained\":   _accuracy_non_abstained,
    \"deferral_rate\":            _deferral_rate,
    \"fpr_on_real_at_p50\":       _fpr_on_real_at_p50,
}


def _summarise(samples: list[float | None]) -> dict | None:
    vals = [s for s in samples if s is not None]
    if not vals: return None
    return {
        \"point\":      float(np.median(vals)),
        \"ci95_lo\":    float(np.percentile(vals, 2.5)),
        \"ci95_hi\":    float(np.percentile(vals, 97.5)),
        \"half_width\": float((np.percentile(vals, 97.5) - np.percentile(vals, 2.5)) / 2),
        \"n_bootstraps\": len(vals),
    }


def main(rows_path: Path, out: Path, by_slug: bool = True) -> None:
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    rng = np.random.default_rng(SEED)
    overall: dict[str, dict | None] = {}
    for name, fn in METRICS.items():
        s = [fn(_resample(rows, rng)) for _ in range(B)]
        overall[name] = _summarise(s)

    per_slug: dict[str, dict] = {}
    if by_slug:
        slugs = sorted({r[\"slug\"] for r in rows})
        for slug in slugs:
            sub = [r for r in rows if r[\"slug\"] == slug]
            if len(sub) < 20:
                continue
            rng_s = np.random.default_rng(SEED ^ hash(slug) & 0xFFFFFFFF)
            per_slug[slug] = {
                \"n\": len(sub),
                \"auroc\":  _summarise([_auroc(_resample(sub, rng_s)) for _ in range(B)]),
                \"ece\":    _summarise([_ece(_resample(sub, rng_s)) for _ in range(B)]),
            }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        \"seed\": SEED, \"B\": B,
        \"n_rows\": len(rows),
        \"overall\": overall,
        \"per_slug\": per_slug,
    }, indent=2))
    log.info(\"wrote %s\", out)


if __name__ == \"__main__\":
    ap = argparse.ArgumentParser()
    ap.add_argument(\"--rows\", type=Path, required=True)
    ap.add_argument(\"--out\",  type=Path, required=True)
    ap.add_argument(\"--no-per-slug\", action=\"store_true\")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format=\"%(asctime)s %(levelname)s %(name)s %(message)s\")
    main(args.rows, args.out, by_slug=not args.no_per_slug)
```

### 3.5 Row-JSONL schema (frozen)

Each line of `storage/eval/rows/<ts>_rows.jsonl` is a JSON object:

```json
{
  \"sample_id\":      \"gen_dalle3_00042\",
  \"slug\":           \"dalle3\",
  \"label\":          1,
  \"content_type\":   \"object_product\",
  \"adversarial_tag\": null,
  \"p_ai\":           0.871,
  \"abstained\":      false,
  \"conformal_set\":  [\"ai\"],
  \"novel_generator_suspected\": false,
  \"provenance_hit\": false,
  \"signals\":        {
      \"img.prithiv\":     0.83,
      \"img.frequency\":   0.71,
      \"img.clip0\":       0.74,
      \"img.meta\":        0.62,
      \"img.compression\": 0.55,
      \"img.ocr_gibberish\": null,
      \"img.eye_forensics\": null,
      \"img.prnu\":        0.59,
      \"tp.hive\":         0.81,
      \"tp.sightengine\":  0.78,
      \"tp.aiornot\":      0.74,
      \"retrieval.knn\":   0.85,
      \"retrieval.patch\": 0.82,
      \"reverse.serpapi\": null,
      \"vlm.gemini\":      0.88,
      \"meta.distill_lr\": 0.86
  },
  \"dur_ms\":          12473
}
```

`null` for a signal means \"not enabled for this sample\" (e.g. `eye_forensics` on a landscape). It is **distinct** from `0.5` (neutral imputation) and must remain `null` in the row-JSONL — the bootstrap metric functions handle absence correctly.

---

## 4. The signal-error correlation matrix

### 4.1 Why we build it

Per `14 §1`, the COEF lift formula holds when errors are **decorrelated**. The cross-modal multiplicative bonus (`Masterplan §3.6`) assumes signals are **independent** given `y`. The distillation head and the conformal quantile are sensitive to over-counted correlated evidence.

We **empirically audit** the independence assumption on GoldenEval after every refDB rebuild. Without this audit, the bonus and the conformal width are systematically over-confident.

### 4.2 The matrix

For every ordered pair of signals (i, j):

```
err_i_k = 1 if (p_i_k >= 0.5) != y_k else 0      # signal i is wrong on sample k
rho_ij  = pearsonr(err_i, err_j)
```

`rho_ij ∈ [-1, +1]`.
- `≈ 0` → decorrelated (good).
- `> +0.7` → redundant (one slot should be dropped or merged through a shared Platt cluster).
- `< -0.3` → systematically opposing (suspicious — usually a sign-bug in one detector).

### 4.3 Action thresholds (locked)

| `rho_ij` band | Action |
|---|---|
| `[-0.3, +0.5]` | Keep both, separate fusion-vector slots. |
| `(+0.5, +0.7]` | Keep both, **down-weight the higher-cost one to 0.5 in the cross-modal bonus**. Logged as `corr_warn`. |
| `(+0.7, +1.0]` | **Merge.** Both go through a single shared Platt cluster; only one slot in the fusion vector. Recorded in `backend/fusion/clusters.json`. Logged as `corr_merge`. |
| `[-1.0, -0.3)` | **Block-the-merge.** Investigate polarity bug in one detector. Logged as `corr_polarity_warn`. |

### 4.4 `backend/fusion/clusters.json` schema

```json
{
  \"version\": 1,
  \"clusters\": [
    { \"name\": \"clip_embedding_family\",
      \"signals\": [\"img.clip0\", \"retrieval.knn\"],
      \"rho_max\": 0.83,
      \"note\": \"Both consume the same CLIP-B/32 embedding; merging Platt.\" },
    { \"name\": \"vendor_cnn_ensemble\",
      \"signals\": [\"tp.hive\", \"tp.sightengine\", \"tp.aiornot\"],
      \"rho_max\": 0.66,
      \"note\": \"Three commercial CNN vendors share LAION-scraped train data.\" }
  ],
  \"updated_at\": \"2026-03-01T14:30:00Z\"
}
```

Fusion / distill / cross-modal-bonus all consult this file at load time; absence = empty list (no clustering).

### 4.5 Implementation blueprint — `backend/scripts/eval/correlation_matrix.py`

```python
# file: /app/backend/scripts/eval/correlation_matrix.py
\"\"\"Emit the GoldenEval signal-error correlation matrix and the action-flag list.

Outputs:
  storage/eval/reports/<ts>.correlation.json  (machine-readable; CI gate input)
  storage/eval/reports/<ts>.correlation.md    (diff-friendly review)
\"\"\"
from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

log = logging.getLogger(\"eval.corr\")

SIGNALS = (
    \"img.prithiv\", \"img.frequency\", \"img.clip0\", \"img.meta\", \"img.compression\",
    \"img.ocr_gibberish\", \"img.eye_forensics\", \"img.prnu\",
    \"tp.hive\", \"tp.sightengine\", \"tp.aiornot\",
    \"retrieval.knn\", \"retrieval.patch\", \"reverse.serpapi\",
    \"vlm.gemini\", \"meta.distill_lr\",
)
MIN_PAIRWISE_N = 50


def _errs_for(rows: list[dict], sig: str, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.array([r[\"signals\"].get(sig) for r in rows], dtype=object)
    mask = np.array([v is not None for v in p])
    if mask.sum() < MIN_PAIRWISE_N:
        return np.array([], dtype=int), mask
    p_arr = np.array([float(v) for v in p[mask]])
    errs = ((p_arr >= 0.5).astype(int) ^ y[mask]).astype(int)
    return errs, mask


def main(rows_path: Path, out_json: Path, out_md: Path) -> None:
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    y = np.array([r[\"label\"] for r in rows], dtype=int)
    matrix: dict[str, dict[str, float]] = {s: {} for s in SIGNALS}
    flags: list[dict] = []
    cached: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for s in SIGNALS:
        cached[s] = _errs_for(rows, s, y)
    for a, b in combinations(SIGNALS, 2):
        ea, ma = cached[a]; eb, mb = cached[b]
        if ea.size == 0 or eb.size == 0: continue
        # Pairwise-complete: indices where BOTH are present
        both = ma & mb
        if both.sum() < MIN_PAIRWISE_N: continue
        ya = ((np.array([float(r[\"signals\"][a]) for r in rows if r[\"signals\"].get(a) is not None and r[\"signals\"].get(b) is not None]) >= 0.5).astype(int)
              ^ y[both]).astype(int)
        yb = ((np.array([float(r[\"signals\"][b]) for r in rows if r[\"signals\"].get(a) is not None and r[\"signals\"].get(b) is not None]) >= 0.5).astype(int)
              ^ y[both]).astype(int)
        rho, _ = pearsonr(ya, yb)
        matrix[a][b] = float(rho); matrix[b][a] = float(rho)
        if   rho > 0.70: flags.append({\"pair\": [a, b], \"rho\": float(rho), \"action\": \"merge\"})
        elif rho > 0.50: flags.append({\"pair\": [a, b], \"rho\": float(rho), \"action\": \"downweight_bonus\"})
        elif rho < -0.30: flags.append({\"pair\": [a, b], \"rho\": float(rho), \"action\": \"investigate_polarity\"})
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({\"matrix\": matrix, \"flags\": flags, \"n_rows\": len(rows)}, indent=2))
    out_md.write_text(_render_md(matrix, flags))
    log.info(\"wrote %s and %s — %d flags\", out_json, out_md, len(flags))


def _render_md(matrix: dict, flags: list[dict]) -> str:
    sigs = sorted(matrix)
    lines = [\"# Signal-Error Correlation Matrix\", \"\",
             \"| \" + \" | \".join([\"\"] + sigs) + \" |\",
             \"|\" + \"---|\" * (len(sigs) + 1)]
    for a in sigs:
        row = [a] + [
            f\"{matrix[a].get(b, float('nan')):+.2f}\" if b != a and b in matrix[a] else
            (\"—\" if b == a else \"·\")
            for b in sigs
        ]
        lines.append(\"| \" + \" | \".join(row) + \" |\")
    lines += [\"\", \"## Action flags\", \"\"]
    if not flags:
        lines.append(\"_None — all measured pairs fall in the keep-both band._\")
    else:
        for f in flags:
            lines.append(f\"- `{f['pair'][0]}` ↔ `{f['pair'][1]}` rho = {f['rho']:+.2f} → **{f['action']}**\")
    return \"
\".join(lines) + \"
\"


if __name__ == \"__main__\":
    ap = argparse.ArgumentParser()
    ap.add_argument(\"--rows\", type=Path, required=True)
    ap.add_argument(\"--json\", type=Path, required=True)
    ap.add_argument(\"--md\",   type=Path, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format=\"%(asctime)s %(levelname)s %(name)s %(message)s\")
    main(args.rows, args.json, args.md)
```

### 4.6 CI gate

`scripts/eval/verify_correlation.py` runs after every GoldenEval pass. CI **fails the merge** if any pair has `rho > 0.85` and the merge action is **not** encoded in `backend/fusion/clusters.json`. This converts \"we'll fix that later\" promises into hard blockers.

---

## 5. KPI formulas (locked)

| KPI | Formula | Notes |
|---|---|---|
| **AUROC** | `sklearn.metrics.roc_auc_score(y, p_ai)` with `y=1 ⇔ AI` | |
| **AUPR-AI** | `average_precision_score(y, p_ai)` | |
| **AUPR-Real** | `average_precision_score(1 - y, 1 - p_ai)` | |
| **Brier** | `mean((p_ai - y) ** 2)` | lower is better |
| **ECE-15** | Section 3.4 above, 15 equal-width bins, no isotonic re-binning | |
| **Accuracy@non-abstained** | `correct(¬abstained) / total(¬abstained)` at threshold 0.5 | headline |
| **Deferral rate** | `abstained.mean()` | published as a KPI, not a hidden cost |
| **Conformal coverage** | `(true_label ∈ conformal_set).mean()` | per-stratum after `16_section3_rewrite.md` |
| **FPR-on-REAL@p=0.5** | `((y==0) & (p_ai ≥ 0.5) & ¬abstained).sum() / ((y==0) & ¬abstained).sum()` | production-critical |
| **Per-slice macro-AUROC** | `mean(AUROC_s for s in slices if AUROC_s is not None)` | unweighted |
| **Signal-error correlation** | §4.2 | per pair, audited |
| **Hardest-slice AUROC** | `min(AUROC_s for s in slices if AUROC_s is not None and n_s ≥ 80)` | floor metric |

All KPIs share the same upstream row-JSONL. No metric may introduce its own pipeline re-run — that would break the \"frozen pipeline\" rule (§1.3).

---

## 6. Coverage measurement for the conformal layer

(Refer to `16_section3_rewrite.md` for the Mondrian + ACI redesign that produces these numbers. This section specifies only **how** they are reported, not how they are computed online.)

### 6.1 Per-content-type coverage (Mondrian)

After the rewrite, the conformal quantile is per-content-type (`qhat_c` for c ∈ {selfie, landscape, object, meme, document, artwork}). Coverage is reported **per c**, not only globally.

**Acceptance:** each `cov_c ≥ 0.93` AND global `cov ≥ 0.95` over the GoldenEval slice for content-type c. Per-stratum exceptions for `meme_screenshot` and `artwork_illustration` (noisier strata): `cov_c ≥ 0.91`.

### 6.2 Online ACI miscoverage

For live traffic, ACI maintains a running `α_t` such that `err_t = 𝟙{true ∉ set_t}`. The published guarantee is `mean(err_t over window) → α*` in expectation. The drift monitor (rewritten `08 §6.4` per `16_section3_rewrite.md §4`) reports the delta `α* − mean(err_t)` and fires `coverage_drift_warn` at `|delta| > 0.03` over a 200-window.

### 6.3 What `/api/health.conformal` looks like

```json
\"conformal\": {
    \"method\": \"mondrian_split_aci\",
    \"alpha_target\": 0.05,
    \"alpha_t\": 0.043,
    \"per_content_type\": {
        \"selfie_portrait\":      { \"n\": 200, \"empirical_coverage\": 0.955, \"degraded\": false },
        \"landscape_scene\":      { \"n\": 200, \"empirical_coverage\": 0.940, \"degraded\": false },
        \"meme_screenshot\":      { \"n\":  87, \"empirical_coverage\": 0.920, \"degraded\": false },
        \"document_scan\":        { \"n\":  41, \"empirical_coverage\": null,  \"degraded\": false }
    }
}
```

Each per-c entry: `degraded=true` iff `n ≥ WINDOW (200) AND coverage < target − slack`.

---

## 7. Drift metrics for live (unlabelled) traffic

These run nightly via cron (added in `19_runbook_ops.md §1`):

| Metric | Formula | Alert threshold |
|---|---|---|
| `kl_p_ai_vs_baseline` | `KL(hist(p_ai_last_1k, 20-bin), hist(p_ai_first_1k_after_M3, 20-bin))` | `> 0.05` |
| `ood_trigger_rate_24h` | `(novel_generator_suspected.sum() / total)` over 24 h | `> 0.12` sustained 200 jobs |
| `median_agreement_200` | rolling median of `agreement` over last 200 jobs | `< 0.45` |
| `ece_on_corrections` | ECE-15 on the last 100 user-corrected jobs (label is the correction) | `> 0.10` |
| `aci_miscoverage_window` | §6.2 | `> 0.03` over 200 |

All five are emitted as structured-log events with `event=\"drift_alert\"` and surfaced in `/api/health.alerts[]`. Operator runbook in `19_runbook_ops.md`.

---

## 8. Acceptance gates (the bar to declare a metric \"published\")

A metric must clear **all six gates** before it may be cited in any externally-visible doc, dashboard, README, or marketing copy:

| Gate | Rule | Failure mode |
|---|---|---|
| **G1 — Disjointness** | `20_refdb_eval_disjointness.md` CI script returns 0 | Reject merge |
| **G2 — Reproducibility** | Re-running `bootstrap_metrics.py` on the same row-JSONL produces byte-identical JSON | Bug in code, not data — block |
| **G3 — Correlation** | §4.6 CI gate passes — no `rho > 0.85` outside `clusters.json` | Update `clusters.json` or fix detector |
| **G4 — Pipeline config snapshot** | `eval/pipeline_configs/<ts>.json` written and committed alongside the metric | The number is unsourced — block |
| **G5 — Slice coverage** | No more than 2 slices have `AUROC == null` (mono-label slices); each remaining slice has `n ≥ 80` | Expand GoldenEval before publishing |
| **G6 — Per-slice floor** | `hardest_slice_AUROC ≥ profile gate` from `17 §5.2` | The metric is carried by easy buckets — block |

A number that fails any gate **may not be cited in 14 §3, 16 §1.7 / §5, or 17 §5**. The Masterplan §1.5 KPI table must be updated to reflect only actually-measured values; speculative numbers must carry an explicit `(target, unmeasured)` annotation.

---

## 9. Folder layout

```
backend/
├── scripts/
│   └── eval/
│       ├── bootstrap_metrics.py
│       ├── correlation_matrix.py
│       ├── verify_correlation.py        # CI gate G3
│       ├── verify_gates.py              # CI gate orchestrator G1..G6
│       └── (existing) download_goldeneval.py, run_goldeneval.py
└── storage/
    └── eval/
        ├── rows/                          # per-run JSONL outputs (gitignored)
        │   └── 20260301_1430_rows.jsonl
        ├── reports/                       # markdown + JSON (committed)
        │   ├── 20260301_1430.md
        │   ├── 20260301_1430.bootstrap.json
        │   ├── 20260301_1430.correlation.json
        │   └── 20260301_1430.correlation.md
        └── pipeline_configs/              # snapshot per run (committed)
            └── 20260301_1430.json
```

`.gitignore` adds:
```
/app/backend/storage/eval/rows/
!/app/backend/storage/eval/reports/
!/app/backend/storage/eval/pipeline_configs/
```

---

## 10. Worked example — what a v1.5-quality report looks like

Excerpt from a passing `storage/eval/reports/20260301_1430.md`:

```
# GoldenEval Report — 2026-03-01 14:30 UTC
Profile: cloud_lite  N: 1750  Pipeline snapshot: 20260301_1430.json

## Overall (bootstrap B=1000, seed=20260201)
| KPI                  | Point  | 95 % CI         | Half-width |
|----------------------|--------|-----------------|------------|
| AUROC                | 0.913  | 0.898 – 0.927   | 0.014      |
| AUPR-AI              | 0.927  | 0.913 – 0.941   | 0.014      |
| AUPR-Real            | 0.901  | 0.882 – 0.918   | 0.018      |
| Brier                | 0.084  | 0.077 – 0.092   | 0.008      |
| ECE-15               | 0.052  | 0.043 – 0.061   | 0.009      |
| Coverage             | 0.951  | 0.940 – 0.961   | 0.010      |
| Accuracy@¬abstain    | 0.962  | 0.950 – 0.972   | 0.011      |
| Deferral rate        | 0.193  | 0.178 – 0.208   | 0.015      |
| FPR-on-REAL@.5       | 0.014  | 0.008 – 0.022   | 0.007      |
| Hardest-slice AUROC  | 0.787  | (slug: meme_screenshot, n=92) |

## Correlation flags (G3)
- img.clip0 ↔ retrieval.knn          rho = +0.81 → merge   (resolved in clusters.json)
- tp.hive ↔ tp.sightengine           rho = +0.62 → downweight_bonus
- tp.hive ↔ tp.aiornot               rho = +0.58 → downweight_bonus
- tp.sightengine ↔ tp.aiornot        rho = +0.61 → downweight_bonus

## Gates
- G1 Disjointness         : PASS
- G2 Reproducibility      : PASS (sha256 of bootstrap.json matches prior run)
- G3 Correlation          : PASS (all flagged pairs encoded in clusters.json)
- G4 Snapshot present     : PASS
- G5 Slice coverage       : PASS (16/16 slices ≥ 80, 0 mono-label)
- G6 Hardest-slice floor  : PASS (0.787 ≥ cloud_lite gate 0.75)

→ PUBLISHABLE.
```

If those numbers and gates hold across **two independent rebuilds** of refDB, they may be cited in 14 §3 and the README.

---

## 11. Adversarial-bench scoring

The 10 adversarial fixtures from `17 §6` are tagged `adversarial_tag != null` in the row-JSONL. The bootstrap runner produces a separate **\"adversarial-only\" block** in the report:

```
## Adversarial-only (10 fixtures, no bootstrap — assertions binary)
| Fixture                          | Expected      | Got          | Pass |
|----------------------------------|---------------|--------------|------|
| recompressed_sdxl_q50.jpg        | AI-GENERATED  | AI-GENERATED | ✓    |
| screenshot_of_mjv6.png           | not REAL      | AI-GENERATED | ✓    |
| composite_ai_bg_real_face.png    | not REAL      | INCONCLUSIVE | ✓    |
| pre_ai_reuters_2018.jpg          | REAL          | REAL         | ✓    |
| civitai_realistic_portrait.png   | AI-GENERATED  | AI-GENERATED | ✓    |
| sd_watermark_stripped.png        | AI-GENERATED  | AI-GENERATED | ✓    |
| counter_prompt_disagree.png      | INCONCLUSIVE  | INCONCLUSIVE | ✓    |
| ood_novel_generator.png          | INCONCLUSIVE  | INCONCLUSIVE | ✓    |
| vsco_filtered_dslr.jpg           | not AI        | REAL         | ✓    |
| c2pa_signed_camera.jpg           | REAL          | REAL         | ✓    |
→ 10/10 PASS
```

All 10 must pass. A single adversarial failure blocks the M3 final-gate regardless of bootstrap numbers.

---

## 12. AGENTS.md mapping

| Standard | Where honoured |
|---|---|
| §5 TDD | `bootstrap_metrics.py`, `correlation_matrix.py`, `verify_gates.py` all have unit tests on synthetic row-JSONL fixtures |
| §7 Observability | Drift metrics (§7); structured logs; per-run pipeline snapshots committed |
| §9 Documentation | Reports auto-generated, diff-friendly, dated; report markdown is the changelog |
| §14 AI/ML — versioning + A/B | `pipeline_config_{ts}.json` is the canonical A/B unit |
| §15 Type safety | NumPy boundaries; mypy strict on `scripts/eval/`; Pydantic schema for row-JSONL |
| Naming | `bootstrap_metrics.py`, `correlation_matrix.py`, `verify_gates.py` — short, professional |
| §11 Idempotency | Re-running the bootstrap on the same JSONL is byte-identical (rule §1.2 enforces it) |

---

## 13. Exit criteria

```bash
# 1) Run the bench (single source of truth p_ai per sample)
python -m backend.scripts.run_goldeneval \
  --eval-dir storage/eval/goldeneval \
  --report-out storage/eval/reports/$(date +%Y%m%d_%H%M).md \
  --rows-out  storage/eval/rows/$(date +%Y%m%d_%H%M)_rows.jsonl \
  --snapshot  storage/eval/pipeline_configs/$(date +%Y%m%d_%H%M).json

# 2) Bootstrap every KPI
python -m backend.scripts.eval.bootstrap_metrics \
  --rows storage/eval/rows/<ts>_rows.jsonl \
  --out  storage/eval/reports/<ts>.bootstrap.json

# 3) Build the correlation matrix
python -m backend.scripts.eval.correlation_matrix \
  --rows storage/eval/rows/<ts>_rows.jsonl \
  --json storage/eval/reports/<ts>.correlation.json \
  --md   storage/eval/reports/<ts>.correlation.md

# 4) Verify all six gates (G1..G6)
python -m backend.scripts.eval.verify_gates --ts <ts>
# Expected: stdout \"ALL GATES PASS\"; exit 0

# 5) Reproducibility check (G2)
python -m backend.scripts.eval.bootstrap_metrics \
  --rows storage/eval/rows/<ts>_rows.jsonl \
  --out  /tmp/repro.json
diff <(jq -S . storage/eval/reports/<ts>.bootstrap.json) <(jq -S . /tmp/repro.json)
# Expected: empty diff
```

When all five commands succeed, every KPI in 14 §3, 16 §5, and 17 §5 may be **rewritten with measured values + 95 % CI**. Until then they are aspirations.

---

End of `15_evaluation_protocol.md`. Source of truth for the measurement protocol that backs every accuracy claim cited in 14, 16, 17. Pair with `20_refdb_eval_disjointness.md` (the disjointness contract) and `16_section3_rewrite.md` (the Mondrian+ACI conformal redesign).
"