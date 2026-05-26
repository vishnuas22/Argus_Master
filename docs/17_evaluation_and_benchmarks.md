"# 17 — Evaluation & Benchmarks (GoldenEval)

> Goal: turn the headline KPI \"≥95 % precision on non-abstained uploads\" from an
> aspiration into a measurable, reproducible number. Without this doc, every
> accuracy figure in the rest of the repo is unverifiable.
>
> Status: **P0 — must complete before M3 close.** GoldenEval pass is the final
> gate on \"First-Finish.\"
>
> Source-of-truth datasets are public; we ship **download scripts only**, never
> the data itself (license & repo-size hygiene).

---

## 1. Philosophy — three non-negotiable rules

1. **Held-out forever.** GoldenEval samples are *never* indexed into the
   reference DB, *never* used to fit Platt scaling, *never* used to fit
   conformal quantiles, *never* used to tune content-type thresholds. They
   exist only to score the assembled system end-to-end.
2. **Public + diverse.** Pulled from 6+ independent sources covering 8+
   generator families and 6 content types. No single generator > 15 % of the
   AI slice. No single photographer/agency > 10 % of the REAL slice.
3. **Adversarial bench included.** ~30 % of the eval set is intentionally
   \"hard\" (re-encoded, screenshotted, watermark-stripped, composites,
   pre-AI-era news photos). The headline KPI must hold on the *full* set,
   not just the easy slice.

---

## 2. GoldenEval composition (image first-finish)

Target total: **1700 samples** (≈ 850 REAL + 850 AI). Sized so a full eval
runs in < 30 min on `cloud_lite`, < 8 min on `mac_full`.

| Bucket | Count | Source | License | Notes |
|---|---|---|---|---|
| Real — natural photos | 300 | Flickr Commons CC0 + Wikimedia Commons PD | CC0 / PD | Mixed content types |
| Real — news / journalism (pre-2022) | 150 | Wikipedia \"in the news\" archive ≤ 2021-12 | Editorial | Pre-AI-era anchor |
| Real — selfies / portraits | 100 | FFHQ test split (NVIDIA, research only) | Research | Face slice |
| Real — products / objects | 100 | Open Images v7 (Google, CC BY) | CC BY | Object slice |
| Real — documents / scans | 100 | RVL-CDIP test split | Research | Document slice |
| Real — Instagram-filtered | 100 | Self-curated from CC0 + manual VSCO/Lightroom passes | CC0 | Heavily-processed real |
| **REAL total** | **850** | | | |
| AI — SDXL | 120 | DiffusionDB held-out + manual gen | CC0 metadata | |
| AI — SD 1.5 | 100 | GenImage SD1.5 split | Research | |
| AI — Flux.1 [schnell/dev] | 100 | Civitai-flagged + manual gen | Mixed | |
| AI — Midjourney v6 | 100 | Civitai-flagged subset | Mixed | |
| AI — DALL·E 3 | 80 | LAION-DALL·E-3 subset | CC0 metadata | |
| AI — Ideogram + Adobe Firefly | 80 | Public gallery scrape | Mixed | Includes embedded text |
| AI — Stable Cascade + PixArt-Σ | 70 | Civitai-flagged | Mixed | Off-the-beaten-path generators |
| AI — face-only (DeepFaceLab / e4e / StyleGAN3) | 100 | FF++ / WildDeepfake stills | Research | Selfie face slice |
| AI — watermark-stripped SD | 50 | Manually re-encoded from SDXL bucket | derived | Tier-0 evasion |
| AI — screenshot-of-AI | 50 | Manually screenshotted + re-saved JPEG q=50 | derived | Tier-1 evasion |
| AI — composite (AI background + real face) | 50 | Manually composited | derived | Patch retrieval target |
| **AI total** | **900** | | | |

> Slice sizes are *targets*; minimum acceptable per slice = 80 % of target.
> If a source is unavailable, log it in the build report and continue.

### 2.1 Slice tags

Every sample carries:

```json
{ \"id\": \"gen_dalle3_00042\",
  \"label\": \"ai\",
  \"generator\": \"dalle3\",
  \"content_type\": \"object_product\",
  \"adversarial_tag\": null,            // or \"screenshot\" | \"watermark_stripped\" | \"composite\" | \"pre_ai_era\"
  \"source\": \"laion-dalle3-subset\",
  \"license\": \"cc0\",
  \"sha256\": \"...\",
  \"added_at\": \"2026-02-01\" }
```

Slice tags drive per-slice AUROC reporting (§5.2). The headline KPI is the
*macro-average* across slices, weighted equally — protecting against a
system that scores 99 % on the easy slice and 60 % on adversarial.

---

## 3. Download scripts (data never shipped)

```
backend/scripts/eval/
├── download_goldeneval.py        # CLI orchestrator
├── sources/
│   ├── flickr_commons.py         # CC0 photo crawl
│   ├── wikimedia.py              # Commons API
│   ├── wikipedia_news.py         # 2018-2021 \"in the news\" pics
│   ├── ffhq_subset.py            # NVIDIA FFHQ test split
│   ├── open_images.py            # Google CC BY split
│   ├── rvl_cdip.py               # Document split
│   ├── diffusiondb_holdout.py    # USC HF dataset, held-out hash list
│   ├── genimage_subset.py        # 8-generator HF dataset
│   ├── laion_dalle3.py           # CC0-metadata subset
│   ├── civitai_flagged.py        # Civitai \"real-style\" gallery filter
│   └── manual_adversarial.py     # Composites/screenshots/stripped — pre-rendered, hosted in GH release
└── manifest_v1.json              # Frozen SHA256 list of every sample
```

```python
# file: /app/backend/scripts/eval/download_goldeneval.py
\"\"\"Build the GoldenEval set on a fresh machine.

Usage:
    python -m backend.scripts.eval.download_goldeneval --out storage/eval/goldeneval

Idempotent: skips files already present whose SHA matches manifest_v1.json.
Reports missing samples; never silently truncates.\"\"\"
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path

from backend.scripts.eval.sources import (
    civitai_flagged, diffusiondb_holdout, ffhq_subset, flickr_commons,
    genimage_subset, laion_dalle3, manual_adversarial, open_images,
    rvl_cdip, wikimedia, wikipedia_news,
)

log = logging.getLogger(\"eval.download\")

MANIFEST = Path(__file__).with_name(\"manifest_v1.json\")
SOURCES = [
    (flickr_commons.fetch,     \"real\", \"flickr_commons\",   300, \"cc0\"),
    (wikipedia_news.fetch,     \"real\", \"wiki_news_pre22\",  150, \"editorial\"),
    (ffhq_subset.fetch,        \"real\", \"ffhq\",             100, \"research\"),
    (open_images.fetch,        \"real\", \"open_images_v7\",   100, \"cc_by\"),
    (rvl_cdip.fetch,           \"real\", \"rvl_cdip\",         100, \"research\"),
    (wikimedia.fetch_filtered, \"real\", \"vsco_filtered\",    100, \"cc0\"),
    (diffusiondb_holdout.fetch,\"ai\",   \"sdxl\",             120, \"cc0_meta\"),
    (genimage_subset.fetch,    \"ai\",   \"sd15\",             100, \"research\"),
    (civitai_flagged.fetch,    \"ai\",   \"flux\",             100, \"mixed\"),
    (civitai_flagged.fetch,    \"ai\",   \"midjourney_v6\",    100, \"mixed\"),
    (laion_dalle3.fetch,       \"ai\",   \"dalle3\",            80, \"cc0_meta\"),
    (civitai_flagged.fetch,    \"ai\",   \"ideogram_firefly\",  80, \"mixed\"),
    (civitai_flagged.fetch,    \"ai\",   \"cascade_pixart\",    70, \"mixed\"),
    (ffhq_subset.fetch,        \"ai\",   \"face_deepfake\",    100, \"research\"),
    (manual_adversarial.fetch, \"ai\",   \"wm_stripped\",       50, \"derived\"),
    (manual_adversarial.fetch, \"ai\",   \"screenshot\",        50, \"derived\"),
    (manual_adversarial.fetch, \"ai\",   \"composite\",         50, \"derived\"),
]


async def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    report: dict[str, dict] = {}

    for fn, label, slug, cap, lic in SOURCES:
        slug_dir = out_dir / label / slug
        slug_dir.mkdir(parents=True, exist_ok=True)
        try:
            n = await fn(slug_dir, cap, manifest=manifest)
            report[slug] = {\"label\": label, \"got\": n, \"target\": cap, \"license\": lic}
            log.info(\"%s: %d/%d\", slug, n, cap)
        except Exception as exc:                 # noqa: BLE001 — boundary
            log.error(\"%s failed: %s\", slug, exc)
            report[slug] = {\"label\": label, \"got\": 0, \"target\": cap, \"error\": str(exc)}

    # Persist run report
    (out_dir / \"build_report.json\").write_text(json.dumps(report, indent=2))
    missing = [s for s, r in report.items() if r.get(\"got\", 0) < r[\"target\"] * 0.8]
    if missing:
        log.warning(\"Buckets below 80%% of target: %s\", missing)


if __name__ == \"__main__\":
    ap = argparse.ArgumentParser()
    ap.add_argument(\"--out\", type=Path, default=Path(\"storage/eval/goldeneval\"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format=\"%(asctime)s %(levelname)s %(name)s %(message)s\")
    asyncio.run(main(args.out))
```

> Each `sources/*.py` is a thin async iterator that yields `(bytes, meta)` and
> stops at the cap. Implementation pattern mirrors `retrieval/sources/*` in
> doc 12 §3.3. SHA256 of every downloaded file is checked against
> `manifest_v1.json`; mismatch → reject + log.

### 3.1 First-time bootstrap

On a fresh machine, run **once**:

```bash
python -m backend.scripts.eval.download_goldeneval \
    --out /app/backend/storage/eval/goldeneval
```

Approx run time: 25–45 min (network bound). Total disk: ~5.2 GB.

### 3.2 Reproducibility

`manifest_v1.json` pins SHA256 of every expected file. CI fails if a sample
is missing from the manifest *and* not flagged in `manual_adversarial`.

When the bench needs to evolve, bump `manifest_v2.json` — never edit v1.

---

## 4. Eval runner — `scripts/run_goldeneval.py`

```python
# file: /app/backend/scripts/run_goldeneval.py
\"\"\"End-to-end GoldenEval runner.

Loops over every sample, runs the FULL pipeline as a real /analyze call
(through the in-process runner, not the HTTP API — saves ~80 ms per sample),
records: p_ai, verdict, abstained, durations, content_type, signals_used.
Computes AUROC, ECE, precision-on-non-abstained, abstention rate per slice
and overall. Writes Markdown + JSON report under storage/eval/reports/.

Usage:
    python -m backend.scripts.run_goldeneval \
        --eval-dir storage/eval/goldeneval \
        --report-out storage/eval/reports/$(date +%Y%m%d_%H%M).md
\"\"\"
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

from backend.services.runner import run_pipeline           # async, in-process
from backend.services.device import detect_profile

log = logging.getLogger(\"eval.run\")


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    \"\"\"Expected Calibration Error.\"\"\"
    bins = np.linspace(0, 1, n_bins + 1)
    inds = np.digitize(p, bins) - 1
    e, n = 0.0, len(p)
    for b in range(n_bins):
        mask = inds == b
        if not mask.any(): continue
        e += abs(p[mask].mean() - y[mask].mean()) * mask.sum() / n
    return float(e)


async def main(eval_dir: Path, report_out: Path) -> None:
    profile = detect_profile()
    samples: list[dict] = []
    for label_dir in eval_dir.glob(\"*\"):
        if label_dir.name not in {\"real\", \"ai\"}: continue
        y = 1 if label_dir.name == \"ai\" else 0
        for slug_dir in label_dir.glob(\"*\"):
            for img in slug_dir.glob(\"*\"):
                if img.is_file() and img.suffix.lower() in {\".jpg\",\".jpeg\",\".png\",\".webp\"}:
                    samples.append({\"path\": img, \"y\": y, \"slug\": slug_dir.name})

    log.info(\"Loaded %d samples; profile=%s\", len(samples), profile)

    rows: list[dict] = []
    started = time.time()
    for i, s in enumerate(samples):
        t0 = time.time()
        try:
            result = await run_pipeline(s[\"path\"].read_bytes(), filename=s[\"path\"].name)
        except Exception as exc:                # noqa: BLE001 — bench tolerates
            log.exception(\"sample %s failed: %s\", s[\"path\"].name, exc)
            continue
        rows.append({
            \"y\": s[\"y\"],
            \"slug\": s[\"slug\"],
            \"p_ai\": float(result[\"p_ai_generated\"]),
            \"verdict\": result[\"verdict\"],
            \"abstained\": bool(result[\"abstained\"]),
            \"content_type\": result[\"content_type\"],
            \"dur_ms\": int((time.time() - t0) * 1000),
            \"provenance_hit\": bool(result[\"provenance\"][\"hit\"]),
        })
        if (i + 1) % 50 == 0:
            log.info(\"%d/%d  median=%dms\", i + 1, len(samples),
                     int(np.median([r[\"dur_ms\"] for r in rows])))

    # ─── metrics ────────────────────────────────────────────────────────
    y = np.array([r[\"y\"] for r in rows])
    p = np.array([r[\"p_ai\"] for r in rows])
    ab = np.array([r[\"abstained\"] for r in rows], dtype=bool)

    overall = {
        \"n_total\": len(rows),
        \"auroc\": float(roc_auc_score(y, p)),
        \"ece\": ece(p, y),
        \"abstention_rate\": float(ab.mean()),
        \"precision_non_abstained_ai\":
            float(((y == 1) & (p >= 0.5) & ~ab).sum() / max(1, ((p >= 0.5) & ~ab).sum())),
        \"precision_non_abstained_real\":
            float(((y == 0) & (p < 0.5) & ~ab).sum() / max(1, ((p < 0.5) & ~ab).sum())),
        \"median_latency_ms\": int(np.median([r[\"dur_ms\"] for r in rows])),
        \"p95_latency_ms\":    int(np.percentile([r[\"dur_ms\"] for r in rows], 95)),
        \"wall_clock_s\":      int(time.time() - started),
        \"profile\":           profile,
    }

    by_slug = defaultdict(list)
    for r in rows: by_slug[r[\"slug\"]].append(r)
    slice_metrics = {}
    for slug, srows in by_slug.items():
        ys = np.array([r[\"y\"] for r in srows])
        ps = np.array([r[\"p_ai\"] for r in srows])
        abs_s = np.array([r[\"abstained\"] for r in srows], dtype=bool)
        try:
            auc_s = float(roc_auc_score(ys, ps)) if len(set(ys)) > 1 else None
        except Exception: auc_s = None
        slice_metrics[slug] = {
            \"n\": len(srows),
            \"auroc\": auc_s,
            \"abstention_rate\": float(abs_s.mean()),
            \"median_ms\": int(np.median([r[\"dur_ms\"] for r in srows])),
        }

    # ─── persist ────────────────────────────────────────────────────────
    report_out.parent.mkdir(parents=True, exist_ok=True)
    json_out = report_out.with_suffix(\".json\")
    json_out.write_text(json.dumps({\"overall\": overall, \"by_slug\": slice_metrics, \"rows\": rows}, indent=2))
    _write_markdown(report_out, overall, slice_metrics)
    log.info(\"Wrote %s and %s\", report_out, json_out)
    _print_gate_check(overall, profile)


def _write_markdown(path: Path, overall: dict, by_slug: dict) -> None:
    lines = [
        f\"# GoldenEval Report — {time.strftime('%Y-%m-%d %H:%M %Z')}\",
        f\"
**Profile:** `{overall['profile']}`  **N:** {overall['n_total']}
\",
        \"## Overall\",
        f\"- AUROC: **{overall['auroc']:.4f}**\",
        f\"- ECE: **{overall['ece']:.4f}**\",
        f\"- Abstention rate: **{overall['abstention_rate']:.2%}**\",
        f\"- Precision @ non-abstained (AI):   **{overall['precision_non_abstained_ai']:.2%}**\",
        f\"- Precision @ non-abstained (REAL): **{overall['precision_non_abstained_real']:.2%}**\",
        f\"- Median latency: **{overall['median_latency_ms']} ms**\",
        f\"- p95 latency:    **{overall['p95_latency_ms']} ms**\",
        \"
## Per-slice\",
        \"| Slice | N | AUROC | Abstain | Median ms |\",
        \"|---|---|---|---|---|\",
    ]
    for slug, m in sorted(by_slug.items()):
        auc = f\"{m['auroc']:.3f}\" if m[\"auroc\"] is not None else \"n/a\"
        lines.append(f\"| {slug} | {m['n']} | {auc} | {m['abstention_rate']:.1%} | {m['median_ms']} |\")
    path.write_text(\"
\".join(lines))


def _print_gate_check(o: dict, profile: str) -> None:
    gates = GATES[profile]
    ok = (
        o[\"auroc\"] >= gates[\"auroc_min\"]
        and o[\"ece\"] <= gates[\"ece_max\"]
        and o[\"abstention_rate\"] <= gates[\"abstain_max\"]
        and o[\"precision_non_abstained_ai\"] >= gates[\"precision_min\"]
        and o[\"precision_non_abstained_real\"] >= gates[\"precision_min\"]
        and o[\"median_latency_ms\"] <= gates[\"latency_med_ms\"]
    )
    log.info(\"Gate check (%s): %s\", profile, \"PASS\" if ok else \"FAIL\")


GATES = {
    \"cloud_lite\": {\"auroc_min\": 0.85, \"ece_max\": 0.08, \"abstain_max\": 0.28,
                   \"precision_min\": 0.94, \"latency_med_ms\": 15000},
    \"mac_full\":   {\"auroc_min\": 0.92, \"ece_max\": 0.06, \"abstain_max\": 0.18,
                   \"precision_min\": 0.96, \"latency_med_ms\": 8000},
    \"cuda_full\":  {\"auroc_min\": 0.91, \"ece_max\": 0.06, \"abstain_max\": 0.18,
                   \"precision_min\": 0.96, \"latency_med_ms\": 8000},
}


if __name__ == \"__main__\":
    ap = argparse.ArgumentParser()
    ap.add_argument(\"--eval-dir\",  type=Path, default=Path(\"storage/eval/goldeneval\"))
    ap.add_argument(\"--report-out\", type=Path, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format=\"%(asctime)s %(levelname)s %(name)s %(message)s\")
    asyncio.run(main(args.eval_dir, args.report_out))
```

---

## 5. Report format

### 5.1 Markdown (committed to `storage/eval/reports/`)

Generated by `_write_markdown` above. Always shows overall + per-slice
table — diff-friendly for PR review.

### 5.2 Per-slice gates (the actual quality bar)

Macro-average across slices is what gates \"First-Finish.\" Hard threshold:

| Profile | Macro AUROC | Hardest-slice AUROC ≥ | Macro abstain ≤ | Median p95 latency ≤ |
|---|---|---|---|---|
| `cloud_lite` | 0.85 | 0.75 | 28 % | 15 s / 25 s |
| `mac_full`   | 0.92 | 0.84 | 18 % |  8 s / 14 s |
| `cuda_full`  | 0.91 | 0.83 | 18 % |  8 s / 14 s |

\"Hardest-slice\" caveat: prevents passing by carrying easy buckets.

### 5.3 ECE gate

Macro ECE ≤ 0.08 on `cloud_lite`, ≤ 0.06 on `mac_full`/`cuda_full`. ECE
above this → recalibration required before merge.

---

## 6. Adversarial test bench (must-pass)

Specifically pulled from doc 14 §4 scenarios, asserted as unit-level tests
within `tests/integration/test_adversarial.py`:

| Scenario | Expected behaviour | Asserted via |
|---|---|---|
| JPEG-recompressed AI image (q=50) | Verdict = AI; abstained=false | `assert v==\"AI\" and not ab` |
| Screenshot of AI image | Verdict = AI OR INCONCLUSIVE; never REAL with high confidence | `assert v != \"REAL\" or p_ai > 0.4` |
| Composite (AI bg + real face) | INCONCLUSIVE OR AI; patch-retrieval signal > 0.6 | `assert sigs[\"img.retrieval_patch\"] > 0.6` |
| Pre-AI-era news photo (≤ 2021) | Verdict = REAL; reverse-search signal < 0.2 | `assert v==\"REAL\"` |
| Civitai-hosted photoreal | Verdict = AI; reverse-search > 0.85 | `assert v==\"AI\"` |
| Watermark-stripped SDXL | Verdict = AI; freq + compression carry it | `assert v==\"AI\" and (sigs[\"img.freq\"]+sigs[\"img.compression\"])/2 > 0.6` |
| Counter-prompt VLM disagreement | VLM 2nd-opinion bracket → INCONCLUSIVE on contradictory pair | `assert ab` |
| OOD novel generator (held-out) | INCONCLUSIVE via OOD override | `assert v==\"INCONCLUSIVE\" and ood_triggered` |
| Heavily VSCO-filtered real | Verdict = REAL OR INCONCLUSIVE; never AI with high confidence | `assert v != \"AI\" or p_ai < 0.65` |
| Multi-AI mosaic (collage of AI tiles) | Verdict = AI; patch-retrieval > 0.8 | `assert v==\"AI\"` |

These 10 scenarios are part of the GoldenEval set and tagged with
`adversarial_tag != null`. The runner script reports a separate
\"adversarial-only\" AUROC + abstain. Both must clear the gates.

---

## 7. Continuous evaluation

| Trigger | What runs | Where reported |
|---|---|---|
| RefDB rebuild (any reason) | Full GoldenEval | `storage/eval/reports/refdb_rebuild_<ts>.md` |
| Calibration re-fit | Full GoldenEval | `storage/eval/reports/recalib_<ts>.md` |
| Detector added or weight bumped | Full GoldenEval | PR comment via CI hook |
| Weekly cron (optional) | Full GoldenEval | `storage/eval/reports/weekly/<iso_week>.md` |
| `/api/health` deep-check | 50-sample mini-eval | In-memory; surfaced in `/api/health.eval_mini` |

The 50-sample mini-eval (stratified) takes ~90 s on `cloud_lite` and gives a
canary for drift between full runs.

---

## 8. Drift detection on real traffic (no labels needed)

Post-launch, even without ground truth on every upload, drift signals:

| Metric | What it tells us | Alert threshold |
|---|---|---|
| Distribution of `p_ai` shifts | New generator family appearing | KL-div vs. last 1k baseline > 0.05 |
| OOD-trigger rate climbs | RefDB stale | > 12 % sustained over 200 jobs |
| `agreement` median drops | Signals diverging — generator targeting one detector | < 0.45 over 200 jobs |
| ECE-on-corrections rises | Calibration drift | > 0.10 |

Emit a structured-log line `event=\"drift_alert\"` per trigger. Surfaced in
`/api/health.alerts[]`.

---

## 9. Folder layout

```
backend/
├── scripts/
│   ├── eval/
│   │   ├── download_goldeneval.py
│   │   ├── manifest_v1.json                # frozen SHA256 list
│   │   └── sources/                        # per-dataset fetchers (10 files)
│   └── run_goldeneval.py
└── storage/
    └── eval/
        ├── goldeneval/                     # downloaded data (gitignored)
        │   ├── real/                       # 850 across 6 slugs
        │   └── ai/                         # 900 across 10 slugs
        └── reports/                        # markdown + JSON, committed
            ├── 20260215_1430.md
            ├── 20260215_1430.json
            └── weekly/
```

`.gitignore` adds:

```
/app/backend/storage/eval/goldeneval/
!/app/backend/storage/eval/reports/
```

---

## 10. Failure modes & fallbacks

| Failure | Behaviour |
|---|---|
| HF dataset access blocked (rate limit / region) | Skip slug, log; build_report records `error` field; eval still runs on whatever was fetched, marked `partial=true` |
| One slice < 80 % of target | Eval runs but report banner says `INSUFFICIENT_DATA`; macro metrics computed with available |
| `run_pipeline` raises on a sample | Sample excluded; logged; aggregate `n_failed_samples` recorded |
| Network down at runtime | Eval still runs against on-disk data; SerpAPI signal absent for every sample |
| Quotas exhausted mid-eval | Continue without that signal; report flags `signal_dropped` count |

The eval is *always* runnable, even degraded. Hidden assumption to test:
the system works as a deepfake detector with any combination of signals
missing.

---

## 11. Definition of \"GoldenEval pass\" (M3 exit gate)

All of the following must hold simultaneously:

- [ ] Macro AUROC ≥ gate for current profile (§5.2)
- [ ] Hardest-slice AUROC ≥ gate
- [ ] Macro ECE ≤ gate (§5.3)
- [ ] Abstention rate ≤ gate
- [ ] Precision-on-non-abstained ≥ gate, separately for AI and REAL
- [ ] All 10 adversarial scenarios pass their assertion
- [ ] No more than 2 of 16 slices have AUROC `n/a` (mono-label)
- [ ] Report committed to `storage/eval/reports/`
- [ ] PRD.md \"Implementation Notes\" updated with the report hash

A failing gate is not a bug to \"work around\" — it is a signal to revisit
calibration, fusion weights, or signal portfolio composition.

---

## 12. AGENTS.md mapping for this file

| AGENTS.md principle | Where addressed |
|---|---|
| §5 Test-driven development | Adversarial bench tests committed under `tests/integration/` |
| §6 Compliance / GDPR | GoldenEval data is gitignored; only metrics committed |
| §7 Observability | Reports machine-parsable JSON; drift alerts via structured logs |
| §9 Documentation | Reports auto-generated, diff-friendly, dated |
| §14 AI/ML — model versioning + A/B | Slice-level metrics enable detector A/B; SHA-pinned manifest |
| §15 Type safety | Numpy + Pydantic boundaries; mypy strict on `scripts/eval/` |

---

## 13. Section exit criteria

```bash
# 1) Build the GoldenEval set
python -m backend.scripts.eval.download_goldeneval --out storage/eval/goldeneval
ls storage/eval/goldeneval/real | wc -l   # expect 6 buckets
ls storage/eval/goldeneval/ai | wc -l     # expect 10 buckets

# 2) Run the eval
python -m backend.scripts.run_goldeneval \
    --eval-dir storage/eval/goldeneval \
    --report-out storage/eval/reports/$(date +%Y%m%d_%H%M).md

# 3) Confirm pass-gate
tail -1 /var/log/.../eval.log | grep \"Gate check.*PASS\"
```

When that three-step sequence prints `PASS` on `cloud_lite`, M3 is gateable.
"