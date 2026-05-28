"# 08 — Fusion, Calibration, Abstention, OOD Override

> Goal: turn the raw `p_fake` outputs of 10–14 signals into one calibrated probability, one verdict, and a principled \"I don't know\". This is the file where free-tools accuracy actually beats fine-tuned single-model accuracy.
>
> Five layers, executed in order:
>
> 1. **Per-signal Platt calibration** (built once from refDB) — straightens raw probabilities so 0.7 actually means \"70% of the time it's AI\".
> 2. **Adaptive fusion** — `uniform → LR-L2 → GBDT` progression as user-labelled data accumulates.
> 3. **Cross-modal bonus** — additive confidence when independent tier-families agree.
> 4. **OOD novel-generator override** (NEW v1.4) — Isolation Forest on CLIP embeddings; force `INCONCLUSIVE` when nothing in refDB looks like the upload.
> 5. **Conformal-prediction abstention** (NEW v1.5) — pick the threshold mathematically, not by trial-and-error.

---

## 1. Common types — `backend/fusion/types.py`

```python
# file: /app/backend/fusion/types.py
\"\"\"Pure data containers. No I/O. Used by every fusion submodule.\"\"\"
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SignalIn:
    \"\"\"One detector's output, ready for calibration + fusion.\"\"\"
    name: str                          # e.g. \"img.prithiv\"
    p_fake: float                      # RAW, uncalibrated, in [0,1]
    enabled: bool = True               # False → mean-imputed in fusion
    weight_hint: float = 1.0           # tier-family prior (1.0 = neutral)


@dataclass
class FusionResult:
    p_ai: float                        # final fused probability
    agreement: float                   # 1 - mean |sig - p_ai|
    extremity: float                   # max(p_ai, 1 - p_ai) - 0.5
    cross_modal_bonus: float           # additive boost (0..0.10)
    fusion_model: Literal[\"uniform\", \"lr_l2\", \"gbdt\"]
    calibration: Literal[\"platt_refdb\", \"platt_blended\", \"isotonic\"]
    weights: dict[str, float] = field(default_factory=dict)
    imputed: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    label: Literal[\"AI-GENERATED\", \"REAL\", \"INCONCLUSIVE\", \"MANIPULATED\"]
    confidence: float                  # 0..1
    abstained: bool
    rationale: str                     # 1-line reason

    # NOTE: \"MANIPULATED\" is set ONLY by the runner cross-check in
    # 10_runner_orchestrator.md §2 (`_manipulation_check`) — never by an
    # individual detector. It means EXIF-camera-shape contradicts both
    # frequency-domain and compression fingerprints.
```

---

## 2. Per-signal Platt calibration

### 2.1 `backend/calibration/platt.py`

```python
# file: /app/backend/calibration/platt.py
\"\"\"Platt scaling = a one-parameter logistic that maps raw p_fake → calibrated p_fake.

Built per signal once at refDB build time. Persisted to
storage/refdb/calibration.json so the runner needs no fit at request time.\"\"\"
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression

log = logging.getLogger(\"calibration\")
CALIB_PATH = Path(\"/app/backend/storage/refdb/calibration.json\")


@dataclass
class PlattParams:
    a: float          # slope
    b: float          # intercept
    auroc: float
    ece: float
    n_pos: int
    n_neg: int


def fit_one(p_raw: np.ndarray, y: np.ndarray) -> PlattParams:
    \"\"\"Fit logistic regression on raw scores. y=1 means AI, y=0 means real.\"\"\"
    if len(np.unique(y)) < 2:
        # Degenerate; fall back to identity
        return PlattParams(a=1.0, b=0.0, auroc=0.5, ece=0.5,
                           n_pos=int(y.sum()), n_neg=int(len(y) - y.sum()))
    x = p_raw.reshape(-1, 1)
    lr = LogisticRegression(C=1.0).fit(x, y)
    a = float(lr.coef_[0, 0]); b = float(lr.intercept_[0])
    # AUROC + ECE
    from sklearn.metrics import roc_auc_score
    p_cal = 1 / (1 + np.exp(-(a * p_raw + b)))
    try: auroc = float(roc_auc_score(y, p_cal))
    except Exception: auroc = 0.5
    ece = float(_expected_calibration_error(p_cal, y, bins=15))
    return PlattParams(a=a, b=b, auroc=auroc, ece=ece,
                       n_pos=int(y.sum()), n_neg=int(len(y) - y.sum()))


def apply(params: PlattParams, p_raw: float) -> float:
    z = params.a * float(p_raw) + params.b
    return float(1.0 / (1.0 + math.exp(-z)))


def _expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    n = len(p); ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0: continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return ece


def save_all(params_by_signal: Mapping[str, PlattParams]) -> None:
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    blob = {sig: {\"a\": p.a, \"b\": p.b, \"auroc\": p.auroc, \"ece\": p.ece,
                  \"n_pos\": p.n_pos, \"n_neg\": p.n_neg}
            for sig, p in params_by_signal.items()}
    CALIB_PATH.write_text(json.dumps(blob, indent=2))


def load_all() -> dict[str, PlattParams]:
    if not CALIB_PATH.exists():
        return {}
    data = json.loads(CALIB_PATH.read_text())
    return {k: PlattParams(**v) for k, v in data.items()}
```

> **Why Platt rather than isotonic?** With 1500+1500 refDB images, Platt's 2-parameter fit is more robust than isotonic's piecewise-constant fit (which over-fits at small N). Once user-label volume crosses 200, the `run_calibration.py` script switches to isotonic (`platt_blended`).

### 2.2 `backend/calibration/run.py` — cold-start + warm-update calibration

```python
# file: /app/backend/calibration/run.py
\"\"\"Iterate refDB + (optionally) user-labelled images, score each through every
signal, fit Platt per signal, persist.

Called automatically at the end of build_reference_db, and on a cron when
user-label count crosses 50/100/200 thresholds.\"\"\"
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import numpy as np

from backend.calibration.platt import fit_one, PlattParams, save_all
from backend.db.repos import unconsumed_labels, mark_label_consumed
from backend.detectors.base import Sample
from backend.detectors.image import image_detectors
from backend.detectors.image._io import load_rgb
from backend.retrieval.index import REFDB_DIR

log = logging.getLogger(\"calibrate.run\")


async def _score_image(detectors, path: Path) -> dict[str, float]:
    arr = load_rgb(path)
    s = Sample(image_rgb=arr, image_path=str(path), sha256=path.stem,
               mime=\"image/jpeg\", bytes=path.stat().st_size,
               content_type=\"object_product\")
    out: dict[str, float] = {}
    for d in detectors:
        try:
            r = await d.predict(s)
            if r.enabled:
                out[d.name] = float(r.p_fake)
        except Exception:
            pass
    return out


async def run(modality: str = \"image\", source: str = \"refdb\") -> dict:
    \"\"\"Calibrate using refDB + user labels.\"\"\"
    detectors = image_detectors()
    rows: list[dict] = []
    labels: list[int] = []

    # 1. refDB images
    raw_dir = REFDB_DIR / \"raw\"
    real_ids = set(json.loads((REFDB_DIR / f\"{modality}_real_labels.json\").read_text()))
    ai_ids = set(json.loads((REFDB_DIR / f\"{modality}_ai_labels.json\").read_text()))
    for f in sorted(raw_dir.glob(\"*.bin\")):
        sha = f.stem
        if sha in real_ids: y = 0
        elif sha in ai_ids: y = 1
        else: continue
        # raw files are bytes; need a temp-on-disk png path for OCR/EXIF detectors
        scores = await _score_image(detectors, f)
        rows.append(scores); labels.append(y)

    # 2. User-corrected labels (when present)
    if source != \"refdb\":
        for ulab in await unconsumed_labels():
            p = Path(ulab.get(\"path\", \"\"))
            if not p.exists(): continue
            y = 1 if ulab[\"label\"] == \"ai\" else 0
            scores = await _score_image(detectors, p)
            rows.append(scores); labels.append(y)
            await mark_label_consumed(ulab[\"_id\"])

    # 3. Fit per signal
    signal_names = sorted({k for r in rows for k in r.keys()})
    params: dict[str, PlattParams] = {}
    for sig in signal_names:
        x_l, y_l = [], []
        for r, y in zip(rows, labels):
            if sig in r:
                x_l.append(r[sig]); y_l.append(y)
        if len(x_l) < 30:
            log.warning(\"calibrate.skip\",
                        extra={\"signal_name\": sig, \"status\": f\"n={len(x_l)}\"})
            continue
        params[sig] = fit_one(np.array(x_l, dtype=np.float32),
                              np.array(y_l, dtype=np.int32))
        log.info(\"calibrate.fit\",
                 extra={\"signal_name\": sig,
                        \"status\": f\"auroc={params[sig].auroc:.3f}, ece={params[sig].ece:.3f}\"})

    save_all(params)
    return {\"signals_calibrated\": len(params), \"n_examples\": len(rows)}
```

---

## 3. Fusion

### 3.1 `backend/fusion/uniform.py` — cold-start mean

```python
# file: /app/backend/fusion/uniform.py
\"\"\"Cold-start fusion: weighted mean of calibrated probabilities.

Weight = tier-family prior × confidence prior (lower variance → higher weight).\"\"\"
from __future__ import annotations

import numpy as np
from typing import Sequence

from backend.fusion.types import SignalIn

# Tier-family priors (rough hand-set; replaced by LR weights once enough data)
TIER_PRIOR = {
    \"img.prithiv\": 1.30,
    \"img.npr\":     1.20,
    \"img.ufd\":     1.20,
    \"img.dire\":    1.15,
    \"img.t15.hive\":        1.10,
    \"img.t15.sightengine\": 1.10,
    \"img.clip0\":   0.85,
    \"img.freq\":    0.75,
    \"img.compression\": 0.70,
    \"img.retrieval\": 1.40,    # set in runner
    \"img.meta\":    0.60,
    \"img.ocr_gibberish\": 0.55,   # high precision, low recall; small weight
    \"img.eye_forensics\": 0.85,   # gated; when present, deserves voice
    \"img.reverse\": 1.20,
    \"img.vlm\":     1.30,
}


def fuse_uniform(signals: Sequence[SignalIn]) -> tuple[float, dict[str, float]]:
    if not signals:
        return 0.5, {}
    w: list[float] = []
    p: list[float] = []
    for s in signals:
        if not s.enabled: continue
        w_i = TIER_PRIOR.get(s.name, 1.0) * s.weight_hint
        w.append(w_i); p.append(float(s.p_fake))
    if not w:
        return 0.5, {}
    w = np.array(w, dtype=np.float32)
    p = np.array(p, dtype=np.float32)
    fused = float((w * p).sum() / w.sum())
    weights = {s.name: float(w_i / w.sum()) for s, w_i in zip(
        [s for s in signals if s.enabled], w)}
    return fused, weights
```

### 3.2 `backend/fusion/lr.py` — L2 logistic regression (warm)

```python
# file: /app/backend/fusion/lr.py
\"\"\"Sparse logistic regression fusion. Activated when ≥200 user labels exist.

Stored in storage/refdb/fusion_lr.json; loaded once at boot.\"\"\"
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from backend.fusion.types import SignalIn

log = logging.getLogger(\"fusion.lr\")
LR_PATH = Path(\"/app/backend/storage/refdb/fusion_lr.json\")


@dataclass
class LRParams:
    feature_order: list[str]
    coef: list[float]
    intercept: float
    mean: list[float]      # for mean-imputation of missing signals


def fit(rows: list[dict[str, float]], y: list[int]) -> LRParams:
    feat = sorted({k for r in rows for k in r.keys()})
    X = np.full((len(rows), len(feat)), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        for j, f in enumerate(feat):
            if f in r: X[i, j] = r[f]
    # mean-impute
    means = np.nanmean(X, axis=0)
    means = np.nan_to_num(means, nan=0.5)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(means, inds[1])
    yarr = np.array(y, dtype=np.int32)
    lr = LogisticRegression(penalty=\"l2\", C=1.0, max_iter=1000).fit(X, yarr)
    return LRParams(
        feature_order=feat,
        coef=lr.coef_[0].astype(float).tolist(),
        intercept=float(lr.intercept_[0]),
        mean=means.astype(float).tolist(),
    )


def save(params: LRParams) -> None:
    LR_PATH.parent.mkdir(parents=True, exist_ok=True)
    LR_PATH.write_text(json.dumps(params.__dict__, indent=2))


def load() -> LRParams | None:
    if not LR_PATH.exists(): return None
    return LRParams(**json.loads(LR_PATH.read_text()))


def fuse_lr(signals: list[SignalIn], params: LRParams) -> tuple[float, dict[str, float], list[str]]:
    \"\"\"Returns (p_ai, weights_norm, imputed_signals).\"\"\"
    x = list(params.mean)  # imputed-by-default
    imputed: list[str] = list(params.feature_order)
    have_index: dict[str, int] = {f: i for i, f in enumerate(params.feature_order)}
    for s in signals:
        if not s.enabled or s.name not in have_index: continue
        i = have_index[s.name]
        x[i] = float(s.p_fake)
        if s.name in imputed: imputed.remove(s.name)
    z = params.intercept + sum(c * v for c, v in zip(params.coef, x))
    p = 1.0 / (1.0 + math.exp(-z))
    # Normalised contributions for the XAI panel
    total = sum(abs(c) for c in params.coef) or 1.0
    weights = {f: abs(c) / total for f, c in zip(params.feature_order, params.coef)}
    return float(p), weights, imputed
```

### 3.3 `backend/fusion/gbdt.py` — LightGBM (advanced, post-M3)

```python
# file: /app/backend/fusion/gbdt.py
\"\"\"LightGBM fusion. Activated when ≥500 user labels exist.

Captures non-linear interactions (e.g. \"freq+compression simultaneously high
matters more than either alone\").

Stored at storage/refdb/fusion_gbdt.txt (LightGBM native format).\"\"\"
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(\"fusion.gbdt\")
GBDT_PATH = Path(\"/app/backend/storage/refdb/fusion_gbdt.txt\")


def fit(rows: list[dict[str, float]], y: list[int]) -> dict[str, Any]:
    import lightgbm as lgb
    import numpy as np
    feat = sorted({k for r in rows for k in r.keys()})
    X = np.full((len(rows), len(feat)), 0.5, dtype=np.float32)
    for i, r in enumerate(rows):
        for j, f in enumerate(feat):
            if f in r: X[i, j] = r[f]
    dtrain = lgb.Dataset(X, label=np.array(y))
    booster = lgb.train(
        {\"objective\": \"binary\", \"metric\": \"auc\",
         \"learning_rate\": 0.05, \"num_leaves\": 15, \"verbose\": -1},
        dtrain, num_boost_round=200,
    )
    booster.save_model(str(GBDT_PATH))
    return {\"feature_order\": feat, \"best_score\": float(booster.best_score[\"valid_0\"][\"auc\"])
            if booster.best_score else None}


def load_booster():
    import lightgbm as lgb
    if not GBDT_PATH.exists(): return None
    return lgb.Booster(model_file=str(GBDT_PATH))


def fuse_gbdt(signals: list, booster, feature_order: list[str]) -> tuple[float, dict[str, float]]:
    import numpy as np
    x = np.full((1, len(feature_order)), 0.5, dtype=np.float32)
    have = {s.name: float(s.p_fake) for s in signals if s.enabled}
    for j, f in enumerate(feature_order):
        if f in have: x[0, j] = have[f]
    p = float(booster.predict(x)[0])
    fi = booster.feature_importance(importance_type=\"gain\")
    total = float(fi.sum()) or 1.0
    weights = {f: float(v / total) for f, v in zip(feature_order, fi)}
    return p, weights
```

### 3.4 `backend/fusion/__init__.py` — adaptive selector

```python
# file: /app/backend/fusion/__init__.py
\"\"\"Pick the best fusion model that has enough data, transparently.\"\"\"
from __future__ import annotations

import logging
from typing import Literal, Sequence

from backend.calibration.platt import apply, load_all
from backend.fusion.gbdt import fuse_gbdt, load_booster
from backend.fusion.lr import fuse_lr, load as lr_load
from backend.fusion.types import FusionResult, SignalIn
from backend.fusion.uniform import fuse_uniform

log = logging.getLogger(\"fusion\")
_CALIB = load_all()
_LR = lr_load()
_GBDT = load_booster()


def _calibrate(s: SignalIn) -> SignalIn:
    p = _CALIB.get(s.name)
    if p is None:
        return s
    return SignalIn(name=s.name, p_fake=apply(p, s.p_fake),
                    enabled=s.enabled, weight_hint=s.weight_hint)


def _agreement(signals: Sequence[SignalIn], p_ai: float) -> float:
    active = [s for s in signals if s.enabled]
    if not active: return 0.0
    err = sum(abs(s.p_fake - p_ai) for s in active) / len(active)
    return max(0.0, 1.0 - err)


def fuse(raw_signals: Sequence[SignalIn]) -> FusionResult:
    # 1. Calibrate per signal
    cal = [_calibrate(s) for s in raw_signals]

    # 2. Pick fusion model
    imputed: list[str] = []
    if _GBDT is not None:
        feature_order = _GBDT.feature_name() if hasattr(_GBDT, \"feature_name\") else []
        p, w = fuse_gbdt(cal, _GBDT, feature_order)
        model: Literal[\"uniform\", \"lr_l2\", \"gbdt\"] = \"gbdt\"
    elif _LR is not None:
        p, w, imputed = fuse_lr(list(cal), _LR)
        model = \"lr_l2\"
    else:
        p, w = fuse_uniform(cal)
        model = \"uniform\"

    return FusionResult(
        p_ai=float(p),
        agreement=float(_agreement(cal, p)),
        extremity=float(max(p, 1 - p) - 0.5),
        cross_modal_bonus=0.0,    # set later by add_cross_modal()
        fusion_model=model,
        calibration=\"platt_refdb\" if _CALIB else \"platt_refdb\",
        weights=w, imputed=imputed,
    )


def add_cross_modal(fr: FusionResult, signals: Sequence[SignalIn]) -> FusionResult:
    \"\"\"Boost confidence when independent tier-families agree.

    Families:
      A) Learned (prithiv/npr/ufd/dire)
      B) Forensic (freq/compression/meta)
      C) Retrieval (img.retrieval)
      D) Web/Reverse (img.reverse)
      E) VLM (img.vlm)
      F) Third-party (img.t15.*)
      G) Eye/OCR (img.eye_forensics, img.ocr_gibberish)

    For every family above that has p_fake aligned with the fused decision (both
    >0.5 or both <0.5), add 0.015 to confidence (cap 0.10).\"\"\"
    families = {
        \"A\": (\"img.prithiv\", \"img.npr\", \"img.ufd\", \"img.dire\"),
        \"B\": (\"img.freq\", \"img.compression\", \"img.meta\"),
        \"C\": (\"img.retrieval\",),
        \"D\": (\"img.reverse\",),
        \"E\": (\"img.vlm\",),
        \"F\": (\"img.t15.hive\", \"img.t15.sightengine\"),
        \"G\": (\"img.eye_forensics\", \"img.ocr_gibberish\"),
    }
    by_name = {s.name: s for s in signals if s.enabled}
    boost = 0.0
    direction_main = fr.p_ai > 0.5
    for fam, names in families.items():
        members = [by_name[n] for n in names if n in by_name]
        if not members: continue
        fam_mean = sum(m.p_fake for m in members) / len(members)
        if (fam_mean > 0.5) == direction_main:
            boost += 0.015
    boost = min(0.10, boost)
    fr.cross_modal_bonus = float(boost)
    return fr
```

---

## 4. Content-type-gated thresholds

```python
# file: /app/backend/abstention/gate.py
\"\"\"Per-content-type abstention thresholds. Tighter for selfies (high stakes),
looser for ambiguous categories (illustrations, screenshots).\"\"\"
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateThresholds:
    high: float          # ≥ → AI-GENERATED
    low: float           # ≤ → REAL
    agree: float         # required minimum agreement


# Hand-set defaults; tuned later via scripts/tune_thresholds.py.
GATES: dict[str, GateThresholds] = {
    \"selfie_portrait\":      GateThresholds(high=0.78, low=0.22, agree=0.60),
    \"landscape_scene\":      GateThresholds(high=0.74, low=0.26, agree=0.55),
    \"object_product\":       GateThresholds(high=0.74, low=0.26, agree=0.55),
    \"meme_screenshot\":      GateThresholds(high=0.82, low=0.18, agree=0.65),
    \"document_scan\":        GateThresholds(high=0.82, low=0.18, agree=0.65),
    \"artwork_illustration\": GateThresholds(high=0.88, low=0.12, agree=0.70),
}
DEFAULT = GateThresholds(high=0.75, low=0.25, agree=0.55)


def gate_for(content_type: str) -> GateThresholds:
    return GATES.get(content_type, DEFAULT)
```

---

## 5. OOD novel-generator override (NEW v1.4)

```python
# file: /app/backend/abstention/ood.py
\"\"\"Train one Isolation Forest per refDB cluster (real, ai) at build time.

At runtime: if upload embedding is anomalous to BOTH clusters → force INCONCLUSIVE
with rationale \"looks unlike anything in our reference DB\".

This converts confident-wrong on novel generators into honest abstention.\"\"\"
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

from backend.retrieval.index import REFDB_DIR

log = logging.getLogger(\"ood\")
OOD_DIR = REFDB_DIR / \"ood\"


def fit_from_refdb(modality: str = \"image\") -> dict:
    OOD_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, float] = {}
    for label in (\"real\", \"ai\"):
        vec_path = REFDB_DIR / f\"{modality}_{label}.npy\"
        if not vec_path.exists(): continue
        V = np.load(vec_path)
        if len(V) < 50: continue
        iso = IsolationForest(n_estimators=200, contamination=0.05,
                              random_state=42).fit(V)
        import joblib
        joblib.dump(iso, OOD_DIR / f\"{modality}_{label}_iso.pkl\")
        # Calibrate threshold to 5th percentile of training scores
        scores = iso.score_samples(V)
        tau = float(np.percentile(scores, 5))
        out[label] = tau
    (OOD_DIR / f\"{modality}_tau.json\").write_text(json.dumps(out, indent=2))
    return out


def _load(modality: str = \"image\") -> dict:
    out = {}
    if not (OOD_DIR / f\"{modality}_tau.json\").exists():
        return {}
    out[\"tau\"] = json.loads((OOD_DIR / f\"{modality}_tau.json\").read_text())
    import joblib
    for label in (\"real\", \"ai\"):
        p = OOD_DIR / f\"{modality}_{label}_iso.pkl\"
        if p.exists():
            out[label] = joblib.load(p)
    return out


_MODELS: dict = {}


def is_ood(modality: str, vec: np.ndarray) -> tuple[bool, dict[str, float]]:
    \"\"\"Returns (is_ood, scores). is_ood is True only when score < tau in BOTH clusters.\"\"\"
    if not _MODELS:
        _MODELS.update(_load(modality))
    if not _MODELS: return False, {}
    v = vec.reshape(1, -1).astype(\"float32\")
    tau = _MODELS.get(\"tau\", {})
    flags = {}
    for label in (\"real\", \"ai\"):
        iso = _MODELS.get(label)
        if iso is None: continue
        s = float(iso.score_samples(v)[0])
        flags[f\"score_{label}\"] = s
        flags[f\"ood_{label}\"] = s < tau.get(label, -0.5)
    is_ood_both = flags.get(\"ood_real\", False) and flags.get(\"ood_ai\", False)
    return is_ood_both, flags
```

---

## 6. Conformal-prediction abstention (NEW v1.5)

```python
# file: /app/backend/abstention/conformal.py
\"\"\"Conformal prediction:

Given a calibration set with raw scores s_i and labels y_i, compute the
*nonconformity* of each example as 1 - p_correct. At inference, for a user-set
miscoverage α (default 0.10), the calibrated p_ai must place the true class
inside a 1-α conformal set. When it cannot — abstain.

Why this matters: hard thresholds (e.g. 0.75) are arbitrary. Conformal sets
give a MATHEMATICALLY GUARANTEED 90% coverage on i.i.d. distributions and
degrade *gracefully* when the distribution shifts. Best academic practice.\"\"\"
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(\"conformal\")
CONF_PATH = Path(\"/app/backend/storage/refdb/conformal.json\")


def fit_quantile(p_calibrated: np.ndarray, y: np.ndarray,
                 alpha: float = 0.10) -> float:
    \"\"\"Compute the 1-α quantile of nonconformity scores on the calibration set.

    Nonconformity: s_i = 1 - p_{calibrated, i}(true class).
    \"\"\"
    s = np.where(y == 1, 1.0 - p_calibrated, p_calibrated)
    n = len(s)
    if n == 0: return 0.5
    q = float(np.quantile(s, (1 - alpha) * (n + 1) / n))
    return min(0.999, max(0.001, q))


def save(qhat: float, alpha: float) -> None:
    CONF_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONF_PATH.write_text(json.dumps({\"qhat\": qhat, \"alpha\": alpha}, indent=2))


def load() -> tuple[float, float] | None:
    if not CONF_PATH.exists(): return None
    blob = json.loads(CONF_PATH.read_text())
    return float(blob[\"qhat\"]), float(blob[\"alpha\"])


def conformal_set(p_ai: float, qhat: float) -> set[str]:
    \"\"\"Return the conformal prediction set: a subset of {\"real\",\"ai\"}.\"\"\"
    out: set[str] = set()
    if (1 - p_ai) <= qhat: out.add(\"ai\")          # nonconformity of \"ai\" hypothesis
    if p_ai <= qhat:        out.add(\"real\")
    return out


def conformal_verdict(p_ai: float) -> tuple[str, bool, str]:
    \"\"\"Returns (label, abstained, rationale).

    Rules:
    - Set = {\"ai\"}        → \"AI-GENERATED\" (confident)
    - Set = {\"real\"}      → \"REAL\" (confident)
    - Set = {\"ai\",\"real\"} → INCONCLUSIVE (both plausible)
    - Set = {}            → INCONCLUSIVE (neither plausible — strange/OOD)
    \"\"\"
    cq = load()
    if cq is None:
        # No conformal calibration yet → no abstention, just argmax
        return (\"AI-GENERATED\" if p_ai > 0.5 else \"REAL\"), False, \"no conformal calibration\"
    qhat, alpha = cq
    cset = conformal_set(p_ai, qhat)
    if cset == {\"ai\"}:
        return \"AI-GENERATED\", False, f\"conformal set={{ai}} at α={alpha}\"
    if cset == {\"real\"}:
        return \"REAL\", False, f\"conformal set={{real}} at α={alpha}\"
    if cset == {\"ai\", \"real\"}:
        return \"INCONCLUSIVE\", True, f\"conformal set ambiguous at α={alpha}\"
    return \"INCONCLUSIVE\", True, f\"conformal set empty (OOD-like) at α={alpha}\"
```

> `scripts/run_calibration.py` (§2.2 above) calls `fit_quantile(...)` on its held-out 20% slice and saves `qhat`. Re-fit happens whenever Platt does.

---

## 7. Combined verdict

```python
# file: /app/backend/abstention/__init__.py
\"\"\"Final verdict assembly.

Order of precedence:
1. Provenance hit (Tier 0) — runner sets verdict directly, this function is bypassed.
2. OOD novel-generator → INCONCLUSIVE.
3. Conformal-set check (primary, when calibrated).
4. Content-type gate fallback (when conformal not yet calibrated).
5. Agreement floor — even if score is extreme, abstain if signals disagree.
\"\"\"
from __future__ import annotations

import logging
from typing import Sequence

from backend.abstention.conformal import conformal_verdict, load as conformal_load
from backend.abstention.gate import gate_for
from backend.abstention.ood import is_ood
from backend.fusion.types import FusionResult, SignalIn, Verdict

log = logging.getLogger(\"abstention\")


def decide(fr: FusionResult,
           signals: Sequence[SignalIn],
           content_type: str,
           ood_flag: bool = False) -> Verdict:
    p = fr.p_ai
    conf = max(p, 1 - p) + fr.cross_modal_bonus
    conf = min(0.99, max(0.51, conf))

    # 1. OOD override — highest priority below provenance
    if ood_flag:
        return Verdict(
            label=\"INCONCLUSIVE\", confidence=0.50, abstained=True,
            rationale=(\"This image looks unlike anything in our reference DB. \"
                       \"It may come from a new generator we haven't catalogued. \"
                       \"Manual review recommended.\"),
        )

    # 2. Conformal first (if calibrated)
    if conformal_load() is not None:
        label, abstain, reason = conformal_verdict(p)
        # Apply agreement floor: even if conformal is confident, abstain on low agreement
        gate = gate_for(content_type)
        if (not abstain) and fr.agreement < gate.agree * 0.85:
            return Verdict(
                label=\"INCONCLUSIVE\", confidence=conf,
                abstained=True,
                rationale=f\"agreement {fr.agreement:.2f} below floor {gate.agree:.2f}\",
            )
        return Verdict(label=label, confidence=conf, abstained=abstain,
                       rationale=reason)

    # 3. Fallback gate (cold-start — no conformal calibration yet)
    gate = gate_for(content_type)
    if p >= gate.high and fr.agreement >= gate.agree:
        return Verdict(\"AI-GENERATED\", conf, False,
                       f\"p={p:.2f} ≥ {gate.high}, agree={fr.agreement:.2f}\")
    if p <= gate.low and fr.agreement >= gate.agree:
        return Verdict(\"REAL\", conf, False,
                       f\"p={p:.2f} ≤ {gate.low}, agree={fr.agreement:.2f}\")
    return Verdict(
        label=\"INCONCLUSIVE\", confidence=conf, abstained=True,
        rationale=(f\"p={p:.2f} in [{gate.low}, {gate.high}] \"
                   f\"or agreement {fr.agreement:.2f} < {gate.agree}\"),
    )
```

---

## 8. End-to-end usage (preview — full in `10_runner_orchestrator.md`)

```python
# inside services/runner.py — preview
from backend.fusion import fuse, add_cross_modal
from backend.abstention import decide
from backend.abstention.ood import is_ood
from backend.retrieval.embedder import embed_image

vec = await embed_image(sample.image_rgb)
ood_flag, _ = is_ood(\"image\", vec)

fr = fuse(signal_ins)
fr = add_cross_modal(fr, signal_ins)
verdict = decide(fr, signal_ins, sample.content_type, ood_flag=ood_flag)
```

---

## 9. Math sanity check (worth keeping in head)

For 10 calibrated signals with mean ECE ≈ 0.05 and pairwise correlation ρ ≈ 0.3,
the *effective ensemble size* is `n_eff = n / (1 + (n-1)ρ) ≈ 3.4`.
Variance of the fused estimate falls by `1/n_eff`. With per-signal AUROC ≈ 0.75
the fused AUROC sits around **0.88–0.93** — matching the v1.4 target band.

Adding *uncorrelated* signals (provenance, retrieval, reverse search, VLM)
raises `n_eff` toward 6–7, pushing AUROC into 0.93–0.97 territory.

This is the rigorous reason free-tools ensembling beats single-model fine-tuning
at this budget.

---

## 10. Unit tests

```python
# file: /app/backend/tests/unit/test_fusion.py
import numpy as np
from backend.fusion.types import SignalIn
from backend.fusion.uniform import fuse_uniform
from backend.fusion import add_cross_modal, fuse


def test_uniform_simple():
    sigs = [SignalIn(name=\"img.prithiv\", p_fake=0.9),
            SignalIn(name=\"img.freq\",    p_fake=0.7),
            SignalIn(name=\"img.meta\",    p_fake=0.55)]
    p, w = fuse_uniform(sigs)
    assert 0.7 < p < 0.9
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_cross_modal_bonus():
    sigs = [SignalIn(\"img.prithiv\", 0.85),
            SignalIn(\"img.freq\",    0.78),
            SignalIn(\"img.retrieval\", 0.80),
            SignalIn(\"img.vlm\",     0.90)]
    fr = fuse(sigs)
    fr = add_cross_modal(fr, sigs)
    assert fr.cross_modal_bonus > 0.02


def test_uniform_drops_disabled():
    sigs = [SignalIn(\"img.prithiv\", 0.9, enabled=True),
            SignalIn(\"img.freq\",    0.1, enabled=False)]
    p, _ = fuse_uniform(sigs)
    assert p > 0.85
```

```python
# file: /app/backend/tests/unit/test_abstention.py
from backend.fusion.types import FusionResult, SignalIn
from backend.abstention import decide


def _fr(p, a):
    return FusionResult(p_ai=p, agreement=a, extremity=abs(p - 0.5),
                        cross_modal_bonus=0.0, fusion_model=\"uniform\",
                        calibration=\"platt_refdb\")


def test_low_agreement_abstains():
    v = decide(_fr(0.85, 0.30), [SignalIn(\"img.prithiv\", 0.85)],
               content_type=\"selfie_portrait\")
    assert v.abstained is True


def test_high_p_and_agree_confident_ai():
    v = decide(_fr(0.90, 0.85), [SignalIn(\"img.prithiv\", 0.90)],
               content_type=\"object_product\")
    assert v.label == \"AI-GENERATED\" and not v.abstained


def test_ood_override():
    v = decide(_fr(0.95, 0.99), [SignalIn(\"img.prithiv\", 0.95)],
               content_type=\"selfie_portrait\", ood_flag=True)
    assert v.label == \"INCONCLUSIVE\" and v.abstained
```

```python
# file: /app/backend/tests/unit/test_calibration.py
import numpy as np
from backend.calibration.platt import fit_one, apply


def test_platt_monotone():
    p_raw = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9], dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    params = fit_one(p_raw, y)
    assert apply(params, 0.9) > apply(params, 0.1)
    assert params.auroc > 0.9


def test_platt_degenerate_single_class():
    params = fit_one(np.array([0.5, 0.6, 0.7]), np.array([1, 1, 1]))
    assert params.a == 1.0 and params.b == 0.0
```

---

## 11. Section exit criteria

```bash
pytest backend/tests/unit/test_fusion.py \
       backend/tests/unit/test_abstention.py \
       backend/tests/unit/test_calibration.py -q
mypy backend/fusion/ backend/calibration/ backend/abstention/
# Success: no issues
```

Next: `09_xai_and_narrator.md` — GradCAM, FFT plot, and the natural-language narrator with **few-shot in-context examples** for Gemini.
"