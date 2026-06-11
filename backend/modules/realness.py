"""Module E — Real-distribution probe (docs 8.2 Module E).

Frozen DINOv2 kNN distance percentile against a bundled real-image reference
set (ref-v0, ~2.5k embeddings — small-N caveat capped into confidence_score
and stated in the checkable_claim). Never saw a fake; cannot overfit to one.
"""
import json
from pathlib import Path

import numpy as np

from base import EvidenceModule, ImageContext
from dino_service import DinoService
from schemas import Artifact, DegradationState, ModuleOutput
from viz import save_bar_panel

_DATA = Path(__file__).resolve().parent.parent / "data"
_REF = _DATA / "reference_embeddings.npy"
_CAL = _DATA / "knn_calib_distances.npy"
_META = _DATA / "reference_meta.json"
_K = 5

_cache = None


def _load():
    global _cache
    if _cache is None:
        ref = np.load(_REF).astype(np.float32)
        cal = np.sort(np.load(_CAL).astype(np.float32))
        meta = json.loads(_META.read_text()) if _META.exists() else {"n": len(ref), "version": "ref-v0"}
        _cache = (ref, cal, meta)
    return _cache


class RealnessModule(EvidenceModule):
    module_id = "real_distribution_probe"
    version = "0.1.0"

    def assess(self, ctx: ImageContext, d: DegradationState, base_reliability: float) -> ModuleOutput:
        service = DinoService.get()
        if not service.available():
            return self._unavailable("backbone_unavailable: torch/timm not installed")
        if not (_REF.exists() and _CAL.exists()):
            return self._unavailable("reference_set_missing")

        ref, cal, meta = _load()
        emb = service.embed(ctx.pil, cache_key=ctx.sha256)
        dists = 1.0 - ref @ emb
        top = np.sort(dists)[:_K]
        d_knn = float(top.mean())
        p = float(np.searchsorted(cal, d_knn) / len(cal))

        e = float(np.clip((1.0 - 2.0 * p) * 0.9, -1.0, 1.0))
        direction = "synthetic" if e < -0.15 else ("authentic" if e > 0.15 else "neutral")
        # small-N reference: confidence hard-capped at 0.65 (DECISIONS.md)
        confidence = min(0.65, 0.30 + 0.45 * abs(2.0 * p - 1.0))

        panel_name = "nn_distances.png"
        med, p95 = float(np.median(cal)), float(np.percentile(cal, 95))
        save_bar_panel(
            [f"NN{i+1}" for i in range(_K)], [float(x) for x in top],
            ctx.artifact_abs(panel_name),
            "Cosine distance to 5 nearest real-image neighbors",
            ref_lines=[("real-calib median", med, "#3FB950"), ("real-calib p95", p95, "#F85149")],
            ylabel="cosine distance",
        )

        claim = (f"Mean cosine distance to the {_K} nearest real-image neighbors = {d_knn:.4f}, "
                 f"at the {p*100:.1f}th percentile of the real-image calibration distribution "
                 f"(reference set {meta.get('version','ref-v0')}, n={meta.get('n', len(ref))}; "
                 f"calib median {med:.4f}, p95 {p95:.4f}; small-N caveat: confidence capped at 0.65)")

        return ModuleOutput(
            module_id=self.module_id, version=self.version,
            evidence_score=round(e, 4), reliability_score=base_reliability,
            confidence_score=round(confidence, 4), verdict_direction=direction,
            artifacts=[Artifact(
                type="embedding_outlier" if e < 0 else "embedding_inlier",
                description="DINOv2 kNN distance profile against the real-image reference set",
                strength=min(1.0, abs(e) + 0.1),
                visual=ctx.artifact_rel(panel_name),
                checkable_claim=claim,
            )],
        )


MODULE = RealnessModule
