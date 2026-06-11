"""Module B — Compression history (docs 8.2 Module B).

ELA (recompress@q90, amplified diff heatmap PNG) + JPEG-ghost sweep.
Localizes spliced/inpainted regions with divergent compression history;
blind to whole-image generations by design (other modules cover).
"""
import numpy as np
from PIL import Image
import io

from base import EvidenceModule, ImageContext
from schemas import Artifact, DegradationState, ModuleOutput
from triage import ghost_curve
from viz import save_curve, save_heatmap

_BLOCK = 32
_ANALYSIS_MAX = 1024


class CompressionModule(EvidenceModule):
    module_id = "compression_history"
    version = "0.1.0"

    def assess(self, ctx: ImageContext, d: DegradationState, base_reliability: float) -> ModuleOutput:
        rgb = ctx.pil.convert("RGB")
        scale = 1.0
        if max(rgb.size) > _ANALYSIS_MAX:
            scale = _ANALYSIS_MAX / max(rgb.size)
            rgb_a = rgb.resize((max(8, int(rgb.width * scale)), max(8, int(rgb.height * scale))), Image.LANCZOS)
        else:
            rgb_a = rgb

        # --- ELA at q90 ---
        arr = np.asarray(rgb_a, dtype=np.float64)
        buf = io.BytesIO()
        rgb_a.save(buf, "JPEG", quality=90)
        buf.seek(0)
        rec = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float64)
        diff = np.abs(arr - rec).mean(axis=2)

        ela_name = "ela_heatmap.png"
        save_heatmap(np.clip(diff * 12, 0, 255), ctx.artifact_abs(ela_name))

        # block statistics
        h, w = diff.shape
        bh, bw = h // _BLOCK, w // _BLOCK
        blocks = diff[: bh * _BLOCK, : bw * _BLOCK].reshape(bh, _BLOCK, bw, _BLOCK).mean(axis=(1, 3))
        mu, sd = float(blocks.mean()), float(blocks.std())
        cv = sd / (mu + 1e-9)
        anomaly_mask = (blocks > mu + 2.8 * sd) & (blocks > 1.6 * mu)
        n_anom = int(anomaly_mask.sum())
        region = None
        max_ratio = float(blocks.max() / (mu + 1e-9))
        if 0 < n_anom <= 0.25 * blocks.size:
            rows, cols = np.where(anomaly_mask)
            inv = 1.0 / scale
            region = [int(cols.min() * _BLOCK * inv), int(rows.min() * _BLOCK * inv),
                      int((cols.max() + 1) * _BLOCK * inv), int((rows.max() + 1) * _BLOCK * inv)]

        # --- JPEG ghost sweep ---
        qs, curve, minima = ghost_curve(rgb)
        ghost_name = "ghost_curve.png"
        save_curve(qs, curve, ctx.artifact_abs(ghost_name),
                   "JPEG-ghost sweep: recompression MSE vs quality", "recompression quality q", "MSE",
                   marks=minima)
        extra_ghosts = [m for m in minima if d.jpeg_quality_est is None or abs(m - d.jpeg_quality_est) > 7]

        artifacts = [Artifact(
            type="ela_heatmap",
            description="Error-level analysis: residual after q90 recompression, amplified 12x",
            strength=min(1.0, cv),
            visual=ctx.artifact_rel(ela_name),
            checkable_claim=(f"ELA(q90) block-energy coefficient of variation = {cv:.2f}; "
                             f"max 32px-block energy = {max_ratio:.1f}x image mean; "
                             f"{n_anom} of {blocks.size} blocks exceed mean+2.8sigma"),
        )]
        if minima:
            artifacts.append(Artifact(
                type="jpeg_ghost",
                description="JPEG-ghost minima in the recompression MSE curve",
                strength=0.5,
                visual=ctx.artifact_rel(ghost_name),
                checkable_claim=(f"Recompression-MSE local minima at q={minima} "
                                 f"(triage-estimated last quality: {d.jpeg_quality_est}); minima away from "
                                 "the last quality indicate prior compression generations"),
            ))
        else:
            artifacts.append(Artifact(
                type="jpeg_ghost",
                description="JPEG-ghost sweep found no secondary minima",
                strength=0.2,
                visual=ctx.artifact_rel(ghost_name),
                checkable_claim="Recompression-MSE curve q=50..100 is unimodal: no concealed prior JPEG quality detected",
            ))

        if ctx.fmt == "jpeg":
            if region is not None and cv > 0.5:
                e, direction, conf = -0.55, "manipulated", 0.7
                artifacts[0].location = {"regions": [region]}
                artifacts[0].description += " — localized anomaly region detected"
            elif extra_ghosts:
                e, direction, conf = -0.15, "neutral", 0.5
            else:
                e, direction, conf = 0.3, "authentic", 0.55
        else:  # png/webp containers
            if minima:
                e, direction, conf = -0.25, "synthetic", 0.6
                artifacts.append(Artifact(
                    type="concealed_history",
                    description="Lossless container carries JPEG ghosts: prior lossy history concealed",
                    strength=0.6,
                    checkable_claim=f"PNG/lossless container but ghost minima at q={minima}: "
                                    "the image was previously JPEG-compressed and re-wrapped",
                ))
            else:
                e, direction, conf = 0.0, "neutral", 0.3
                artifacts[0].description += " — no compression history in lossless container (uninformative for splice detection)"

        return ModuleOutput(
            module_id=self.module_id, version=self.version,
            evidence_score=round(e, 4), reliability_score=base_reliability,
            confidence_score=round(conf, 4), verdict_direction=direction,
            artifacts=artifacts,
        )


MODULE = CompressionModule
