"""EvidenceModule base class and shared image context.

Contract (docs 4.1 + execution prompt standard #1):
- run(image_ctx, degradation_state) -> ModuleOutput
- built-in timing, exception capture (a crashing module returns
  unavailable_reason="internal_error", NEVER crashes the pipeline)
- reliability lookup r_m(d) is performed here and passed to assess();
  availability checks inside a module may only LOWER it.
- A module with unavailable_reason set is forced to reliability 0 /
  evidence 0 (fail-closed, never fail-silent).
"""
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from reliability import reliability
from schemas import DegradationState, ModuleOutput

logger = logging.getLogger("argus.module")


@dataclass
class ImageContext:
    """Everything a module may read. Modules never import other modules."""

    pil: Image.Image
    raw_bytes: bytes
    sha256: str
    fmt: str  # "jpeg" | "png" | "webp" | ...
    verdict_id: str
    artifact_dir: Path  # /app/backend/artifacts/{verdict_id}
    src_path: Optional[Path] = None  # temp on-disk copy (deleted post-pipeline)

    def artifact_rel(self, filename: str) -> str:
        return f"artifacts/{self.verdict_id}/{filename}"

    def artifact_abs(self, filename: str) -> Path:
        return self.artifact_dir / filename


class EvidenceModule(ABC):
    module_id: str = "abstract"
    version: str = "0.0.0"

    @abstractmethod
    def assess(self, ctx: ImageContext, d: DegradationState, base_reliability: float) -> ModuleOutput:
        """Module forensic logic. May raise — run() captures everything."""

    def _unavailable(self, reason: str) -> ModuleOutput:
        return ModuleOutput(
            module_id=self.module_id,
            version=self.version,
            evidence_score=0.0,
            reliability_score=0.0,
            confidence_score=0.0,
            verdict_direction="neutral",
            artifacts=[],
            unavailable_reason=reason,
        )

    def run(self, ctx: ImageContext, d: DegradationState) -> ModuleOutput:
        t0 = time.perf_counter()
        base_r = reliability(self.module_id, d)
        try:
            out = self.assess(ctx, d, base_r)
        except Exception:
            logger.exception("module %s crashed; isolated per contract", self.module_id)
            out = self._unavailable("internal_error")
        # enforce contract invariants
        out.module_id = self.module_id
        out.version = self.version
        if out.unavailable_reason is not None:
            out.evidence_score = 0.0
            out.reliability_score = 0.0
            out.confidence_score = 0.0
            out.verdict_direction = "neutral"
        else:
            out.reliability_score = min(out.reliability_score, base_r)
        out.compute_ms = int((time.perf_counter() - t0) * 1000)
        return out


class FailedImportModule(EvidenceModule):
    """Registered in place of a module whose file failed to import.
    Fail-closed: the dead module stays visible in unavailable_evidence."""

    def __init__(self, module_id: str, error: str):
        self.module_id = module_id
        self.version = "0.0.0"
        self._error = error

    def assess(self, ctx, d, base_reliability):
        return self._unavailable(f"import_error: {self._error[:160]}")
