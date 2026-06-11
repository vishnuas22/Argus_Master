"""Stub module — only registered when ARGUS_ENABLE_STUB=1.

Used by the M1 gate and crash-isolation tests:
  ARGUS_STUB_MODE=ok    -> emits a fixed neutral-ish output
  ARGUS_STUB_MODE=crash -> raises (pipeline must survive, reporting
                           unavailable_reason="internal_error")
"""
import os

from base import EvidenceModule, ImageContext
from schemas import Artifact, DegradationState, ModuleOutput


class StubModule(EvidenceModule):
    module_id = "stub"
    version = "0.1.0"

    def assess(self, ctx: ImageContext, d: DegradationState, base_reliability: float) -> ModuleOutput:
        if os.environ.get("ARGUS_STUB_MODE", "ok") == "crash":
            raise RuntimeError("intentional stub crash for isolation test")
        return ModuleOutput(
            module_id=self.module_id,
            version=self.version,
            evidence_score=0.1,
            reliability_score=base_reliability,
            confidence_score=0.5,
            verdict_direction="neutral",
            artifacts=[Artifact(
                type="stub_finding",
                description="Stub module placeholder finding",
                strength=0.1,
                checkable_claim=f"Stub observed an image of {ctx.pil.width}x{ctx.pil.height} pixels in {ctx.fmt} format",
            )],
        )


MODULE = StubModule
