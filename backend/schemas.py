"""ARGUS contract schemas.

Binding contracts from /app/docs:
- docs 04_PHASE4 section 4.1  -> ModuleOutput / Artifact / DegradationState
- docs 07_PHASE7 section 7.1  -> Verdict (the API response contract)

Any change here requires bumping SCHEMA_VERSION and updating the docs in the
same commit (continuous-build rule).
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"
PIPELINE_VERSION = "0.1.0"

Direction = Literal["synthetic", "authentic", "neutral", "manipulated"]
Capacity = Literal["HIGH", "MODERATE", "LOW"]

HYPOTHESES = ["camera_original", "ai_generated", "manipulated"]


class Artifact(BaseModel):
    """A named, checkable finding (docs 4.1)."""

    type: str
    description: str
    location: Optional[Dict[str, Any]] = None  # e.g. {"regions": [[x1,y1,x2,y2], ...]}
    strength: float = Field(ge=0.0, le=1.0)
    visual: Optional[str] = None  # relative path "artifacts/{verdict_id}/file.png"
    checkable_claim: str


class ModuleOutput(BaseModel):
    """The module output contract (docs 4.1). Three scores, never collapsed."""

    module_id: str
    version: str
    evidence_score: float = Field(ge=-1.0, le=1.0)
    reliability_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    verdict_direction: Direction
    artifacts: List[Artifact] = Field(default_factory=list)
    unavailable_reason: Optional[str] = None
    compute_ms: int = 0


class DegradationState(BaseModel):
    """Tier-1 triage output `d` (docs 4.2.1, field names per docs 7.1)."""

    jpeg_quality_est: Optional[int] = None  # None for non-JPEG containers
    recompression_generations: int = 0
    resize_factor_est: float = 1.0  # 1.0 = no resampling detected
    screenshot_probability: float = Field(ge=0.0, le=1.0, default=0.0)
    effective_resolution: int = 0
    evidence_capacity: Capacity = "HIGH"


class ConformalInfo(BaseModel):
    alpha: float
    set: List[str]
    calibration_stratum: str


class VerdictCore(BaseModel):
    hypothesis_set: List[str]
    abstained: bool
    probabilities: Dict[str, float]  # camera_original / ai_generated / manipulated
    authenticity_score: float = Field(ge=0.0, le=1.0)
    trust_score: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    conformal: ConformalInfo


class VerdictInput(BaseModel):
    sha256: str
    dimensions: List[int]
    format: str
    degradation_state: DegradationState


class EvidenceRankEntry(BaseModel):
    rank: int
    module: str
    direction: Direction
    evidence_score: float = Field(ge=-1.0, le=1.0)
    reliability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    # v0 voting judge: exact additive contribution of this module to the fused
    # score (field name kept per docs 7.1 contract; SHAP proper arrives with
    # the LightGBM judge). See /app/memory/DECISIONS.md.
    shap_contribution: float
    likelihood_ratio: float
    artifacts: List[Artifact] = Field(default_factory=list)


class Contradiction(BaseModel):
    modules: List[str]
    description: str
    interpretation: str
    conflict_contribution: float


class UnavailableEvidence(BaseModel):
    module: str
    reason: str


class Explanation(BaseModel):
    summary: str
    detail: str
    audience: str = "analyst"


class VerdictMeta(BaseModel):
    module_versions: Dict[str, str]
    fusion_model: str
    reliability_curves: str
    total_compute_ms: int


class Verdict(BaseModel):
    """The verdict.json contract (docs 7.1)."""

    verdict_id: str
    schema_version: str = SCHEMA_VERSION
    input: VerdictInput
    verdict: VerdictCore
    evidence_ranking: List[EvidenceRankEntry] = Field(default_factory=list)
    contradictions: List[Contradiction] = Field(default_factory=list)
    unavailable_evidence: List[UnavailableEvidence] = Field(default_factory=list)
    explanation: Explanation
    meta: VerdictMeta
