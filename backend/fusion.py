"""Tier 3 — The Court. Fusion v0 (docs 05, M4).

reliability gate (floor 0.25) -> reliability-weighted voting (the mandated
fallback judge) -> logistic squashing to calibrated [0,1] -> conflict metric
K (docs 4.4.1) -> contradictions via the pattern table (docs 4.4.2) ->
trust/risk per docs 7.2 -> conformal-stub prediction set.

The judge is pluggable: any object with decide(features)->probabilities can
replace WeightedVotingJudge (LightGBM+conformal later) WITHOUT API changes.
"""
import math
from typing import Dict, List

from reliability import reliability_explanation
from schemas import (
    Contradiction,
    ConformalInfo,
    DegradationState,
    EvidenceRankEntry,
    HYPOTHESES,
    ModuleOutput,
    UnavailableEvidence,
    VerdictCore,
)

FUSION_MODEL = "voting-v0"
RELIABILITY_FLOOR = 0.25
CONFORMAL_ALPHA = 0.10

_STAT_MODULES = {"spectral_probe", "real_distribution_probe", "perturbation_probe"}
_RISK_ARTIFACT_TYPES = {
    "date_anomaly", "concealed_history", "implausible_cleanliness",
    "metadata_mismatch", "prompt_residue", "anti_forensics_indicator",
}

_CAPACITY_SCORE = {"HIGH": 1.0, "MODERATE": 0.6, "LOW": 0.25}
_STRATUM = {"HIGH": "low_degradation", "MODERATE": "moderate_degradation", "LOW": "heavy_degradation"}


class WeightedVotingJudge:
    """Reliability-weighted voting, logistic-squashed (docs 5.2.1 fallback)."""

    name = FUSION_MODEL

    def decide(self, survivors: List[ModuleOutput]) -> Dict[str, float]:
        if not survivors:
            return {h: round(1.0 / 3.0, 4) for h in HYPOTHESES}
        num = sum(m.reliability_score * m.evidence_score for m in survivors)
        den = sum(m.reliability_score for m in survivors) + 1e-9
        fused = num / den  # [-1, 1], + = authentic
        p_auth = 1.0 / (1.0 + math.exp(-3.0 * fused))
        # split the synthetic mass into ai_generated vs manipulated using the
        # localization signal: manipulated-direction survivors with localized
        # artifacts indicate splice/inpaint rather than full generation.
        lam = 0.0
        for m in survivors:
            if m.verdict_direction == "manipulated" and any(a.location for a in m.artifacts):
                lam = max(lam, abs(m.evidence_score) * m.reliability_score)
        p_syn = 1.0 - p_auth
        p_manip = p_syn * min(0.85, lam * 1.2)
        p_ai = p_syn - p_manip
        return {
            "camera_original": round(p_auth, 4),
            "ai_generated": round(p_ai, 4),
            "manipulated": round(p_manip, 4),
        }


def gate(outputs: List[ModuleOutput], d: DegradationState):
    survivors, unavailable = [], []
    for m in outputs:
        if m.unavailable_reason is not None:
            unavailable.append(UnavailableEvidence(module=m.module_id, reason=m.unavailable_reason))
        elif m.reliability_score < RELIABILITY_FLOOR:
            unavailable.append(UnavailableEvidence(
                module=m.module_id,
                reason=f"reliability {m.reliability_score:.2f} below floor {RELIABILITY_FLOOR}: "
                       f"{reliability_explanation(m.module_id, d)}",
            ))
        else:
            survivors.append(m)
    return survivors, unavailable


def conflict_metric(survivors: List[ModuleOutput]) -> float:
    """K = Σ r_i·r_j·max(0, −e_i·e_j) normalized (docs 4.4.1)."""
    num, den = 0.0, 0.0
    for i in range(len(survivors)):
        for j in range(i + 1, len(survivors)):
            a, b = survivors[i], survivors[j]
            w = a.reliability_score * b.reliability_score
            num += w * max(0.0, -a.evidence_score * b.evidence_score)
            den += w
    return round(num / den, 4) if den > 0 else 0.0


def _pattern(mi: ModuleOutput, mj: ModuleOutput) -> str:
    """Interpretation per the docs 4.4.2 pattern table."""
    syn, auth = (mi, mj) if mi.evidence_score < 0 else (mj, mi)
    if syn.module_id == "compression_history" and any(a.location for a in syn.artifacts):
        return ("Local compression-history anomaly against whole-image authentic signals: "
                "consistent with local manipulation of a real photo (splice/inpaint); "
                "the manipulated hypothesis is promoted.")
    if syn.module_id in _STAT_MODULES and auth.module_id == "metadata":
        return ("Metadata presents camera provenance while statistical evidence indicates "
                "synthesis: consistent with metadata forgery or a recycled container; "
                "risk_score raised.")
    if syn.module_id in _STAT_MODULES and auth.module_id == "compression_history":
        return ("Statistical traces indicate synthesis while compression history is uniform: "
                "consistent with a whole-image generation (uniform history is expected and "
                "does not outweigh statistical evidence).")
    if auth.module_id in _STAT_MODULES and syn.module_id == "metadata":
        return ("Metadata anomalies against statistically-normal pixels: consistent with a "
                "real photograph in a tampered or stripped container.")
    return ("Reliability-weighted disagreement without a recognized pattern: genuine "
            "ambiguity or novel attack; unexplained conflict feeds abstention.")


def build_contradictions(survivors: List[ModuleOutput]) -> List[Contradiction]:
    out = []
    den = sum(
        survivors[i].reliability_score * survivors[j].reliability_score
        for i in range(len(survivors)) for j in range(i + 1, len(survivors))
    ) + 1e-9
    for i in range(len(survivors)):
        for j in range(i + 1, len(survivors)):
            a, b = survivors[i], survivors[j]
            if a.evidence_score * b.evidence_score < -0.04 and min(abs(a.evidence_score), abs(b.evidence_score)) >= 0.2:
                contrib = a.reliability_score * b.reliability_score * max(0.0, -a.evidence_score * b.evidence_score) / den
                out.append(Contradiction(
                    modules=[a.module_id, b.module_id],
                    description=(f"{a.module_id} indicates {a.verdict_direction} (e={a.evidence_score:+.2f}, "
                                 f"r={a.reliability_score:.2f}) while {b.module_id} indicates "
                                 f"{b.verdict_direction} (e={b.evidence_score:+.2f}, r={b.reliability_score:.2f})"),
                    interpretation=_pattern(a, b),
                    conflict_contribution=round(contrib, 4),
                ))
    out.sort(key=lambda c: -c.conflict_contribution)
    return out[:5]


def conformal_set(probs: Dict[str, float], alpha: float = CONFORMAL_ALPHA) -> List[str]:
    """Conformal-stub (APS-style on calibrated probabilities): include classes
    by descending probability until cumulative mass >= 1-alpha. Heuristic
    placeholder for MAPIE split-conformal — see DECISIONS.md."""
    ordered = sorted(probs.items(), key=lambda kv: -kv[1])
    cum, chosen = 0.0, []
    for h, p in ordered:
        chosen.append(h)
        cum += p
        if cum >= 1.0 - alpha:
            break
    return chosen


def trust_score(d: DegradationState, survivors, n_registered: int, set_size: int, k: float) -> float:
    """docs 7.2: monotone combination of evidence capacity, panel coverage,
    conformal set size, unresolved conflict mass."""
    capacity = _CAPACITY_SCORE[d.evidence_capacity]
    coverage = min(1.0, sum(m.reliability_score for m in survivors) / (0.9 * max(1, n_registered)))
    set_factor = {1: 1.0, 2: 0.5}.get(set_size, 0.2)
    t = 0.35 * capacity + 0.30 * coverage + 0.20 * set_factor + 0.15 * (1.0 - k)
    return round(max(0.0, min(1.0, t)), 4)


def risk_score(d: DegradationState, outputs: List[ModuleOutput], contradictions: List[Contradiction]) -> float:
    """docs 7.2: adversarial-posture indicators."""
    r = 0.0
    if d.recompression_generations >= 3:
        r += 0.35
    elif d.recompression_generations == 2:
        r += 0.15
    if abs(d.resize_factor_est - 1.0) > 0.08 and (d.jpeg_quality_est or 100) < 60:
        r += 0.30
    flagged = 0
    for m in outputs:
        for a in m.artifacts:
            if a.type in _RISK_ARTIFACT_TYPES:
                flagged += 1
    r += min(0.45, 0.18 * flagged)
    if any("metadata forgery" in c.interpretation for c in contradictions):
        r += 0.15
    return round(max(0.0, min(1.0, r)), 4)


def rank_evidence(survivors: List[ModuleOutput]) -> List[EvidenceRankEntry]:
    den = sum(m.reliability_score for m in survivors) + 1e-9
    entries = []
    for m in survivors:
        contrib = m.reliability_score * m.evidence_score / den
        lr = round(math.exp(3.6 * abs(m.evidence_score) * m.reliability_score), 1) if abs(m.evidence_score) > 0.1 else 1.0
        entries.append(EvidenceRankEntry(
            rank=0,
            module=m.module_id,
            direction=m.verdict_direction,
            evidence_score=round(m.evidence_score, 4),
            reliability=round(m.reliability_score, 4),
            confidence=round(m.confidence_score, 4),
            shap_contribution=round(contrib, 4),
            likelihood_ratio=lr,
            artifacts=m.artifacts,
        ))
    entries.sort(key=lambda e: -abs(e.shap_contribution))
    for idx, e in enumerate(entries):
        e.rank = idx + 1
    return entries


def fuse(outputs: List[ModuleOutput], d: DegradationState, judge=None):
    judge = judge or WeightedVotingJudge()
    survivors, unavailable = gate(outputs, d)
    probs = judge.decide(survivors)
    k = conflict_metric(survivors)
    contradictions = build_contradictions(survivors)
    cset = conformal_set(probs) if survivors else list(HYPOTHESES)
    abstained = len(cset) > 1
    t = trust_score(d, survivors, len(outputs), len(cset), k)
    r = risk_score(d, outputs, contradictions)
    core = VerdictCore(
        hypothesis_set=cset,
        abstained=abstained,
        probabilities=probs,
        authenticity_score=probs["camera_original"],
        trust_score=t,
        risk_score=r,
        conformal=ConformalInfo(alpha=CONFORMAL_ALPHA, set=cset, calibration_stratum=_STRATUM[d.evidence_capacity]),
    )
    ranking = rank_evidence(survivors)
    return core, ranking, contradictions, unavailable, k
