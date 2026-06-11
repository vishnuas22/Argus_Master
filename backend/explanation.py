"""Template-based explanation generator (docs 7.3).

Hard rule: no free text generation. Template skeletons per verdict pattern,
slot-filled EXCLUSIVELY from verdict JSON fields. Every sentence traces 1:1
to schema fields; checkable_claim strings are the only factual content.
"""
from typing import List

from schemas import (
    Contradiction,
    DegradationState,
    EvidenceRankEntry,
    Explanation,
    UnavailableEvidence,
    VerdictCore,
)

_LABEL = {
    "camera_original": "camera-original",
    "ai_generated": "AI-generated",
    "manipulated": "manipulated",
}


def _band(x: float) -> str:
    if x >= 0.75:
        return "high"
    if x >= 0.5:
        return "moderate"
    if x >= 0.25:
        return "low"
    return "very low"


def _pattern(core: VerdictCore, d: DegradationState, k: float) -> str:
    top = max(core.probabilities, key=core.probabilities.get)
    if core.abstained and d.evidence_capacity == "LOW":
        return "abstain-degraded"
    if core.abstained and k > 0.35:
        return "abstain-conflict"
    if core.abstained:
        return "abstain-uncertain"
    if top == "manipulated":
        return "manipulated-localized"
    if top == "ai_generated":
        return "decisive-synthetic"
    return "decisive-authentic"


def _degradation_sentence(d: DegradationState) -> str:
    parts = []
    if d.recompression_generations >= 2:
        parts.append(f"the image shows {d.recompression_generations} recompression generations")
    elif d.recompression_generations == 1:
        parts.append("the image shows a single compression generation")
    else:
        parts.append("the image shows no lossy compression history")
    if abs(d.resize_factor_est - 1.0) > 0.08:
        parts.append(f"resampling by a factor of ~{d.resize_factor_est:.2f} was detected")
    if d.jpeg_quality_est is not None:
        parts.append(f"estimated JPEG quality {d.jpeg_quality_est}")
    if d.screenshot_probability >= 0.6:
        parts.append(f"screenshot probability {d.screenshot_probability:.2f}")
    return (", ".join(parts).capitalize()
            + f"; evidence capacity: {d.evidence_capacity}.")


def generate(core: VerdictCore, d: DegradationState, ranking: List[EvidenceRankEntry],
             contradictions: List[Contradiction], unavailable: List[UnavailableEvidence],
             k: float) -> Explanation:
    pattern = _pattern(core, d, k)
    top = max(core.probabilities, key=core.probabilities.get)
    set_str = "{" + ", ".join(_LABEL[h] for h in core.conformal.set) + "}"

    if core.abstained:
        head = f"Assessment: ABSTAIN — conformal {1 - core.conformal.alpha:.0%} set: {set_str}"
    else:
        head = (f"Assessment: {_LABEL[top]} ({core.probabilities[top]:.0%} · "
                f"conformal {1 - core.conformal.alpha:.0%} set: {set_str})")
    summary = (f"{head} · Trust: {_band(core.trust_score)} ({core.trust_score:.2f}) · "
               f"Gaming risk: {_band(core.risk_score)} ({core.risk_score:.2f}).")

    sentences = [_degradation_sentence(d)]

    n = len(ranking)
    n_syn = sum(1 for e in ranking if e.evidence_score < -0.1)
    n_auth = sum(1 for e in ranking if e.evidence_score > 0.1)
    if n:
        sentences.append(f"Of the {n} available evidence streams, {n_syn} indicate synthesis or "
                         f"manipulation and {n_auth} indicate camera-origin.")
    else:
        sentences.append("No evidence stream survived the reliability gate.")

    if ranking and ranking[0].artifacts:
        top_e = ranking[0]
        sentences.append(f"Strongest evidence — {top_e.module} (LR ≈ {top_e.likelihood_ratio}): "
                         f"{top_e.artifacts[0].checkable_claim}")

    if pattern == "abstain-degraded":
        sentences.append("Insufficient evidence: the image is too degraded for a reliable "
                         "assessment; the verdict is an abstention, not a classification.")
    elif pattern == "abstain-conflict":
        sentences.append(f"Reliable evidence streams disagree (conflict mass K={k:.2f}); "
                         "the verdict abstains rather than averaging the conflict away.")

    if contradictions:
        c = contradictions[0]
        sentences.append(f"Contradiction noted: {c.description}. {c.interpretation}")

    if unavailable:
        listed = "; ".join(f"{u.module} ({u.reason})" for u in unavailable)
        sentences.append(f"Evidence unavailable: {listed}.")

    if core.risk_score >= 0.4:
        sentences.append(f"Gaming-risk indicators are elevated (risk {core.risk_score:.2f}): "
                         "see flagged artifacts in the evidence ranking.")

    return Explanation(summary=summary, detail=" ".join(sentences), audience="analyst")
