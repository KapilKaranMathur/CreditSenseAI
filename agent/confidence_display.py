import logging
from typing import Dict

logger = logging.getLogger(__name__)


def compute_confidence_breakdown(decision: Dict, risk: Dict) -> Dict:
    from agent.llm_reasoner import BORDERLINE_RANGE, _normalize_decision_payload

    normalized = _normalize_decision_payload(decision or {})

    try:
        llm_confidence = float(normalized.get("Confidence", 0.5))
    except (TypeError, ValueError):
        llm_confidence = 0.5
    llm_confidence = max(0.0, min(llm_confidence, 1.0))

    signal_1 = {
        "name": "LLM Self-Reported Confidence",
        "score": round(llm_confidence, 4),
        "weight": 0.40,
        "weighted_score": round(0.40 * llm_confidence, 4),
        "description": f"The LLM rated its own confidence at {llm_confidence:.0%}",
    }

    probability = float(risk.get("probability", 0.5))
    llm_decision = normalized.get("Lending Decision", "REJECT")

    if llm_decision == "REJECT" and probability >= BORDERLINE_RANGE[1]:
        alignment, alignment_desc = 1.0, "Strong agreement — both ML and LLM indicate high risk"
    elif llm_decision == "APPROVE" and probability <= BORDERLINE_RANGE[0]:
        alignment, alignment_desc = 1.0, "Strong agreement — both ML and LLM indicate low risk"
    elif llm_decision == "CONDITIONAL" and BORDERLINE_RANGE[0] < probability < BORDERLINE_RANGE[1]:
        alignment, alignment_desc = 1.0, "Agreement — both recognize borderline risk"
    elif llm_decision == "CONDITIONAL":
        alignment, alignment_desc = 0.5, "Partial — LLM is cautious despite ML clarity"
    else:
        alignment, alignment_desc = 0.2, "Conflict — LLM disagrees with ML risk assessment"

    signal_2 = {
        "name": "ML-LLM Decision Alignment",
        "score": round(alignment, 4),
        "weight": 0.35,
        "weighted_score": round(0.35 * alignment, 4),
        "description": alignment_desc,
    }

    low, high = BORDERLINE_RANGE
    if probability < low or probability > high:
        clarity = 1.0
        clarity_desc = f"Clear-cut case — probability {probability:.0%} is well outside the borderline zone ({low:.0%}–{high:.0%})"
    else:
        center = (low + high) / 2
        distance = abs(probability - center) / (center - low)
        clarity = min(distance, 1.0)
        clarity_desc = f"Ambiguous — probability {probability:.0%} falls in the borderline zone ({low:.0%}–{high:.0%})"

    signal_3 = {
        "name": "Risk Clarity",
        "score": round(clarity, 4),
        "weight": 0.25,
        "weighted_score": round(0.25 * clarity, 4),
        "description": clarity_desc,
    }

    signals = [signal_1, signal_2, signal_3]
    overall = round(sum(s["weighted_score"] for s in signals), 4)
    overall = max(0.0, min(overall, 1.0))

    if overall >= 0.75:
        label, color = "High", "#22c55e"
    elif overall >= 0.50:
        label, color = "Medium", "#f59e0b"
    else:
        label, color = "Low", "#ef4444"

    return {
        "overall_score": overall,
        "overall_label": label,
        "overall_color": color,
        "signals": signals,
    }
