"""
Grader for Task 3 (Hard): Response Generation

Scoring breakdown (total max = 1.0):
  - Structure correctness   : 0.30
      * 'tone' is valid                : 0.10
      * 'resolution_steps' non-empty   : 0.10
      * 'escalation' is boolean        : 0.10
  - Tone correctness         : 0.20
      * Predicted tone matches reference: 0.20 (0.10 if in valid set but wrong)
  - Escalation correctness   : 0.20
      * Matches reference boolean      : 0.20
  - Semantic similarity      : 0.30
      * Keyword overlap between predicted resolution and reference: 0.30

  Penalties (applied AFTER raw score):
  - unsafe/irrelevant content detected: -0.30
  - resolution_steps too short (<20 chars): -0.10
"""

import re
from tasks.task_hard import VALID_TONES, UNSAFE_KEYWORDS


# ─────────────────────────────────────────────
# Keyword extraction helper
# ─────────────────────────────────────────────
def _extract_keywords(text: str) -> set:
    """Simple keyword extractor: lowercase alpha tokens, length ≥ 4."""
    tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
    stopwords = {
        "this", "that", "with", "from", "will", "have", "your", "their",
        "they", "them", "been", "were", "when", "also", "some", "more",
        "into", "very", "just", "make", "does", "done", "step", "steps",
        "customer", "provide", "offer", "send", "take", "need",
    }
    return {t for t in tokens if t not in stopwords}


def _keyword_overlap_score(predicted_text: str, reference_text: str) -> float:
    """Score based on keyword overlap between predicted and reference text."""
    pred_kw = _extract_keywords(predicted_text)
    ref_kw = _extract_keywords(reference_text)

    if not ref_kw:
        return 0.5  # neutral if no reference keywords

    overlap = pred_kw & ref_kw
    precision = len(overlap) / len(pred_kw) if pred_kw else 0.0
    recall = len(overlap) / len(ref_kw) if ref_kw else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return round(min(1.0, f1), 4)


def _has_unsafe_content(text: str) -> bool:
    """Return True if any unsafe keyword appears in the text."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in UNSAFE_KEYWORDS)


def grade(
    predicted_action: dict,
    ground_truth: dict,
) -> float:
    """
    Grade the response generation prediction.

    Args:
        predicted_action: dict with keys 'tone', 'resolution_steps', 'escalation'
        ground_truth: reference dict with same keys + 'valid_tones', 'requires_escalation'

    Returns:
        float in [0.0, 1.0] (after penalties, may be slightly below 0; clamped to 0)
    """
    score = 0.0
    ref_tone = ground_truth.get("tone", "")
    ref_steps = ground_truth.get("resolution_steps", "")
    ref_escalation = ground_truth.get("escalation", False)

    # ── 1. Structure correctness (0.30) ──────────────────────────
    # tone field exists and is valid
    predicted_tone = str(predicted_action.get("tone", "")).strip().lower()
    if predicted_tone in VALID_TONES:
        score += 0.10

    # resolution_steps exists and non-empty
    resolution_steps = str(predicted_action.get("resolution_steps", "")).strip()
    if resolution_steps:
        score += 0.10

    # escalation is bool-like
    escalation_val = predicted_action.get("escalation", None)
    if isinstance(escalation_val, bool):
        score += 0.10

    # ── 2. Tone correctness (0.20) ────────────────────────────────
    if predicted_tone == ref_tone:
        score += 0.20
    elif predicted_tone in VALID_TONES:
        score += 0.10  # partial credit for valid but wrong tone

    # ── 3. Escalation correctness (0.20) ─────────────────────────
    if isinstance(escalation_val, bool) and escalation_val == ref_escalation:
        score += 0.20

    # ── 4. Semantic similarity of resolution steps (0.30) ─────────
    if resolution_steps and ref_steps:
        similarity = _keyword_overlap_score(resolution_steps, ref_steps)
        score += similarity * 0.30

    # ── Penalties ─────────────────────────────────────────────────
    full_text = predicted_tone + " " + resolution_steps
    if _has_unsafe_content(full_text):
        score -= 0.30

    if len(resolution_steps) < 20:
        score -= 0.10

    # Clamp to [0.0, 1.0]
    return round(max(0.0, min(1.0, score)), 4)


def grade_batch(predictions: list[dict], samples: list[dict]) -> dict:
    """
    Grade a batch of response generation predictions.

    Args:
        predictions: List of dicts with 'tone', 'resolution_steps', 'escalation'.
        samples: List of sample dicts with reference_response.

    Returns:
        dict with 'scores', 'mean_score', 'structure_rate', 'escalation_accuracy'
    """
    assert len(predictions) == len(samples), "Mismatch between predictions and samples."

    scores = []
    structure_ok = 0
    escalation_correct = 0

    for pred, sample in zip(predictions, samples):
        ref = sample["reference_response"]
        ref["requires_escalation"] = sample["requires_escalation"]

        s = grade(pred, ref)
        scores.append(s)

        # Structure check
        tone_ok = str(pred.get("tone", "")).strip().lower() in VALID_TONES
        steps_ok = bool(str(pred.get("resolution_steps", "")).strip())
        esc_bool = isinstance(pred.get("escalation"), bool)
        if tone_ok and steps_ok and esc_bool:
            structure_ok += 1

        # Escalation check
        if isinstance(pred.get("escalation"), bool):
            if pred["escalation"] == sample["requires_escalation"]:
                escalation_correct += 1

    mean_score = sum(scores) / len(scores) if scores else 0.0
    structure_rate = structure_ok / len(scores) if scores else 0.0
    escalation_accuracy = escalation_correct / len(scores) if scores else 0.0

    return {
        "scores": scores,
        "mean_score": round(mean_score, 4),
        "structure_rate": round(structure_rate, 4),
        "escalation_accuracy": round(escalation_accuracy, 4),
        "num_samples": len(scores),
    }
