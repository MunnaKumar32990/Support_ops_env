"""
Grader for Task 1 (Easy): Email Classification

Scoring:
  - +1.0 for exact correct label
  - +0.5 for semantically close label (billing ↔ general, technical ↔ general)
  - +0.0 for wrong label
  - -0.2 penalty for invalid label (not in valid set)
"""

from tasks.task_easy import VALID_LABELS, SEMANTIC_GROUPS


# Which pairs are "close" (partial credit)
CLOSE_PAIRS = {
    ("billing", "general"),
    ("general", "billing"),
    ("technical", "general"),
    ("general", "technical"),
}


def grade(predicted_label: str, ground_truth_label: str) -> float:
    """
    Grade the email classification prediction.

    Args:
        predicted_label: The label predicted by the agent (string).
        ground_truth_label: The correct label from the dataset.

    Returns:
        float in [−0.2, 1.0]
    """
    # Normalize
    predicted_label = str(predicted_label).strip().lower()
    ground_truth_label = str(ground_truth_label).strip().lower()

    # Invalid label penalty
    if predicted_label not in VALID_LABELS:
        return -0.2

    # Exact match
    if predicted_label == ground_truth_label:
        return 1.0

    # Semantic proximity (partial credit)
    if (predicted_label, ground_truth_label) in CLOSE_PAIRS:
        return 0.5

    # Wrong
    return 0.0


def grade_batch(predictions: list[str], samples: list[dict]) -> dict:
    """
    Grade a batch of predictions.

    Args:
        predictions: List of predicted labels.
        samples: List of sample dicts (must contain 'label' key).

    Returns:
        dict with 'scores', 'mean_score', 'accuracy'
    """
    assert len(predictions) == len(samples), "Mismatch between predictions and samples."

    scores = []
    exact_correct = 0

    for pred, sample in zip(predictions, samples):
        gt = sample["label"]
        score = grade(pred, gt)
        scores.append(score)
        if str(pred).strip().lower() == gt:
            exact_correct += 1

    mean_score = sum(scores) / len(scores) if scores else 0.0
    accuracy = exact_correct / len(scores) if scores else 0.0

    return {
        "scores": scores,
        "mean_score": round(mean_score, 4),
        "accuracy": round(accuracy, 4),
        "num_samples": len(scores),
    }
