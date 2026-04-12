"""
Grader for Task 2 (Medium): Ticket Prioritization

Scoring:
  - Proportional to 'distance' between predicted and true priority
  - scale: low=0, medium=1, high=2
  - Score = 1 - (|pred_score - gt_score| / max_distance)
  - Additional penalty for underestimating high-priority issues:
      if gt == 'high' and pred != 'high': extra -0.3 penalty
  - Penalty for invalid priority: -0.2
"""

from tasks.task_medium import VALID_PRIORITIES, PRIORITY_SCORE


MAX_DISTANCE = 2  # distance between 'low' (0) and 'high' (2)
UNDERESTIMATE_PENALTY = 0.3  # penalty for missing high-priority tickets


def grade(predicted_priority: str, ground_truth_priority: str) -> float:
    """
    Grade the ticket prioritization prediction.

    Args:
        predicted_priority: The priority predicted by the agent.
        ground_truth_priority: The correct priority from the dataset.

    Returns:
        float in [−0.5, 1.0]  (clamped to [-0.5, 1.0])
    """
    predicted_priority = str(predicted_priority).strip().lower()
    ground_truth_priority = str(ground_truth_priority).strip().lower()

    # Invalid label penalty
    if predicted_priority not in VALID_PRIORITIES:
        return -0.2

    pred_val = PRIORITY_SCORE[predicted_priority]
    gt_val = PRIORITY_SCORE[ground_truth_priority]

    distance = abs(pred_val - gt_val)
    base_score = 1.0 - (distance / MAX_DISTANCE)

    # Penalty for underestimating high-priority tickets
    if ground_truth_priority == "high" and predicted_priority != "high":
        base_score -= UNDERESTIMATE_PENALTY

    # Clamp to [0.001, 0.999] for strict limits
    return round(max(0.001, min(0.999, base_score)), 4)


def grade_batch(predictions: list[str], samples: list[dict]) -> dict:
    """
    Grade a batch of prioritization predictions.

    Args:
        predictions: List of predicted priorities.
        samples: List of sample dicts (must contain 'priority' key).

    Returns:
        dict with 'scores', 'mean_score', 'accuracy',
               'high_priority_recall'
    """
    assert len(predictions) == len(samples), "Mismatch between predictions and samples."

    scores = []
    exact_correct = 0
    high_priority_samples = 0
    high_priority_correct = 0

    for pred, sample in zip(predictions, samples):
        gt = sample["priority"]
        score = grade(pred, gt)
        scores.append(score)

        pred_norm = str(pred).strip().lower()
        if pred_norm == gt:
            exact_correct += 1
        if gt == "high":
            high_priority_samples += 1
            if pred_norm == "high":
                high_priority_correct += 1

    mean_score = sum(scores) / len(scores) if scores else 0.0
    accuracy = exact_correct / len(scores) if scores else 0.0
    high_recall = (
        high_priority_correct / high_priority_samples
        if high_priority_samples > 0
        else 1.0
    )

    return {
        "scores": scores,
        "mean_score": round(mean_score, 4),
        "accuracy": round(accuracy, 4),
        "high_priority_recall": round(high_recall, 4),
        "num_samples": len(scores),
    }
