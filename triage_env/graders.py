from __future__ import annotations

from typing import List


def _efficiency_score(steps_taken: int, ideal_steps: int) -> float:
    if steps_taken <= 0:
        return 0.01
    return min(1.0, ideal_steps / float(steps_taken))


def grade_task(task_name: str, action_history: List[str], max_steps: int) -> float:
    """Deterministic score strictly in (0.0, 1.0) based on correctness and efficiency."""
    steps_taken = min(len(action_history), max_steps)

    if task_name == "task_easy":
        correctness = 1.0 if action_history[:1] == ["assign_low_priority"] else 0.0
        efficiency = _efficiency_score(steps_taken, ideal_steps=1)
    elif task_name == "task_medium":
        allowed = {"assign_medium_priority", "request_additional_tests"}
        correctness = 1.0 if len(action_history) >= 1 and action_history[0] in allowed else 0.0
        efficiency = _efficiency_score(steps_taken, ideal_steps=1)
    elif task_name == "task_hard":
        has_priority = "assign_high_priority" in action_history
        has_emergency = "send_to_emergency" in action_history
        correctness = (float(has_priority) + float(has_emergency)) / 2.0

        if has_priority and has_emergency:
            priority_idx = action_history.index("assign_high_priority")
            emergency_idx = action_history.index("send_to_emergency")
            if priority_idx <= emergency_idx:
                correctness = 1.0
            else:
                correctness = 0.75

        efficiency = _efficiency_score(steps_taken, ideal_steps=2)
    else:
        raise ValueError(f"Unknown task name: {task_name}")

    score = (0.8 * correctness) + (0.2 * efficiency)
    score = max(0.01, min(0.99, float(score)))
    assert 0.0 < score < 1.0, f"Invalid score: {score}"
    return round(score, 2)
