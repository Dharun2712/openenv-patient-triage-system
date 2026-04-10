from __future__ import annotations

from itertools import product

from triage_env.environment import AIHospitalTriageEnv
from triage_env.graders import grade_task
from triage_env.tasks import TASKS

ACTIONS = [
    "assign_low_priority",
    "assign_medium_priority",
    "assign_high_priority",
    "send_to_emergency",
    "request_additional_tests",
]


def check_grader_scores() -> None:
    for task_name, task in TASKS.items():
        for length in range(1, task.max_steps + 1):
            for sequence in product(ACTIONS, repeat=length):
                score = grade_task(task_name, list(sequence), task.max_steps)
                assert 0.0 < score < 1.0, (
                    f"grader out-of-range task={task_name} seq={sequence} score={score}"
                )


def check_environment_scores() -> None:
    for task_name, task in TASKS.items():
        for length in range(1, task.max_steps + 1):
            for sequence in product(ACTIONS, repeat=length):
                env = AIHospitalTriageEnv(task_name=task_name)
                env.reset()
                done = False
                last_info = {}

                for action in sequence:
                    _, reward, done, info = env.step(action)
                    assert 0.0 < reward.score < 1.0, (
                        f"reward out-of-range task={task_name} seq={sequence} score={reward.score}"
                    )
                    last_info = info
                    if done:
                        break

                if done and "task_score" in last_info:
                    task_score = float(last_info["task_score"])
                    assert 0.0 < task_score < 1.0, (
                        f"task_score out-of-range task={task_name} seq={sequence} score={task_score}"
                    )


def main() -> None:
    check_grader_scores()
    check_environment_scores()
    print("strict_check passed")


if __name__ == "__main__":
    main()
