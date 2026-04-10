from __future__ import annotations

from copy import deepcopy
from typing import Dict, Tuple

from triage_env.graders import grade_task
from triage_env.models import Action, Observation, Reward
from triage_env.tasks import TaskDefinition, get_task


SCORE_EPSILON = 0.01


class AIHospitalTriageEnv:
    """OpenEnv-style environment for hospital patient triage decisions."""

    env_id = "triage"

    def __init__(self, task_name: str = "task_easy") -> None:
        self.task_name = task_name
        self.task: TaskDefinition = get_task(task_name)
        self.current_observation: Observation = deepcopy(self.task.observation)
        self.step_count = 0
        self.progress_index = 0
        self.done = False
        self.action_history = []

    def reset(self, task_name: str | None = None) -> Observation:
        if task_name is not None:
            self.task_name = task_name
        self.task = get_task(self.task_name)
        self.current_observation = deepcopy(self.task.observation)
        self.step_count = 0
        self.progress_index = 0
        self.done = False
        self.action_history = []
        return self.current_observation

    def state(self) -> Observation:
        return self.current_observation

    def _is_critical_vitals(self, observation: Observation) -> bool:
        systolic = 120
        try:
            systolic = int(observation.blood_pressure.split("/")[0])
        except (ValueError, IndexError):
            pass

        return observation.heart_rate >= 130 or observation.heart_rate <= 40 or systolic < 90

    def _priority_match(self, action: str, severity: str) -> bool:
        mapping = {
            "low": {"assign_low_priority"},
            "medium": {"assign_medium_priority", "request_additional_tests"},
            "high": {"assign_high_priority"},
        }
        return action in mapping[severity]

    def _strict_score(self, score: float) -> float:
        bounded = max(0.0, min(1.0, score))
        if bounded <= 0.0:
            return SCORE_EPSILON
        if bounded >= 1.0:
            return 1.0 - SCORE_EPSILON
        return round(bounded, 2)

    def step(self, action: Action | str) -> Tuple[Observation, Reward, bool, Dict[str, object]]:
        if self.done:
            reward = Reward(
                score=0.0,
                reason="Episode already completed.",
                priority_correct=False,
                emergency_handling_correct=False,
                delay_penalty_applied=False,
                wrong_decision_penalty_applied=False,
            )
            return self.current_observation, reward, True, {
                "task_name": self.task_name,
                "error": "episode_done",
            }

        parsed_action = action if isinstance(action, Action) else Action(action=action)
        action_name = parsed_action.action

        expected_now = set(self.task.expected_actions[self.progress_index])
        is_expected_action = action_name in expected_now

        priority_correct = self._priority_match(action_name, self.current_observation.injury_severity)
        emergency_required = (
            self.current_observation.injury_severity == "high"
            and self._is_critical_vitals(self.current_observation)
        )
        emergency_handling_correct = (
            action_name == "send_to_emergency" if emergency_required else action_name != "send_to_emergency"
        )

        delay_penalty_applied = self.current_observation.waiting_time >= 60
        wrong_decision_penalty_applied = not is_expected_action

        score = 0.0
        if priority_correct:
            score += 0.5
        if emergency_handling_correct:
            score += 0.5
        if delay_penalty_applied:
            score -= 0.2
        if wrong_decision_penalty_applied:
            score -= 0.5
        score = self._strict_score(score)

        reason_parts = []
        if priority_correct:
            reason_parts.append("correct_priority")
        if emergency_handling_correct:
            reason_parts.append("correct_emergency_handling")
        if delay_penalty_applied:
            reason_parts.append("delay_penalty")
        if wrong_decision_penalty_applied:
            reason_parts.append("wrong_decision_penalty")

        reward = Reward(
            score=score,
            reason=",".join(reason_parts) if reason_parts else "no_reward_components",
            priority_correct=priority_correct,
            emergency_handling_correct=emergency_handling_correct,
            delay_penalty_applied=delay_penalty_applied,
            wrong_decision_penalty_applied=wrong_decision_penalty_applied,
        )

        self.step_count += 1
        self.action_history.append(action_name)

        if is_expected_action and self.progress_index < len(self.task.expected_actions):
            self.progress_index += 1

        self.current_observation.waiting_time += 5

        if self.progress_index >= len(self.task.expected_actions):
            self.done = True
        elif self.step_count >= self.task.max_steps:
            self.done = True

        info: Dict[str, object] = {
            "task_name": self.task_name,
            "difficulty": self.task.difficulty,
            "progress": f"{self.progress_index}/{len(self.task.expected_actions)}",
            "expected_actions_current": []
            if self.done
            else self.task.expected_actions[self.progress_index],
        }

        if self.done:
            info["task_score"] = grade_task(
                task_name=self.task_name,
                action_history=self.action_history,
                max_steps=self.task.max_steps,
            )

        return self.current_observation, reward, self.done, info
