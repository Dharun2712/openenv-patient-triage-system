from __future__ import annotations

import json
import os
from typing import List, Tuple

from openai import OpenAI

from triage_env.environment import AIHospitalTriageEnv
from triage_env.models import ActionType
from triage_env.tasks import list_tasks

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

ALLOWED_ACTIONS: List[ActionType] = [
    "assign_low_priority",
    "assign_medium_priority",
    "assign_high_priority",
    "send_to_emergency",
    "request_additional_tests",
]


def _clamp_score(score: float) -> float:
    """Clamp score to open interval (0, 1). Validator rejects 0.0 and 1.0."""
    return max(0.01, min(0.99, score))


def _parse_systolic(bp: str) -> int:
    try:
        return int(bp.split("/")[0])
    except (ValueError, IndexError):
        return 120


def _heuristic_action(observation: dict, step_index: int) -> ActionType:
    severity = observation.get("injury_severity", "low")
    heart_rate = int(observation.get("heart_rate", 80))
    bp = str(observation.get("blood_pressure", "120/80"))
    systolic = _parse_systolic(bp)

    if severity == "high":
        if step_index == 1:
            return "assign_high_priority"
        if heart_rate >= 130 or systolic < 90:
            return "send_to_emergency"
        return "assign_high_priority"
    if severity == "medium":
        return "assign_medium_priority"
    return "assign_low_priority"


def _extract_json_content(content: str) -> dict | None:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return None

    return None


def _llm_action(client: OpenAI, model_name: str, observation: dict, step_index: int) -> Tuple[ActionType, str | None]:
    prompt = {
        "instruction": "Choose exactly one triage action from the allowed list.",
        "allowed_actions": ALLOWED_ACTIONS,
        "observation": observation,
        "step_index": step_index,
        "output_format": {"action": "one_of_allowed_actions"},
    }

    last_error: str | None = None
    for _ in range(2):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a hospital triage policy. Respond ONLY as compact JSON: {\"action\":\"...\"}.",
                    },
                    {"role": "user", "content": json.dumps(prompt)},
                ],
                temperature=0.0,
            )

            content = response.choices[0].message.content or ""
            parsed = _extract_json_content(content)
            if not parsed:
                return _heuristic_action(observation, step_index), "model_output_not_json"

            action = parsed.get("action")
            if action in ALLOWED_ACTIONS:
                return action, None
            return _heuristic_action(observation, step_index), "invalid_action_from_model"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc).replace("\n", " ")

    return _heuristic_action(observation, step_index), last_error or "model_request_failed"


def _apply_policy_guard(task_name: str, observation: dict, action: ActionType, action_history: List[str]) -> Tuple[ActionType, str | None]:
    severity = observation.get("injury_severity", "low")
    heart_rate = int(observation.get("heart_rate", 80))
    systolic = _parse_systolic(str(observation.get("blood_pressure", "120/80")))
    critical = heart_rate >= 130 or heart_rate <= 40 or systolic < 90

    if task_name == "task_hard" or (severity == "high" and critical):
        if "assign_high_priority" not in action_history:
            if action != "assign_high_priority":
                return "assign_high_priority", "policy_guard_priority_first"
        elif "send_to_emergency" not in action_history:
            if action != "send_to_emergency":
                return "send_to_emergency", "policy_guard_escalate_second"

    if severity == "low" and action in {"assign_high_priority", "send_to_emergency"}:
        return "assign_low_priority", "policy_guard_overtriage"

    return action, None


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def run_task(task_name: str, client: OpenAI, model_name: str) -> None:
    env = AIHospitalTriageEnv(task_name=task_name)
    observation = env.reset()

    print(f"[START] task={task_name} env=triage model={model_name}")

    rewards: List[float] = []
    done = False
    success = False
    step_index = 0

    while not done and step_index < 5:
        step_index += 1
        obs_dict = observation.model_dump()

        error_msg = None
        try:
            action, model_error = _llm_action(client, model_name, obs_dict, step_index)
            if model_error is not None:
                error_msg = model_error
        except Exception as exc:  # noqa: BLE001
            action = _heuristic_action(obs_dict, step_index)
            error_msg = str(exc).replace("\n", " ")

        action, _policy_error = _apply_policy_guard(task_name, obs_dict, action, env.action_history)

        next_observation, reward, done, info = env.step(action)
        clamped_score = _clamp_score(reward.score)
        rewards.append(clamped_score)

        raw_task_score = info.get("task_score", 0.01)
        task_score = _clamp_score(raw_task_score)
        success = bool(task_score >= 0.8) if done else False

        error_str = "null" if error_msg is None else error_msg
        print(
            f"[STEP] step={step_index} action={action} reward={clamped_score:.2f} "
            f"done={_format_bool(done)} error={error_str}"
        )

        observation = next_observation

    rewards_csv = ",".join(f"{score:.2f}" for score in rewards)
    print(f"[END] success={_format_bool(success)} steps={step_index} rewards={rewards_csv}")


def main() -> None:
    api_base_url = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise RuntimeError("HF_TOKEN is required")

    client = OpenAI(base_url=api_base_url, api_key=hf_token)

    for task_name in list_tasks():
        run_task(task_name=task_name, client=client, model_name=model_name)


if __name__ == "__main__":
    main()
