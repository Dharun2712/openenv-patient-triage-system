from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from triage_env.models import Observation


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    difficulty: str
    description: str
    observation: Observation
    expected_actions: List[List[str]]
    max_steps: int


TASKS: Dict[str, TaskDefinition] = {
    "task_easy": TaskDefinition(
        name="task_easy",
        difficulty="easy",
        description="Low severity patient requiring low-priority assignment.",
        observation=Observation(
            patient_id=1001,
            symptoms="Mild cough and sore throat",
            heart_rate=78,
            blood_pressure="120/80",
            injury_severity="low",
            waiting_time=10,
        ),
        expected_actions=[["assign_low_priority"]],
        max_steps=2,
    ),
    "task_medium": TaskDefinition(
        name="task_medium",
        difficulty="medium",
        description="Medium severity patient where medium priority or additional tests are acceptable.",
        observation=Observation(
            patient_id=2002,
            symptoms="Persistent abdominal pain",
            heart_rate=96,
            blood_pressure="132/86",
            injury_severity="medium",
            waiting_time=35,
        ),
        expected_actions=[["assign_medium_priority", "request_additional_tests"]],
        max_steps=2,
    ),
    "task_hard": TaskDefinition(
        name="task_hard",
        difficulty="hard",
        description="High severity patient with critical vitals requiring high priority and emergency escalation.",
        observation=Observation(
            patient_id=3003,
            symptoms="Acute chest pain, shortness of breath, dizziness",
            heart_rate=145,
            blood_pressure="85/55",
            injury_severity="high",
            waiting_time=75,
        ),
        expected_actions=[["assign_high_priority"], ["send_to_emergency"]],
        max_steps=3,
    ),
}


def get_task(task_name: str) -> TaskDefinition:
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}")
    return TASKS[task_name]


def list_tasks() -> List[str]:
    return list(TASKS.keys())
