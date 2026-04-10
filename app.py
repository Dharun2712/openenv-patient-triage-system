from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from triage_env.environment import AIHospitalTriageEnv

app = FastAPI(title="OpenEnv Triage API", version="1.0.0")
env = AIHospitalTriageEnv(task_name="task_easy")


class ResetRequest(BaseModel):
    task_name: str = "task_easy"


class StepRequest(BaseModel):
    action: str


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None) -> dict[str, object]:
    try:
        task_name = req.task_name if req is not None else "task_easy"
        obs = env.reset(task_name=task_name)
        return {"observation": obs.model_dump()}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step")
def step(req: StepRequest) -> dict[str, object]:
    try:
        obs, reward, done, info = env.step(req.action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))
