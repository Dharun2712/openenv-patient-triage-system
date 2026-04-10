---
title: OpenEnv Patient Triage System
emoji: "🏥"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# AI Hospital Patient Triage Environment

OpenEnv-compliant reinforcement learning environment that simulates real-world hospital triage decisions.

An agent receives patient state and must choose the right triage action under safety and urgency constraints.

## Highlights

- OpenEnv-style environment methods: `reset()`, `step(action)`, `state()`
- Pydantic typed models for observation, action, reward
- Deterministic tasks and graders (`easy`, `medium`, `hard`)
- Strict `inference.py` logging format for validator checks
- FastAPI service for API validation and deployment
- Dockerized runtime for local and HF Spaces deployment

## OpenEnv Compliance

- Environment class: `triage_env/environment.py`
- Metadata file: `openenv.yaml`
- Typed models: `triage_env/models.py`
- Task definitions: `triage_env/tasks.py`
- Deterministic graders: `triage_env/graders.py`

## Observation Space

Each observation includes:

- `patient_id` (int)
- `symptoms` (string)
- `heart_rate` (int)
- `blood_pressure` (string)
- `injury_severity` (`low` | `medium` | `high`)
- `waiting_time` (minutes)

## Action Space

Allowed actions:

- `assign_low_priority`
- `assign_medium_priority`
- `assign_high_priority`
- `send_to_emergency`
- `request_additional_tests`

## Reward Design

Per-step reward logic:

- Correct priority: `+0.5`
- Correct emergency handling: `+0.5`
- Delay penalty: `-0.2`
- Wrong decision penalty: `-0.5`

Final reward is clamped to `[0.0, 1.0]`.

## Tasks

1. `task_easy`
Low severity patient; expected action: `assign_low_priority`.

2. `task_medium`
Medium severity patient; expected action: `assign_medium_priority` or `request_additional_tests`.

3. `task_hard`
High severity patient with critical vitals; expected sequence: `assign_high_priority` then `send_to_emergency`.

## Grading

Graders return deterministic score in `[0.0, 1.0]` based on:

- correctness
- steps taken
- efficiency

## Project Structure

```text
.
├── app.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── README.md
└── triage_env
    ├── __init__.py
    ├── environment.py
    ├── graders.py
    ├── models.py
    └── tasks.py
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create `.env` (recommended):

```dotenv
HF_TOKEN=your_huggingface_token
MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
API_BASE_URL=https://router.huggingface.co/v1
```

PowerShell one-liner to load `.env` and run inference:

```powershell
Get-Content .env | ForEach-Object { if ([string]::IsNullOrWhiteSpace($_) -or $_.Trim().StartsWith('#')) { return }; $parts = $_ -split '=',2; [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1], 'Process') }
python inference.py
```

## Inference Runner (`inference.py`)

### Required Variable Handling

- `API_BASE_URL = os.getenv("API_BASE_URL", default)`
- `MODEL_NAME = os.getenv("MODEL_NAME", default)`
- `HF_TOKEN = os.getenv("HF_TOKEN")`

### Strict Output Format

```text
[START] task=<task_name> env=triage model=<model_name>
[STEP] step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...>
```

Guarantees:

- reward formatted to 2 decimals
- `done` and `success` are lowercase booleans
- `error` prints `null` when no error exists

### Example Output

```text
[START] task=task_easy env=triage model=meta-llama/Llama-3.3-70B-Instruct
[STEP] step=1 action=assign_low_priority reward=1.00 done=true error=null
[END] success=true steps=1 rewards=1.00
```

## API Server (`app.py`)

Run locally:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 7860
```

Endpoints:

- `GET /` -> `{"status":"running"}`
- `GET /health` -> `{"status":"ok"}`
- `POST /reset` -> resets env and returns observation
- `POST /step` -> executes one action and returns observation/reward/done/info

Example API calls:

```bash
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d "{}"
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action":"assign_high_priority"}'
```

## Docker

Build image:

```bash
docker build -t test-env .
```

Run container:

```bash
docker run --rm -p 7860:7860 --name test-env-run test-env
```

Verify:

```bash
curl http://localhost:7860/
curl http://localhost:7860/health
```

## Hugging Face Spaces (Docker)

1. Create a Docker Space.
2. Push this project.
3. Set Space secrets/variables:
   - `HF_TOKEN` (required)
   - `MODEL_NAME` (optional)
   - `API_BASE_URL` (optional)
4. Deploy.

## Runtime Limits

Designed to run under:

- 2 vCPU
- 8 GB RAM
- less than 20 minutes runtime
