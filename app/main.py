"""
FastAPI server for the SQL Repair OpenEnv environment.

Endpoints required by the OpenEnv spec:
  POST /reset         → start a new episode
  POST /step          → submit an action
  GET  /state         → current internal state
  GET  /tasks         → list tasks + action schema
  POST /grader        → score a query externally
  GET  /baseline      → run baseline agent and return scores
  GET  /health        → liveness check (returns 200)
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional
import os

from app.env import SQLRepairEnv
from app.models import Action, Observation, Reward, EnvState
from app.tasks import TASKS
from app.graders import grade, grade_all

app = FastAPI(
    title="SQL Repair — OpenEnv Environment",
    description=(
        "An RL environment where agents learn to fix broken SQL queries. "
        "Three tasks of increasing difficulty: syntax repair (easy), "
        "wrong-join semantics (medium), silent semantic error (hard)."
    ),
    version="1.0.0",
)



@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


#one environment instance per server process.
#for multi-agent / parallel use, swap for a session-keyed dict.
env = SQLRepairEnv()


# request / response models:

class ResetRequest(BaseModel):
    task_id: str = "task_easy"

class StepRequest(BaseModel):
    query: str
    explanation: Optional[str] = None

class GraderRequest(BaseModel):
    task_id: str
    query: str

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict


#endpoints:
@app.get("/health")
def health():
    """Liveness check — must return 200 for HF Space ping."""
    return {"status": "ok", "environment": "sql-repair-env"}


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest):
    """
    Start a new episode.
    Pass task_id: "task_easy" | "task_medium" | "task_hard"
    """
    try:
        obs = env.reset(task_id=req.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return obs


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    """
    Submit a repaired SQL query as the agent's action.
    Returns the next observation, shaped reward, done flag, and info dict.
    """
    try:
        action = Action(query=req.query, explanation=req.explanation)
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=EnvState)
def state():
    """Return current internal state of the environment."""
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def tasks():
    """
    Return all available tasks with their objectives and the action schema.
    Required by the OpenEnv spec.
    """
    task_list = []
    for task_id, task in TASKS.items():
        task_list.append({
            "task_id": task_id,
            "difficulty": task["difficulty"],
            "objective": task["objective"],
            "schema_sql": task["schema_sql"],
            "broken_query": task["broken_query"],
        })

    return {
        "tasks": task_list,
        "action_schema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The repaired SQL query the agent proposes",
                },
                "explanation": {
                    "type": "string",
                    "description": "Optional: explanation of what was wrong and what was fixed",
                },
            },
        },
    }


@app.post("/grader")
def grader(req: GraderRequest):
    """
    Score a single query against a task without running a full episode.
    Returns score (0.0–1.0) and component breakdown.
    """
    try:
        result = grade(task_id=req.task_id, query=req.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/baseline")
def baseline():
    """
    Run the built-in baseline agent (GPT-4o via OpenAI API) against all
    three tasks and return scores. Reads OPENAI_API_KEY from environment.

    If OPENAI_API_KEY is not set, returns a demo response with hand-crafted
    queries so the endpoint always responds (required by spec).
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        #the real baseline agent
        try:
            from baseline import run_baseline
            return run_baseline()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Baseline run failed: {e}")
    else:
        #demo mode — return scores for known-correct queries so the
        # /baseline endpoint always returns 200 even without an API key.
        demo_queries = {
            "task_easy": (
                "SELECT first_name, last_name, salary "
                "FROM employees "
                "WHERE department = 'Engineering' "
                "ORDER BY salary DESC;"
            ),
            "task_medium": (
                "SELECT c.name, COALESCE(SUM(o.amount), 0) AS total_value "
                "FROM customers c "
                "LEFT JOIN orders o ON c.id = o.customer_id "
                "GROUP BY c.id, c.name "
                "HAVING COALESCE(SUM(o.amount), 0) > 200 "
                "ORDER BY total_value DESC;"
            ),
            "task_hard": (
                "WITH ranked AS ("
                "SELECT r.name AS region_name, p.name AS product_name, "
                "SUM(s.quantity) AS total_qty, "
                "ROW_NUMBER() OVER ("
                "PARTITION BY s.region_id "
                "ORDER BY SUM(s.quantity) DESC, p.id ASC"
                ") AS rn "
                "FROM sales s "
                "JOIN regions r ON s.region_id = r.id "
                "JOIN products p ON s.product_id = p.id "
                "GROUP BY s.region_id, r.id, r.name, p.id, p.name"
                ") "
                "SELECT region_name, product_name, total_qty "
                "FROM ranked WHERE rn = 1 ORDER BY region_name ASC;"
            ),
            "task_expert": (
                "WITH RECURSIVE org AS ("
                "SELECT id, name, manager_id, 0 AS depth "
                "FROM employees WHERE manager_id IS NULL "
                "UNION ALL "
                "SELECT e.id, e.name, e.manager_id, cte.depth + 1 "
                "FROM employees e "
                "JOIN org cte ON e.manager_id = cte.id"
                ") "
                "SELECT name, depth FROM org ORDER BY depth ASC, name ASC;"
            ),
        }
        return {
            "mode": "demo",
            "note": "Set OPENAI_API_KEY to run the real baseline agent.",
            **grade_all(demo_queries),
        }