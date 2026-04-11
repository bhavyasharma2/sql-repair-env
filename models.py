"""
models.py — Typed Action, Observation, and State for SQL Repair Env.
"""

from typing import Optional

try:
    from openenv.core.env_server import Action, Observation, State
except ImportError:
    from pydantic import BaseModel
    Action = BaseModel
    Observation = BaseModel
    State = BaseModel


class SQLRepairAction(Action):
    query: str = ""
    explanation: Optional[str] = None


class SQLRepairObservation(Observation):
    task_id: str = ""
    difficulty: str = ""
    schema_sql: str = ""
    sample_data: str = ""
    broken_query: str = ""
    objective: str = ""
    last_query: Optional[str] = None
    last_execution_result: Optional[str] = None
    last_reward: Optional[float] = None
    last_error: Optional[str] = None
    step_number: int = 0
    max_steps: int = 5


class SQLRepairState(State):
    task_id: str = ""
    difficulty: str = ""
    step_number: int = 0
    max_steps: int = 5
    done: bool = False
    best_reward: float = 0.001
    current_query: Optional[str] = None
