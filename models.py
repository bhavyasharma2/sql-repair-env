"""
models.py — Typed Action, Observation, and State for SQL Repair Env.

Follows the official OpenEnv spec: dataclasses inheriting from
openenv.core.env_server types.
"""

from dataclasses import dataclass, field
from typing import Optional

try:
    from openenv.core.env_server import Action, Observation, State
except ImportError:
    # Fallback for local development without openenv-core installed
    from dataclasses import dataclass as _dc
    Action = object
    Observation = object
    State = object


@dataclass
class SQLRepairAction(Action):
    """
    The agent submits a repaired SQL query.
    Optional explanation is logged but not graded.
    """
    query: str = ""
    explanation: Optional[str] = None


@dataclass
class SQLRepairObservation(Observation):
    """
    Everything the agent sees at each step.
    Includes the broken query, schema, objective, and feedback from
    the previous step so the agent can learn from its mistakes.
    """
    # Task context
    task_id: str = ""
    difficulty: str = ""
    schema_sql: str = ""
    sample_data: str = ""
    broken_query: str = ""
    objective: str = ""

    # Feedback from last step (None on first step)
    last_query: Optional[str] = None
    last_execution_result: Optional[str] = None
    last_reward: Optional[float] = None
    last_error: Optional[str] = None

    # Episode progress
    step_number: int = 0
    max_steps: int = 5

    # Reward breakdown (for transparency)
    reward_breakdown: Optional[dict] = None


@dataclass
class SQLRepairState(State):
    """
    Internal episode state returned by /state endpoint.
    """
    task_id: str = ""
    difficulty: str = ""
    step_number: int = 0
    max_steps: int = 5
    done: bool = False
    best_reward: float = 0.001
    current_query: Optional[str] = None
    total_steps_taken: int = 0
