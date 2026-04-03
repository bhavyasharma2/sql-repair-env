"""
SQLRepairEnv — core environment class.

Implements the full OpenEnv interface:
  reset(task_id)  → Observation
  step(action)    → (Observation, Reward, done, info)
  state()         → EnvState

Uses DuckDB in-memory for deterministic, sandboxed SQL execution.
Each episode is fully isolated — a fresh DB is created on every reset().
"""

import duckdb
from typing import Any

from app.models import Action, Observation, Reward, EnvState
from app.tasks import TASKS
from app.rewards import compute_reward


MAX_STEPS = 5  #agent gets up to 5 attempts per episode


class SQLRepairEnv:
    def __init__(self) -> None:
        self._task: dict[str, Any] | None = None
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._step_number: int = 0
        self._done: bool = False
        self._best_reward: float = 0.0
        self._last_action: str | None = None
        self._last_exec_result: str | None = None
        self._last_reward: float | None = None

    #reset: 

    def reset(self, task_id: str = "task_easy") -> Observation:
        """
        Start a new episode for the given task.
        Creates a fresh in-memory DuckDB, runs schema + seed SQL,
        and returns the initial observation.
        """
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid options: {list(TASKS.keys())}"
            )

        self._task = TASKS[task_id]
        self._step_number = 0
        self._done = False
        self._best_reward = 0.0
        self._last_action = None
        self._last_exec_result = None
        self._last_reward = None

        #fresh isolated database for this episode
        self._conn = duckdb.connect(":memory:")
        self._conn.execute(self._task["schema_sql"])
        self._conn.execute(self._task["seed_sql"])

        return self._build_observation()

    #step:

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """
        Execute the agent's proposed query against the live database,
        compute shaped reward, and return the next observation.
        """
        if self._task is None or self._conn is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._step_number += 1
        self._last_action = action.query

        #executes the submitted query:
        execution_result: list[dict] | None = None
        execution_error: str | None = None

        try:
            rel = self._conn.execute(action.query)
            rows = rel.fetchall()
            cols = [desc[0] for desc in rel.description]
            execution_result = [dict(zip(cols, row)) for row in rows]
            self._last_exec_result = self._format_result(execution_result)
        except Exception as exc:
            execution_error = str(exc)
            self._last_exec_result = f"ERROR: {execution_error}"

        #computes rewards:
        reward = compute_reward(
            query=action.query,
            execution_result=execution_result,
            execution_error=execution_error,
            ground_truth=self._task["ground_truth"],
        )

        self._last_reward = reward.total
        if reward.total > self._best_reward:
            self._best_reward = reward.total

        #termination of episode:
        #done if: exact match achieved, or max steps reached
        self._done = reward.exact_match or (self._step_number >= MAX_STEPS)

        obs = self._build_observation()

        info = {
            "execution_error": execution_error,
            "rows_returned": len(execution_result) if execution_result else 0,
            "best_reward_so_far": self._best_reward,
        }

        return obs, reward, self._done, info

    #state:

    def state(self) -> EnvState:
        """Return the current internal state (for /state endpoint)."""
        if self._task is None:
            raise RuntimeError("Call reset() first.")
        return EnvState(
            task_id=self._task["task_id"],
            difficulty=self._task["difficulty"],
            step_number=self._step_number,
            max_steps=MAX_STEPS,
            done=self._done,
            best_reward_so_far=self._best_reward,
            current_query=self._last_action,
        )

    #helpers:

    def _build_observation(self) -> Observation:
        assert self._task is not None
        return Observation(
            task_id=self._task["task_id"],
            difficulty=self._task["difficulty"],
            schema_sql=self._task["schema_sql"],
            sample_data=self._task["seed_sql"],
            broken_query=self._task["broken_query"],
            objective=self._task["objective"],
            last_action=self._last_action,
            last_execution_result=self._last_exec_result,
            last_reward=self._last_reward,
            step_number=self._step_number,
            max_steps=MAX_STEPS,
        )

    def _format_result(self, rows: list[dict]) -> str:
        """Human-readable result for the observation."""
        if not rows:
            return "(no rows returned)"
        cols = list(rows[0].keys())
        header = " | ".join(cols)
        sep = "-" * len(header)
        lines = [header, sep]
        for row in rows:
            lines.append(" | ".join(str(v) for v in row.values()))
        return "\n".join(lines)
