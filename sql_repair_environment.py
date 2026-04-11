"""
server/sql_repair_environment.py

Core SQLRepairEnvironment class implementing the OpenEnv Environment interface.
Uses DuckDB in-memory for fully isolated, deterministic SQL execution.
"""

import uuid
import duckdb
from typing import Optional

try:
    from openenv.core.env_server import Environment
    from ..models import SQLRepairAction, SQLRepairObservation, SQLRepairState
    from ..tasks import TASKS
except ImportError:
    from openenv.core.env_server import Environment
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import SQLRepairAction, SQLRepairObservation, SQLRepairState
    from tasks import TASKS

MAX_STEPS = 5


class SQLRepairEnvironment(Environment):
    """
    An RL environment where agents learn to diagnose and fix broken SQL queries.

    The agent receives:
      - A database schema and sample data
      - A broken SQL query
      - A plain-English objective describing what the query should return

    The agent must submit a corrected SQL query. The environment executes it
    against an in-memory DuckDB instance and returns shaped reward:

      +0.20  query is syntactically valid
      +0.30  query executes without runtime error
      +0.30  result has the correct number of rows
      +0.20  result matches ground truth exactly
      ──────
       1.00  maximum (clamped to 0.001–0.999 for training stability)

    Each episode uses a fresh, isolated DuckDB database.
    """

    def __init__(self) -> None:
        super().__init__()
        self._task: Optional[dict] = None
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._step_number: int = 0
        self._done: bool = False
        self._best_reward: float = 0.001
        self._last_query: Optional[str] = None
        self._last_exec_result: Optional[str] = None
        self._last_error: Optional[str] = None
        self._last_reward: Optional[float] = None
        self._episode_id: str = ""

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "task_easy") -> SQLRepairObservation:
        """
        Start a new episode. Creates a fresh DuckDB instance with the task schema.
        Returns the initial observation (no feedback yet).
        """
        if task_id not in TASKS:
            # Default to easy if unknown task
            task_id = "task_easy"

        self._task = TASKS[task_id]
        self._step_number = 0
        self._done = False
        self._best_reward = 0.001
        self._last_query = None
        self._last_exec_result = None
        self._last_error = None
        self._last_reward = None
        self._episode_id = str(uuid.uuid4())

        # Fresh isolated database for this episode
        self._conn = duckdb.connect(":memory:")
        self._conn.execute(self._task["schema_sql"])
        self._conn.execute(self._task["seed_sql"])

        return self._build_observation()

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action: SQLRepairAction) -> SQLRepairObservation:
        """
        Execute the agent's proposed SQL query, compute reward, update state.
        Returns next observation with execution feedback.
        """
        if self._task is None or self._conn is None:
            self.reset()

        if self._done:
            return self._build_observation()

        self._step_number += 1
        self._last_query = action.query

        # Execute the submitted query
        execution_result = None
        execution_error = None

        try:
            rel = self._conn.execute(action.query)
            rows = rel.fetchall()
            cols = [desc[0] for desc in rel.description]
            execution_result = [dict(zip(cols, row)) for row in rows]
            self._last_exec_result = self._format_result(execution_result)
            self._last_error = None
        except Exception as exc:
            execution_error = str(exc)
            self._last_exec_result = f"ERROR: {execution_error}"
            self._last_error = execution_error

        # Compute shaped reward
        reward, breakdown = self._compute_reward(
            action.query, execution_result, execution_error
        )
        self._last_reward = reward

        if reward > self._best_reward:
            self._best_reward = reward

        # Episode ends on exact match or max steps
        self._done = (breakdown["exact_match"] > 0) or (self._step_number >= MAX_STEPS)

        obs = self._build_observation()
        obs.reward_breakdown = breakdown
        return obs

    # ── state ─────────────────────────────────────────────────────────────────

    @property
    def state(self) -> SQLRepairState:
        return SQLRepairState(
            task_id=self._task["task_id"] if self._task else "",
            difficulty=self._task["difficulty"] if self._task else "",
            step_number=self._step_number,
            max_steps=MAX_STEPS,
            done=self._done,
            best_reward=self._best_reward,
            current_query=self._last_query,
            total_steps_taken=self._step_number,
        )

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        query: str,
        execution_result: Optional[list],
        execution_error: Optional[str],
    ) -> tuple[float, dict]:
        breakdown = {
            "syntax_valid": 0.0,
            "executes": 0.0,
            "row_count_correct": 0.0,
            "exact_match": 0.0,
        }

        ground_truth = self._task["ground_truth"]

        # 1. Syntax valid
        syntax_valid = bool(query and query.strip())
        if execution_error and any(
            kw in execution_error.lower()
            for kw in ("syntax error", "parser error", "unexpected token", "expected")
        ):
            syntax_valid = False
        if syntax_valid:
            breakdown["syntax_valid"] = 0.20

        # 2. Executes
        if execution_error is None and execution_result is not None:
            breakdown["executes"] = 0.30

        # 3. Row count
        if execution_result is not None and len(execution_result) == len(ground_truth):
            breakdown["row_count_correct"] = 0.30

        # 4. Exact match
        if breakdown["row_count_correct"] > 0 and execution_result is not None:
            if self._results_match(execution_result, ground_truth):
                breakdown["exact_match"] = 0.20

        total = sum(breakdown.values())
        # Clamp strictly between 0 and 1 for training stability
        total = max(0.001, min(0.999, round(total, 4)))

        return total, breakdown

    def _results_match(self, actual: list, expected: list) -> bool:
        if len(actual) != len(expected):
            return False
        for act_row, exp_row in zip(actual, expected):
            act = {k.lower(): v for k, v in act_row.items()}
            exp = {k.lower(): v for k, v in exp_row.items()}
            if set(act.keys()) != set(exp.keys()):
                return False
            for key in exp:
                av, ev = act[key], exp[key]
                if isinstance(ev, float) or isinstance(av, float):
                    try:
                        if abs(float(av) - float(ev)) > 1e-3:
                            return False
                    except (TypeError, ValueError):
                        return False
                elif isinstance(ev, str):
                    if str(av).strip().lower() != str(ev).strip().lower():
                        return False
                else:
                    if av != ev:
                        return False
        return True

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_observation(self) -> SQLRepairObservation:
        assert self._task is not None
        return SQLRepairObservation(
            task_id=self._task["task_id"],
            difficulty=self._task["difficulty"],
            schema_sql=self._task["schema_sql"],
            sample_data=self._task["seed_sql"],
            broken_query=self._task["broken_query"],
            objective=self._task["objective"],
            last_query=self._last_query,
            last_execution_result=self._last_exec_result,
            last_reward=self._last_reward,
            last_error=self._last_error,
            step_number=self._step_number,
            max_steps=MAX_STEPS,
        )

    def _format_result(self, rows: list) -> str:
        if not rows:
            return "(no rows returned)"
        cols = list(rows[0].keys())
        lines = [" | ".join(cols), "-" * 40]
        for row in rows:
            lines.append(" | ".join(str(v) for v in row.values()))
        return "\n".join(lines)
