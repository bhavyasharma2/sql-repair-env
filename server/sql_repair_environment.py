"""
server/sql_repair_environment.py

Core SQLRepairEnvironment implementing the OpenEnv Environment interface.
Uses DuckDB in-memory for isolated, deterministic SQL execution.
"""

import uuid
import duckdb
from typing import Optional

try:
    from openenv.core.env_server import Environment
    from openenv.core.env_server import State as BaseState
    from ..models import SQLRepairAction, SQLRepairObservation
    from ..tasks import TASKS
except ImportError:
    from openenv.core.env_server import Environment
    from openenv.core.env_server import State as BaseState
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import SQLRepairAction, SQLRepairObservation
    from tasks import TASKS

MAX_STEPS = 5
TASK_ORDER = ["task_easy", "task_medium", "task_hard", "task_expert", "task_adversarial", "task_business"]


class SQLRepairEnvironment(Environment):
    """
    RL environment where agents learn to fix broken SQL queries.

    Shaped reward per step (sums to 1.0, clamped to 0.001-0.999):
      +0.20  syntax valid
      +0.30  executes without error
      +0.30  correct row count
      +0.20  exact match with ground truth
    """

    def __init__(self) -> None:
        super().__init__()
        self._task = None
        self._conn = None
        self._step_number = 0
        self._done = False
        self._best_reward = 0.001
        self._last_query = None
        self._last_exec_result = None
        self._last_error = None
        self._last_reward = None
        self._task_index = 0

    def reset(self, seed=None, episode_id=None, **kwargs):
        task_id = kwargs.get("task_id", None)
        if task_id and task_id in TASKS:
            self._task = TASKS[task_id]
        else:
            if seed is not None:
                idx = seed % len(TASK_ORDER)
            else:
                idx = self._task_index % len(TASK_ORDER)
                self._task_index += 1
            self._task = TASKS[TASK_ORDER[idx]]

        self._step_number = 0
        self._done = False
        self._best_reward = 0.001
        self._last_query = None
        self._last_exec_result = None
        self._last_error = None
        self._last_reward = None

        self._conn = duckdb.connect(":memory:")
        self._conn.execute(self._task["schema_sql"])
        self._conn.execute(self._task["seed_sql"])

        return self._build_observation()

    def step(self, action):
        if self._task is None or self._conn is None:
            self.reset()
        if self._done:
            return self._build_observation()

        self._step_number += 1
        self._last_query = action.query

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

        reward, breakdown = self._compute_reward(action.query, execution_result, execution_error)
        self._last_reward = reward

        if reward > self._best_reward:
            self._best_reward = reward

        self._done = (breakdown["exact_match"] > 0) or (self._step_number >= MAX_STEPS)

        return self._build_observation()

    @property
    def state(self):
        s = BaseState()
        s.step_count = self._step_number
        s.task_id = self._task["task_id"] if self._task else ""
        s.difficulty = self._task["difficulty"] if self._task else ""
        s.max_steps = MAX_STEPS
        s.done = self._done
        s.best_reward = self._best_reward
        s.current_query = self._last_query
        return s

    def _compute_reward(self, query, execution_result, execution_error):
        breakdown = {"syntax_valid": 0.0, "executes": 0.0, "row_count_correct": 0.0, "exact_match": 0.0}
        ground_truth = self._task["ground_truth"]

        syntax_valid = bool(query and query.strip())
        if execution_error and any(
            kw in execution_error.lower()
            for kw in ("syntax error", "parser error", "unexpected token", "expected")
        ):
            syntax_valid = False
        if syntax_valid:
            breakdown["syntax_valid"] = 0.20

        if execution_error is None and execution_result is not None:
            breakdown["executes"] = 0.30

        if execution_result is not None and len(execution_result) == len(ground_truth):
            breakdown["row_count_correct"] = 0.30

        if breakdown["row_count_correct"] > 0 and execution_result is not None:
            if self._results_match(execution_result, ground_truth):
                breakdown["exact_match"] = 0.20

        total = max(0.001, min(0.999, round(sum(breakdown.values()), 4)))
        return total, breakdown

    def _results_match(self, actual, expected):
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

    def _build_observation(self):
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

    def _format_result(self, rows):
        if not rows:
            return "(no rows returned)"
        cols = list(rows[0].keys())
        lines = [" | ".join(cols), "-" * 40]
        for row in rows:
            lines.append(" | ".join(str(v) for v in row.values()))
        return "\n".join(lines)
