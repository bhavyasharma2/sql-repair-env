"""
Reward engine for the SQL Repair environment.

Shaped reward — partial credit at every step so the agent gets a
gradient signal even before fully solving the task:

  +0.20  query is syntactically valid (parses without error)
  +0.30  query executes without a runtime error
  +0.30  result has the correct number of rows
  +0.20  result exactly matches ground truth (values + order)
  ──────
   1.00  maximum

Penalties are baked in by withholding the above credits, so the
total is always in [0.0, 1.0] — no negative values that could
confuse training.
"""

from typing import Any
from app.models import Reward


def compute_reward(
    query: str,
    execution_result: list[dict] | None,
    execution_error: str | None,
    ground_truth: list[dict],
) -> Reward:
    """
    Compare the agent's execution result against ground truth and
    return a fully broken-down Reward object.

    Args:
        query            : the SQL string the agent submitted
        execution_result : rows returned by DuckDB (None if error)
        execution_error  : error message if execution failed (else None)
        ground_truth     : expected rows as list of dicts

    Returns:
        Reward with .total in [0.0, 1.0] and per-component breakdown
    """

    breakdown: dict[str, float] = {
        "syntax_valid": 0.0,
        "executes": 0.0,
        "row_count_correct": 0.0,
        "exact_match": 0.0,
    }

    syntax_valid = False
    executes = False
    row_count_correct = False
    exact_match = False

    # ── 1. Syntax validity ────────────────────────────────────────────────────
    # We consider the query syntactically valid if it's non-empty and doesn't
    # contain an obvious parse-level failure. The real parse check happens
    # during execution; we infer syntax validity from the error message.
    if query and query.strip():
        syntax_valid = True
        # Downgrade if execution error looks like a parse error
        if execution_error and any(
            kw in execution_error.lower()
            for kw in ("syntax error", "parser error", "unexpected token", "expected")
        ):
            syntax_valid = False

    if syntax_valid:
        breakdown["syntax_valid"] = 0.20

    # ── 2. Executes without error ─────────────────────────────────────────────
    if execution_error is None and execution_result is not None:
        executes = True
        breakdown["executes"] = 0.30

    # ── 3. Row count matches ground truth ─────────────────────────────────────
    if executes and execution_result is not None:
        if len(execution_result) == len(ground_truth):
            row_count_correct = True
            breakdown["row_count_correct"] = 0.30

    # ── 4. Exact match ────────────────────────────────────────────────────────
    # Normalise values: floats compared with tolerance, strings lowercased.
    if row_count_correct and execution_result is not None:
        if _results_match(execution_result, ground_truth):
            exact_match = True
            breakdown["exact_match"] = 0.20

    total = round(sum(breakdown.values()), 4)
    total = max(0.001, min(0.999, total))

    return Reward(
        total=total,
        syntax_valid=syntax_valid,
        executes=executes,
        row_count_correct=row_count_correct,
        exact_match=exact_match,
        breakdown=breakdown,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _results_match(actual: list[dict], expected: list[dict]) -> bool:
    """
    Row-by-row, column-by-column comparison.
    Floats compared within 1e-4 tolerance.
    String values compared case-insensitively after stripping whitespace.
    Column names compared case-insensitively.
    """
    if len(actual) != len(expected):
        return False

    for act_row, exp_row in zip(actual, expected):
        # Normalise keys to lowercase
        act_norm = {k.lower(): v for k, v in act_row.items()}
        exp_norm = {k.lower(): v for k, v in exp_row.items()}

        if set(act_norm.keys()) != set(exp_norm.keys()):
            return False

        for key in exp_norm:
            av = act_norm[key]
            ev = exp_norm[key]

            if isinstance(ev, float) or isinstance(av, float):
                try:
                    if abs(float(av) - float(ev)) > 1e-4:
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
