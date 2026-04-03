"""
Graders for the SQL Repair environment.

Each grader runs a fresh isolated episode and returns a score in [0.0, 1.0].
Graders are deterministic — same input always produces same score.

Used by:
  - /grader endpoint (score a submitted query externally)
  - /baseline endpoint (score the baseline agent on all tasks)
"""

import duckdb
from app.tasks import TASKS
from app.rewards import compute_reward, _results_match


def grade(task_id: str, query: str) -> dict:
    """
    Run `query` against a fresh in-memory DB for `task_id`.
    Returns a grading report with score and per-component breakdown.

    Args:
        task_id : one of "task_easy", "task_medium", "task_hard"
        query   : the SQL string to evaluate

    Returns:
        {
            "task_id": str,
            "score": float,          # 0.0 – 1.0
            "syntax_valid": bool,
            "executes": bool,
            "row_count_correct": bool,
            "exact_match": bool,
            "breakdown": dict[str, float],
            "result_preview": str,
            "error": str | None,
        }
    """
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id '{task_id}'")

    task = TASKS[task_id]

    #isolated DB (no state leaks between grader calls)
    conn = duckdb.connect(":memory:")
    conn.execute(task["schema_sql"])
    conn.execute(task["seed_sql"])

    execution_result = None
    execution_error = None
    result_preview = ""

    try:
        rel = conn.execute(query)
        rows = rel.fetchall()
        cols = [desc[0] for desc in rel.description]
        execution_result = [dict(zip(cols, row)) for row in rows]
        result_preview = _format_rows(execution_result)
    except Exception as exc:
        execution_error = str(exc)
        result_preview = f"ERROR: {execution_error}"
    finally:
        conn.close()

    reward = compute_reward(
        query=query,
        execution_result=execution_result,
        execution_error=execution_error,
        ground_truth=task["ground_truth"],
    )

    return {
        "task_id": task_id,
        "score": reward.total,
        "syntax_valid": reward.syntax_valid,
        "executes": reward.executes,
        "row_count_correct": reward.row_count_correct,
        "exact_match": reward.exact_match,
        "breakdown": reward.breakdown,
        "result_preview": result_preview,
        "error": execution_error,
    }


def grade_all(queries: dict[str, str]) -> dict:
    """
    Grade a dict of {task_id: query} and return aggregate results.
    Used by the /baseline endpoint.
    """
    results = {}
    total_score = 0.0

    for task_id, query in queries.items():
        result = grade(task_id, query)
        results[task_id] = result
        total_score += result["score"]

    avg_score = round(total_score / len(queries), 4) if queries else 0.0

    return {
        "individual_scores": results,
        "average_score": avg_score,
        "total_tasks": len(queries),
    }


def _format_rows(rows: list[dict]) -> str:
    if not rows:
        return "(no rows)"
    cols = list(rows[0].keys())
    lines = [" | ".join(cols)]
    for row in rows:
        lines.append(" | ".join(str(v) for v in row.values()))
    return "\n".join(lines)
