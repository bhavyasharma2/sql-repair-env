"""
End-to-end test suite for the SQL Repair OpenEnv environment.

Run with:
    pytest tests.py -v

Or without pytest:
    python tests.py

Tests cover:
  - All three tasks: reset → step → done
  - Reward shaping at each tier (syntax, execution, rows, exact)
  - Grader determinism
  - All FastAPI endpoints via TestClient (no server needed)
  - Edge cases: empty query, already-done episode, unknown task
"""

import sys

# ── Graceful import handling ──────────────────────────────────────────────────
try:
    from fastapi.testclient import TestClient
    from app.main import app
    from app.env import SQLRepairEnv
    from app.rewards import compute_reward, _results_match
    from app.graders import grade, grade_all
    from app.tasks import TASKS
    HAS_DEPS = True
except ImportError as e:
    print(f"[SKIP] Missing dependency: {e}")
    print("Run: pip install -r requirements.txt  then retry.")
    HAS_DEPS = False


# ── Correct queries (used throughout) ────────────────────────────────────────

CORRECT = {
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


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Reward logic (no DB needed)
# ══════════════════════════════════════════════════════════════════════════════

def test_reward_weights_sum_to_one():
    """Reward component weights must sum to exactly 1.0."""
    # Score a correct submission end-to-end is tested later;
    # here we just check the arithmetic of the breakdown dict.
    r = compute_reward(
        query="SELECT 1",
        execution_result=[{"name": "Widget C", "total_revenue": 1200.0}],
        execution_error=None,
        ground_truth=[{"name": "Widget C", "total_revenue": 1200.0}],
    )
    total = sum(r.breakdown.values())
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"
    print("  PASS  reward weights sum to 1.0")


def test_reward_zero_on_syntax_error():
    r = compute_reward(
        query="SELEKT * FORM foo",
        execution_result=None,
        execution_error="Parser error: syntax error at or near 'SELEKT'",
        ground_truth=[{"x": 1}],
    )
    assert r.total == 0.0, f"Expected 0.0 on syntax error, got {r.total}"
    assert not r.syntax_valid
    assert not r.executes
    print("  PASS  reward=0.0 on syntax error")


def test_reward_partial_on_wrong_row_count():
    """Query runs but returns wrong number of rows → gets syntax+executes credit only."""
    r = compute_reward(
        query="SELECT * FROM employees",
        execution_result=[{"x": 1}, {"x": 2}],  # 2 rows
        execution_error=None,
        ground_truth=[{"x": 1}],                 # expected 1 row
    )
    assert r.syntax_valid
    assert r.executes
    assert not r.row_count_correct
    assert not r.exact_match
    expected = 0.20 + 0.30
    assert abs(r.total - expected) < 1e-9, f"Expected {expected}, got {r.total}"
    print("  PASS  partial reward on wrong row count")


def test_reward_full_on_exact_match():
    gt = [{"name": "Widget C", "total_revenue": 1200.0}]
    r = compute_reward(
        query="SELECT ...",
        execution_result=gt,
        execution_error=None,
        ground_truth=gt,
    )
    assert r.total == 1.0
    assert r.exact_match
    print("  PASS  full reward on exact match")


def test_results_match_float_tolerance():
    assert _results_match(
        [{"v": 1.000001}], [{"v": 1.0}]
    ), "Float tolerance should pass"
    assert not _results_match(
        [{"v": 2.0}], [{"v": 1.0}]
    ), "Different floats should not match"
    print("  PASS  float tolerance in result comparison")


def test_results_match_case_insensitive():
    assert _results_match([{"name": "ALICE"}], [{"name": "alice"}])
    assert _results_match([{"name": "  Alice  "}], [{"name": "Alice"}])
    print("  PASS  case-insensitive string comparison")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Graders (requires duckdb)
# ══════════════════════════════════════════════════════════════════════════════

def test_grader_correct_queries_score_1():
    """Correct queries must score 1.0 on all three tasks."""
    for task_id, query in CORRECT.items():
        result = grade(task_id, query)
        assert result["score"] == 1.0, (
            f"Task {task_id}: expected score=1.0, got {result['score']}\n"
            f"Preview: {result['result_preview']}\n"
            f"Error: {result['error']}"
        )
        print(f"  PASS  grader score=1.0 for {task_id}")


def test_grader_broken_queries_score_less_than_1():
    """Broken (original) queries must score < 1.0."""
    for task_id, task in TASKS.items():
        result = grade(task_id, task["broken_query"])
        assert result["score"] < 1.0, (
            f"Task {task_id}: broken query should not score 1.0, got {result['score']}"
        )
        print(f"  PASS  grader score<1.0 for broken {task_id} (got {result['score']:.2f})")


def test_grader_deterministic():
    """Same query must always return same score."""
    q = CORRECT["task_easy"]
    r1 = grade("task_easy", q)
    r2 = grade("task_easy", q)
    r3 = grade("task_easy", q)
    assert r1["score"] == r2["score"] == r3["score"], "Grader is not deterministic"
    print("  PASS  grader is deterministic (3 identical runs)")


def test_grader_unknown_task_raises():
    try:
        grade("task_nonexistent", "SELECT 1")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  PASS  grader raises ValueError on unknown task_id")


def test_grade_all():
    result = grade_all(CORRECT)
    assert result["average_score"] == 1.0
    assert result["total_tasks"] == 4
    print("  PASS  grade_all returns avg=1.0 for all correct queries")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Environment (requires duckdb + pydantic)
# ══════════════════════════════════════════════════════════════════════════════

def test_env_reset_returns_observation():
    env = SQLRepairEnv()
    for task_id in TASKS:
        obs = env.reset(task_id)
        assert obs.task_id == task_id
        assert obs.step_number == 0
        assert obs.broken_query
        assert obs.schema_sql
        assert obs.objective
        print(f"  PASS  reset() returns valid Observation for {task_id}")


def test_env_step_increases_step_number():
    env = SQLRepairEnv()
    from app.models import Action
    env.reset("task_easy")
    obs, reward, done, info = env.step(Action(query="SELECT 1"))
    assert obs.step_number == 1
    print("  PASS  step_number increments on step()")


def test_env_done_on_exact_match():
    env = SQLRepairEnv()
    from app.models import Action
    env.reset("task_easy")
    _, reward, done, _ = env.step(Action(query=CORRECT["task_easy"]))
    assert done, "Episode should be done after exact match"
    assert reward.exact_match
    assert reward.total == 1.0
    print("  PASS  done=True and reward=1.0 on exact match")


def test_env_done_on_max_steps():
    env = SQLRepairEnv()
    from app.models import Action
    env.reset("task_easy")
    done = False
    for _ in range(5):
        _, _, done, _ = env.step(Action(query="SELECT 1"))
    assert done, "Episode should be done after max_steps"
    print("  PASS  done=True after max_steps exhausted")


def test_env_step_before_reset_raises():
    env = SQLRepairEnv()
    from app.models import Action
    try:
        env.step(Action(query="SELECT 1"))
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
    print("  PASS  step() before reset() raises RuntimeError")


def test_env_state():
    env = SQLRepairEnv()
    from app.models import Action
    env.reset("task_medium")
    state = env.state()
    assert state.task_id == "task_medium"
    assert state.step_number == 0
    assert not state.done
    print("  PASS  state() returns correct EnvState")


def test_env_full_episode_easy():
    """Complete episode — should reach exact match in 1 step with correct query."""
    env = SQLRepairEnv()
    from app.models import Action
    obs = env.reset("task_easy")
    assert obs.task_id == "task_easy"

    obs, reward, done, info = env.step(Action(query=CORRECT["task_easy"]))
    assert done
    assert reward.total == 1.0
    assert info["rows_returned"] == 3
    print("  PASS  full easy episode: exact match in 1 step")


def test_env_full_episode_hard():
    """Hard task — correct query should score 1.0."""
    env = SQLRepairEnv()
    from app.models import Action
    env.reset("task_hard")
    _, reward, done, _ = env.step(Action(query=CORRECT["task_hard"]))
    assert done
    assert reward.total == 1.0
    print("  PASS  full hard episode: exact match in 1 step with correct query")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — API endpoints (TestClient)
# ══════════════════════════════════════════════════════════════════════════════

def test_api_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    print("  PASS  GET /health → 200")


def test_api_tasks():
    client = TestClient(app)
    r = client.get("/tasks")
    assert r.status_code == 200
    data = r.json()
    assert len(data["tasks"]) == 4
    assert "action_schema" in data
    difficulties = {t["difficulty"] for t in data["tasks"]}
    assert difficulties == {"easy", "medium", "hard", "expert"}
    print("  PASS  GET /tasks → 3 tasks with easy/medium/hard")


def test_api_reset():
    client = TestClient(app)
    r = client.post("/reset", json={"task_id": "task_easy"})
    assert r.status_code == 200
    obs = r.json()
    assert obs["task_id"] == "task_easy"
    assert obs["step_number"] == 0
    print("  PASS  POST /reset → valid Observation")


def test_api_step():
    client = TestClient(app)
    client.post("/reset", json={"task_id": "task_easy"})
    r = client.post("/step", json={"query": CORRECT["task_easy"]})
    assert r.status_code == 200
    data = r.json()
    assert data["done"] is True
    assert data["reward"]["total"] == 1.0
    print("  PASS  POST /step → done=True, reward=1.0 for correct query")


def test_api_state():
    client = TestClient(app)
    client.post("/reset", json={"task_id": "task_medium"})
    r = client.get("/state")
    assert r.status_code == 200
    state = r.json()
    assert state["task_id"] == "task_medium"
    print("  PASS  GET /state → correct task_id")


def test_api_grader():
    client = TestClient(app)
    r = client.post("/grader", json={
        "task_id": "task_easy",
        "query": CORRECT["task_easy"],
    })
    assert r.status_code == 200
    data = r.json()
    assert data["score"] == 1.0
    print("  PASS  POST /grader → score=1.0 for correct query")


def test_api_baseline_demo_mode():
    """Baseline endpoint should work without OPENAI_API_KEY (demo mode)."""
    import os
    os.environ.pop("OPENAI_API_KEY", None)
    client = TestClient(app)
    r = client.get("/baseline")
    assert r.status_code == 200
    data = r.json()
    assert data["average_score"] == 1.0
    assert data["mode"] == "demo"
    print("  PASS  GET /baseline → demo mode, avg=1.0")


def test_api_reset_invalid_task():
    client = TestClient(app)
    r = client.post("/reset", json={"task_id": "task_bogus"})
    assert r.status_code == 400
    print("  PASS  POST /reset with invalid task_id → 400")


# ══════════════════════════════════════════════════════════════════════════════
#  Runner
# ══════════════════════════════════════════════════════════════════════════════

SECTIONS = [
    ("Reward logic", [
        test_reward_weights_sum_to_one,
        test_reward_zero_on_syntax_error,
        test_reward_partial_on_wrong_row_count,
        test_reward_full_on_exact_match,
        test_results_match_float_tolerance,
        test_results_match_case_insensitive,
    ]),
    ("Graders", [
        test_grader_correct_queries_score_1,
        test_grader_broken_queries_score_less_than_1,
        test_grader_deterministic,
        test_grader_unknown_task_raises,
        test_grade_all,
    ]),
    ("Environment", [
        test_env_reset_returns_observation,
        test_env_step_increases_step_number,
        test_env_done_on_exact_match,
        test_env_done_on_max_steps,
        test_env_step_before_reset_raises,
        test_env_state,
        test_env_full_episode_easy,
        test_env_full_episode_hard,
    ]),
    ("API endpoints", [
        test_api_health,
        test_api_tasks,
        test_api_reset,
        test_api_step,
        test_api_state,
        test_api_grader,
        test_api_baseline_demo_mode,
        test_api_reset_invalid_task,
    ]),
]

if __name__ == "__main__":
    if not HAS_DEPS:
        sys.exit(1)

    passed = 0
    failed = 0
    errors = []

    for section_name, tests in SECTIONS:
        print(f"\n── {section_name} {'─'*(50-len(section_name))}")
        for test_fn in tests:
            try:
                test_fn()
                passed += 1
            except Exception as e:
                failed += 1
                errors.append((test_fn.__name__, str(e)))
                print(f"  FAIL  {test_fn.__name__}: {e}")

    print(f"\n{'═'*55}")
    print(f"  Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailed tests:")
        for name, msg in errors:
            print(f"  {name}: {msg}")
    print(f"{'═'*55}")
    sys.exit(0 if failed == 0 else 1)
