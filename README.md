---
tags:
- openenv
---
# SQL Repair Environment — OpenEnv

An RL environment where agents learn to **diagnose and fix broken SQL queries**.

Given a database schema, sample data, and a broken query, the agent must produce
a corrected query that satisfies a plain-English objective. Graders are fully
deterministic — queries are executed against an in-memory DuckDB database and
results are compared against ground-truth row sets.


## Motivation

Every data team has analysts writing SQL that silently returns wrong results —
wrong joins, bad aggregations, off-by-one filters. An agent that can reliably
detect and fix these bugs has immediate real-world value. No existing OpenEnv
environment covers SQL correctness evaluation.


## Environment Design

### Observation space

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Current task identifier |
| `difficulty` | string | `easy` / `medium` / `hard` |
| `schema_sql` | string | CREATE TABLE statements |
| `sample_data` | string | INSERT statements with seed rows |
| `broken_query` | string | The SQL query to fix |
| `objective` | string | Natural language description of correct result |
| `last_action` | string? | Query submitted in previous step |
| `last_execution_result` | string? | DuckDB output or error from last step |
| `last_reward` | float? | Reward received for last action |
| `step_number` | int | Current step (0-indexed) |
| `max_steps` | int | Episode length limit (5) |

### Action space

```json
{
  "query": "SELECT ...",
  "explanation": "(optional) what was wrong and what was fixed"
}
```

### Reward function

Shaped partial credit — the agent receives signal at every step:

| Component | Weight | Condition |
|---|---|---|
| Syntax valid | +0.20 | Query parses without error |
| Executes | +0.30 | Query runs without runtime error |
| Row count correct | +0.30 | Result has expected number of rows |
| Exact match | +0.20 | Result matches ground truth exactly |

Total reward range: **0.0 – 1.0**


## Tasks

### Task 1 — Easy (`task_easy`)

**Domain:** Employee table, simple SELECT
**Bugs:** Missing comma in SELECT list + `FORM` instead of `FROM`
**Grader:** Does the query execute? Do result rows match ground truth (3 engineering employees ordered by salary)?
**Expected baseline score:** 1.0

### Task 2 — Medium (`task_medium`)

**Domain:** Customers + Orders (2-table join)
**Bugs:** `INNER JOIN` should be `LEFT JOIN`; `HAVING` references `amount` instead of `total_value`
**Grader:** Does result set match expected aggregation (2 customers with total > 200)?
**Expected baseline score:** 0.8–1.0

### Task 3 — Hard (`task_hard`)

**Domain:** Sales + Products + Regions (window function)
**Bug:** `ROW_NUMBER()` partitioned by `product_id` instead of `region_id` — returns wrong top product per region. Query runs cleanly, silent semantic error.
**Grader:** Do result rows match ground truth (correct top product per region)?
**Expected baseline score:** 0.5–0.8 (window function partitioning mistakes are a known LLM weak spot)

### Task 4 — Expert (`task_expert`)

**Domain:** Employees table with manager hierarchy (recursive CTE)
**Bugs:** Two bugs — anchor condition uses `IS NOT NULL` instead of `IS NULL`; recursive join uses `e.id = cte.id` instead of `e.manager_id = cte.id`
**Grader:** Does result return all 6 employees at correct org depths?
**Expected baseline score:** 0.2–0.6 (frontier models rarely fix both bugs in one shot)


## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/reset` | Start new episode |
| POST | `/step` | Submit repaired query |
| GET | `/state` | Current episode state |
| GET | `/tasks` | List all tasks + action schema |
| POST | `/grader` | Score a query without running episode |
| GET | `/baseline` | Run baseline agent on all tasks |

### Example: full episode

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy"}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT first_name, last_name, salary FROM employees WHERE department = '\''Engineering'\'' ORDER BY salary DESC;"}'

# Grade externally
curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy", "query": "SELECT first_name, last_name, salary FROM employees WHERE department = '\''Engineering'\'' ORDER BY salary DESC;"}'
```


## Setup & Usage

### Local development

```bash
git clone <your-repo>
cd sql-repair-env
pip install -r requirements.txt
uvicorn app.main:app --reload --port 7860
```

Open http://localhost:7860/docs for the interactive API docs.

### Docker

```bash
docker build -t sql-repair-env .
docker run -p 7860:7860 sql-repair-env
```

### Baseline script

```bash
export OPENAI_API_KEY=sk-...
python baseline.py
```


## Baseline Scores

| Task | Score | Exact match |
|---|---|---|
| task_easy | — | — |
| task_medium | — | — |
| task_hard | — | — |
| task_expert | — | — |
| **Average** | **—** | |

*Run `python baseline.py` with `OPENAI_API_KEY` set to populate this table.*


## Project Structure

```
sql-repair-env/
├── app/
│   ├── __init__.py
│   ├── main.py        # FastAPI server + all endpoints
│   ├── env.py         # SQLRepairEnv core class
│   ├── tasks.py       # Task definitions with schemas + broken queries
│   ├── graders.py     # Deterministic graders
│   ├── rewards.py     # Shaped reward function
│   └── models.py      # Pydantic Observation / Action / Reward
├── baseline.py        # OpenAI inference script
├── openenv.yaml       # OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
└── README.md
```


## License

MIT

