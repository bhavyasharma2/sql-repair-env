---
title: SQL Repair OpenEnv
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
- openenv
---

# SQL Repair — OpenEnv Environment

An RL environment where agents learn to **diagnose and fix broken SQL queries**. Given a database schema, sample data, and a broken query, the agent must produce a corrected query that satisfies a plain-English objective. Graders are fully deterministic — queries are executed against an in-memory DuckDB database and results compared against ground-truth row sets.

## Motivation

Every data team has analysts writing SQL that silently returns wrong results — wrong joins, bad aggregations, off-by-one filters. An agent that can reliably detect and fix these bugs has immediate real-world value. No existing OpenEnv environment covers SQL correctness evaluation.

---

## Environment Description

The SQL Repair environment presents the agent with a broken SQL query and asks it to fix the query so it returns the correct result set. The agent receives shaped reward signals at every step — not just at the end — so it can learn from partial progress.

The environment uses DuckDB in-memory for fully isolated, deterministic SQL execution. Each episode gets a fresh database instance.

---

## Action Space

The agent submits a repaired SQL query as its action:

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | The repaired SQL query the agent proposes |
| `explanation` | string | No | Optional explanation of the fix |

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Current task identifier |
| `difficulty` | string | easy / medium / hard / expert |
| `schema_sql` | string | CREATE TABLE statements |
| `sample_data` | string | INSERT statements with seed rows |
| `broken_query` | string | The SQL query to fix |
| `objective` | string | Natural language description of correct result |
| `last_action` | string? | Query submitted in previous step |
| `last_execution_result` | string? | DuckDB output or error from last step |
| `last_reward` | float? | Reward received for last action |
| `step_number` | int | Current step (0-indexed) |
| `max_steps` | int | Episode length limit (5) |

---

## Reward Function

Shaped partial credit at every step:

| Component | Weight | Condition |
|---|---|---|
| Syntax valid | +0.20 | Query parses without error |
| Executes | +0.30 | Query runs without runtime error |
| Row count correct | +0.30 | Result has expected number of rows |
| Exact match | +0.20 | Result matches ground truth exactly |

Total reward range: 0.0 to 1.0

---

## Tasks

### Task 1 — Easy (task_easy)

**Domain:** Employee table, simple SELECT
**Bugs:** Missing comma in SELECT list + FORM instead of FROM (syntax errors)
**Grader:** Does the query execute? Do result rows match ground truth?
**Baseline score:** 0.80

### Task 2 — Medium (task_medium)

**Domain:** Customers + Orders (2-table join)
**Bugs:** INNER JOIN should be LEFT JOIN; HAVING references wrong column alias
**Grader:** Does result set match expected aggregation?
**Baseline score:** 1.00

### Task 3 — Hard (task_hard)

**Domain:** Sales + Products + Regions (window function)
**Bug:** ROW_NUMBER() partitioned by wrong column — silent semantic error
**Grader:** Do result rows match ground truth (correct top product per region)?
**Baseline score:** 1.00

### Task 4 — Expert (task_expert)

**Domain:** Employees table with manager hierarchy (recursive CTE)
**Bugs:** Two bugs — wrong anchor condition + wrong recursive join
**Grader:** Does result return all 6 employees at correct org depths?
**Baseline score:** 1.00

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | /health | Liveness check |
| POST | /reset | Start new episode |
| POST | /step | Submit repaired query |
| GET | /state | Current episode state |
| GET | /tasks | List all tasks + action schema |
| POST | /grader | Score a query without running episode |
| GET | /baseline | Run baseline agent on all tasks |

---

## Setup & Usage

### Local development

```bash
git clone https://huggingface.co/spaces/iambhavyaa/sql-repair-env
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

### Running the inference script

```bash
export HF_TOKEN=hf_...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

---

## Baseline Scores

Measured with Qwen/Qwen2.5-72B-Instruct via HF Inference API at temperature=0:

| Task | Score | Exact match |
|---|---|---|
| task_easy | 0.80 | No (returns concatenated name, not separate columns) |
| task_medium | 1.00 | Yes |
| task_hard | 1.00 | Yes |
| task_expert | 1.00 | Yes |
| **Average** | **0.95** | |

---

## Project Structure

```
sql-repair-env/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── env.py
│   ├── tasks.py
│   ├── graders.py
│   ├── rewards.py
│   └── models.py
├── inference.py
├── baseline.py
├── train_grpo.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## License

MIT
