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

An RL environment where agents learn to **diagnose and fix broken SQL queries**. Six tasks of increasing difficulty covering real-world SQL bug patterns — from syntax errors to recursive CTE bugs and multi-CTE date logic errors.

## Install client

```bash
pip install git+https://huggingface.co/spaces/iambhavyaa/sql-repair-env
```

## Quick start

```python
import asyncio
from sql_repair_env import SQLRepairEnv, SQLRepairAction

async def main():
    async with SQLRepairEnv(base_url="https://iambhavyaa-sql-repair-env.hf.space") as env:
        result = await env.reset(task_id="task_easy")
        print(result.observation.broken_query)
        result = await env.step(SQLRepairAction(
            query="SELECT first_name, last_name, salary FROM employees WHERE department = 'Engineering' ORDER BY salary DESC;"
        ))
        print(f"Reward: {result.reward}")

asyncio.run(main())
```

## Motivation

Every data team has analysts writing SQL that silently returns wrong results — wrong joins, bad aggregations, off-by-one filters. An agent that reliably detects and fixes these bugs has immediate real-world value. No existing OpenEnv environment covers SQL correctness evaluation.

## Reward Function

Shaped partial credit at every step (sum = 1.0, clamped to 0.001–0.999):

| Component | Weight | Condition |
|---|---|---|
| Syntax valid | +0.20 | Query parses without error |
| Executes | +0.30 | Query runs without runtime error |
| Row count correct | +0.30 | Result has expected number of rows |
| Exact match | +0.20 | Result matches ground truth exactly |

## Tasks

| Task | Difficulty | Bug Type | Baseline Score |
|---|---|---|---|
| task_easy | easy | Syntax error (typo + missing comma) | 0.80 |
| task_medium | medium | Wrong JOIN + HAVING on wrong alias | 1.00 |
| task_hard | hard | ROW_NUMBER() wrong PARTITION BY | 1.00 |
| task_expert | expert | Recursive CTE — two bugs | 1.00 |
| task_adversarial | adversarial | Aggregate in WHERE instead of HAVING | 0.80 |
| task_business | business | Wrong date truncation in multi-CTE | 0.80 |
| **Average** | | | **0.90** |

Baseline measured with `Qwen/Qwen2.5-72B-Instruct` via HF Inference API.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | /health | Liveness check |
| POST | /reset | Start new episode |
| POST | /step | Submit repaired query |
| GET | /state | Current episode state |
| WS | /ws | WebSocket for persistent sessions |

## Setup

```bash
git clone https://github.com/bhavyasharma2/sql-repair-env
cd sql-repair-env
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Docker

```bash
docker build -t sql-repair-env .
docker run -p 7860:7860 sql-repair-env
```

## Run inference

```bash
export HF_TOKEN=hf_...
python inference.py
```

## License

MIT
