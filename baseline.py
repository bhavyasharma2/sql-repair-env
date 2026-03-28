"""
Baseline inference script for the SQL Repair OpenEnv environment.

Uses the OpenAI API to run a model against all three tasks.
Reads credentials from environment variables.

Usage:
    OPENAI_API_KEY=sk-... python baseline.py

    # Or against a deployed Space:
    OPENAI_API_KEY=sk-... ENV_URL=https://your-space.hf.space python baseline.py
"""

import os
import json
import requests
from openai import OpenAI

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
TASK_IDS = ["task_easy", "task_medium", "task_hard"]
MODEL = "gpt-4o"
MAX_STEPS = 5


SYSTEM_PROMPT = """You are an expert SQL debugger. You will be given:
1. A database schema (CREATE TABLE statements)
2. Some sample data (INSERT statements)
3. A broken SQL query
4. The objective — what the query SHOULD return

Your job is to return a corrected SQL query that exactly satisfies the objective.

Rules:
- Return ONLY the corrected SQL query, nothing else
- No markdown, no code fences, no explanation
- The query must be valid SQL compatible with DuckDB/SQLite
- Fix ALL bugs in the query, not just the most obvious one
"""

def build_user_prompt(obs: dict) -> str:
    return f"""## Schema
{obs['schema_sql']}

## Sample data
{obs['sample_data']}

## Objective
{obs['objective']}

## Broken query to fix
{obs['broken_query']}

{f"## Previous attempt" if obs.get('last_action') else ""}
{f"Query: {obs['last_action']}" if obs.get('last_action') else ""}
{f"Result: {obs['last_execution_result']}" if obs.get('last_execution_result') else ""}
{f"Reward: {obs['last_reward']}" if obs.get('last_reward') is not None else ""}

Return only the corrected SQL query:"""


def run_episode(client: OpenAI, task_id: str) -> dict:
    """Run one full episode for a task, return final grading result."""
    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"{'='*60}")

    # Reset
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()

    print(f"Objective: {obs['objective']}")
    print(f"Broken query:\n{obs['broken_query']}\n")

    best_reward = 0.0
    final_result = None

    for step_num in range(1, MAX_STEPS + 1):
        # Ask the model
        user_prompt = build_user_prompt(obs)
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
        )
        proposed_query = completion.choices[0].message.content.strip()
        print(f"Step {step_num} — proposed query:\n{proposed_query}")

        # Submit action
        resp = requests.post(
            f"{ENV_URL}/step",
            json={"query": proposed_query},
        )
        resp.raise_for_status()
        result = resp.json()

        reward = result["reward"]
        done = result["done"]
        obs = result["observation"]

        print(f"  Reward: {reward['total']:.3f} "
              f"(syntax={reward['syntax_valid']}, "
              f"runs={reward['executes']}, "
              f"rows={reward['row_count_correct']}, "
              f"exact={reward['exact_match']})")

        if reward["total"] > best_reward:
            best_reward = reward["total"]
            final_result = reward

        if done:
            print(f"  Episode done at step {step_num}.")
            break

    print(f"\nFinal best reward for {task_id}: {best_reward:.3f}")
    return {
        "task_id": task_id,
        "best_reward": best_reward,
        "exact_match": final_result["exact_match"] if final_result else False,
    }


def run_baseline() -> dict:
    """Run baseline agent on all tasks and return aggregate scores."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key)
    results = {}

    for task_id in TASK_IDS:
        results[task_id] = run_episode(client, task_id)

    scores = {tid: r["best_reward"] for tid, r in results.items()}
    avg = round(sum(scores.values()) / len(scores), 4)

    summary = {
        "model": MODEL,
        "individual_scores": scores,
        "average_score": avg,
        "exact_matches": {
            tid: r["exact_match"] for tid, r in results.items()
        },
    }

    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_baseline()
