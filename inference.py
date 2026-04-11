"""
inference.py — SQL Repair Env Inference Script

MANDATORY env variables:
    API_BASE_URL   The API endpoint for the LLM
    MODEL_NAME     The model identifier
    HF_TOKEN       Your Hugging Face API key

STDOUT FORMAT:
    [START] task=<task_id> env=sql-repair-env model=<model>
    [STEP]  step=<n> action=<query> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import sys
from typing import List, Optional
from openai import OpenAI
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL", "https://iambhavyaa-sql-repair-env.hf.space")

BENCHMARK = "sql-repair-env"
TASK_IDS  = ["task_easy", "task_medium", "task_hard", "task_expert", "task_adversarial", "task_business"]
MAX_STEPS = 5
SUCCESS_THRESHOLD = 0.8


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    action_safe = str(action).replace("\n", " ")[:120]
    print(f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


SYSTEM_PROMPT = """You are an expert SQL debugger. Fix the broken SQL query to satisfy the objective exactly.
Return ONLY the corrected SQL — no markdown, no code fences, no explanation."""

def build_prompt(obs: dict) -> str:
    parts = [
        f"## Schema\n{obs['schema_sql']}",
        f"## Sample data\n{obs['sample_data']}",
        f"## Objective\n{obs['objective']}",
        f"## Broken query\n{obs['broken_query']}",
    ]
    if obs.get("last_query"):
        parts.append(f"## Previous attempt\n{obs['last_query']}")
    if obs.get("last_execution_result"):
        parts.append(f"## Result\n{obs['last_execution_result']}")
    parts.append("Return the corrected SQL:")
    return "\n\n".join(parts)


def run_episode(client: OpenAI, task_id: str) -> dict:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task_id, BENCHMARK, MODEL_NAME)

    try:
        resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Handle both direct observation and wrapped response
        obs = data.get("observation", data)
        done = data.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": build_prompt(obs)},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
                query = (completion.choices[0].message.content or "").strip()
                for fence in ["```sql", "```SQL", "```"]:
                    if query.startswith(fence):
                        query = query[len(fence):]
                if query.endswith("```"):
                    query = query[:-3]
                query = query.strip()
            except Exception as e:
                query = "SELECT 1"

            error = None
            reward = 0.001
            try:
                step_resp = requests.post(f"{ENV_URL}/step", json={"query": query}, timeout=30)
                step_resp.raise_for_status()
                result = step_resp.json()
                # Handle both flat and nested reward
                reward_data = result.get("reward", {})
                if isinstance(reward_data, dict):
                    reward = reward_data.get("total", 0.001)
                else:
                    reward = float(reward_data) if reward_data else 0.001
                done = result.get("done", False)
                obs_data = result.get("observation", result)
                obs = obs_data if isinstance(obs_data, dict) else obs
                if result.get("info", {}).get("execution_error"):
                    error = result["info"]["execution_error"]
            except Exception as e:
                error = str(e)
                done = True

            rewards.append(reward)
            steps_taken = step
            log_step(step, query, reward, done, error)

        score = max(rewards) if rewards else 0.001
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        if not rewards:
            rewards = [0.001]

    log_end(success, steps_taken, score, rewards)
    return {"task_id": task_id, "score": score, "success": success}


def main():
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN not set.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    results = []

    for task_id in TASK_IDS:
        result = run_episode(client, task_id)
        results.append(result)

    avg = sum(r["score"] for r in results) / len(results)
    print(f"\n[SUMMARY] average_score={avg:.3f} tasks={len(results)}", flush=True)


if __name__ == "__main__":
    main()
