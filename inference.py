"""
inference.py — SQL Repair OpenEnv Inference Script

MANDATORY env variables:
    API_BASE_URL   The API endpoint for the LLM (default: HF router)
    MODEL_NAME     The model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       Your Hugging Face API key

STDOUT FORMAT (strictly followed):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import sys
from typing import List, Optional
from openai import OpenAI
import requests

#environment variables:
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")    or os.getenv("OPENAI_API_KEY")
ENV_URL      = os.getenv("ENV_URL",      "https://iambhavyaa-sql-repair-env.hf.space")

BENCHMARK    = "sql-repair-env"
TASK_IDS     = ["task_easy", "task_medium", "task_hard", "task_expert"]
MAX_STEPS    = 5
SUCCESS_SCORE_THRESHOLD = 0.8

#logging:

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_safe = action.replace("\n", " ")[:120]
    error_val   = error if error else "null"
    done_val    = str(done).lower()
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

#prompt:

SYSTEM_PROMPT = """You are an expert SQL debugger. You will be given:
1. A database schema (CREATE TABLE statements)
2. Some sample data (INSERT statements)
3. A broken SQL query
4. The objective — what the query SHOULD return

Your job is to return a corrected SQL query that exactly satisfies the objective.

Rules:
- Return ONLY the corrected SQL query, nothing else
- No markdown, no code fences, no explanation
- The query must be valid SQL compatible with DuckDB
- Fix ALL bugs in the query, not just the most obvious one
"""

def build_prompt(obs: dict) -> str:
    parts = [
        f"## Schema\n{obs['schema_sql']}",
        f"## Sample data\n{obs['sample_data']}",
        f"## Objective\n{obs['objective']}",
        f"## Broken query to fix\n{obs['broken_query']}",
    ]
    if obs.get("last_action"):
        parts.append(f"## Your previous attempt\n{obs['last_action']}")
    if obs.get("last_execution_result"):
        parts.append(f"## Result of previous attempt\n{obs['last_execution_result']}")
    if obs.get("last_reward") is not None:
        parts.append(f"## Reward for previous attempt: {obs['last_reward']}")
    parts.append("Return only the corrected SQL query:")
    return "\n\n".join(parts)

#episode runner:

def run_episode(client: OpenAI, task_id: str) -> dict:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        #reset
        resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        obs = resp.json()
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            #get model's proposed fix
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
                #strip accidental markdown fences
                for fence in ["```sql", "```SQL", "```"]:
                    if query.startswith(fence):
                        query = query[len(fence):]
                if query.endswith("```"):
                    query = query[:-3]
                query = query.strip()
            except Exception as e:
                query = "SELECT 1"
                print(f"[DEBUG] Model call failed: {e}", flush=True)

            #submit action
            error = None
            reward = 0.0
            try:
                step_resp = requests.post(
                    f"{ENV_URL}/step",
                    json={"query": query},
                    timeout=30,
                )
                step_resp.raise_for_status()
                result = step_resp.json()
                reward  = result["reward"]["total"]
                done    = result["done"]
                obs     = result["observation"]
                if result["info"].get("execution_error"):
                    error = result["info"]["execution_error"]
            except Exception as e:
                error = str(e)
                done  = True

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=query, reward=reward, done=done, error=error)

        score   = max(rewards) if rewards else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        if not rewards:
            rewards = [0.0]

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"task_id": task_id, "score": score, "success": success}


#main:

def main():
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN or OPENAI_API_KEY not set.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    all_results = []
    for task_id in TASK_IDS:
        result = run_episode(client, task_id)
        all_results.append(result)

    avg = sum(r["score"] for r in all_results) / len(all_results)
    print(f"\n[SUMMARY] average_score={avg:.3f} tasks={len(all_results)}", flush=True)
    return all_results


if __name__ == "__main__":
    main()
