"""
GRPO training script for the SQL Repair environment.

Trains a small LLM to fix broken SQL queries using Group Relative Policy
Optimization (GRPO) via the TRL library.

The reward signal comes directly from our OpenEnv graders — the model
learns from the same 0.0–1.0 shaped reward used during evaluation.

Usage:
    # Train locally (requires ~16GB VRAM for Qwen-1.5B, 8GB with quantization)
    python train_grpo.py

    # Train on a specific task only
    python train_grpo.py --task task_hard

    # Push trained model to HF Hub
    python train_grpo.py --push-to-hub --hub-model-id your-org/sql-repair-grpo

Requirements:
    pip install trl>=0.12.0 transformers>=4.46.0 peft accelerate duckdb pydantic
"""

import argparse
import os
import sys
from typing import Any

# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="GRPO training for SQL Repair env")
parser.add_argument("--task", default="all", choices=["all", "task_easy", "task_medium", "task_hard"])
parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--group-size", type=int, default=4, help="Num completions per prompt (GRPO G)")
parser.add_argument("--lr", type=float, default=5e-6)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--push-to-hub", action="store_true")
parser.add_argument("--hub-model-id", default="sql-repair-grpo")
parser.add_argument("--output-dir", default="./grpo_output")
args = parser.parse_args()


# ── Imports (after arg parse so --help works without deps) ───────────────────

try:
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from app.graders import grade
    from app.tasks import TASKS
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with: pip install trl transformers datasets peft accelerate duckdb pydantic")
    sys.exit(1)


# ── Prompt template ───────────────────────────────────────────────────────────

SYSTEM = (
    "You are an expert SQL debugger. Fix the broken SQL query so it satisfies "
    "the objective exactly. Return ONLY the corrected SQL — no explanation, "
    "no markdown, no code fences."
)

def make_prompt(task: dict) -> str:
    return (
        f"## Database schema\n{task['schema_sql']}\n\n"
        f"## Sample data\n{task['seed_sql']}\n\n"
        f"## Objective\n{task['objective']}\n\n"
        f"## Broken query\n{task['broken_query']}\n\n"
        f"Return the corrected SQL query:"
    )


# ── Dataset construction ───────────────────────────────────────────────────────
# GRPO needs a prompt dataset. We expand each task into multiple slightly-varied
# prompts so the model sees enough diversity. For a real run you'd augment with
# more broken query variants.

def build_dataset(task_ids: list[str]) -> Dataset:
    """
    Build a prompt dataset from our task definitions.
    Each task becomes N prompt variants (we use the same prompt for now;
    in production you'd generate multiple broken-query variants per task).
    """
    REPEATS_PER_TASK = 50  # enough for 200-step training run

    records = []
    for task_id in task_ids:
        task = TASKS[task_id]
        prompt = make_prompt(task)
        for _ in range(REPEATS_PER_TASK):
            records.append({
                "prompt": prompt,
                "task_id": task_id,
            })

    return Dataset.from_list(records)


# ── Reward function ───────────────────────────────────────────────────────────
# GRPO calls this with a batch of (prompt, completion) pairs.
# We extract the task_id from the prompt metadata and score via our grader.

def sql_repair_reward(
    prompts: list[str],
    completions: list[str],
    task_ids: list[str],
    **kwargs,
) -> list[float]:
    """
    Reward function for GRPO.

    For each (prompt, completion) pair:
      1. Extract the proposed SQL query from the completion
      2. Run it through the environment grader
      3. Return the shaped reward (0.0–1.0)

    The shaped reward (syntax → execution → row count → exact match)
    ensures the model gets non-zero gradient signal even for partially
    correct fixes, which is critical for learning from sparse rewards.
    """
    rewards = []
    for completion, task_id in zip(completions, task_ids):
        # Strip any accidental markdown fences the model might emit
        query = _clean_sql(completion)

        try:
            result = grade(task_id=task_id, query=query)
            reward = result["score"]
        except Exception:
            reward = 0.0

        rewards.append(reward)

    return rewards


def _clean_sql(text: str) -> str:
    """Remove markdown code fences and whitespace from model output."""
    text = text.strip()
    for fence in ["```sql", "```SQL", "```"]:
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    print(f"\n{'='*60}")
    print(f"  SQL Repair — GRPO Training")
    print(f"  Model : {args.model}")
    print(f"  Tasks : {args.task}")
    print(f"  Steps : {args.steps}")
    print(f"  G     : {args.group_size} completions/prompt")
    print(f"{'='*60}\n")

    task_ids = list(TASKS.keys()) if args.task == "all" else [args.task]

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = build_dataset(task_ids)
    print(f"Dataset: {len(dataset)} prompts across {len(task_ids)} task(s)\n")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── GRPO config ───────────────────────────────────────────────────────────
    config = GRPOConfig(
        # Core GRPO
        num_generations=args.group_size,       # G: completions per prompt
        max_new_tokens=args.max_new_tokens,

        # Training
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        max_steps=args.steps,
        warmup_steps=10,

        # Logging
        logging_steps=10,
        output_dir=args.output_dir,
        report_to="none",                      # set to "wandb" if you want tracking

        # Memory
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,

        # KL penalty (keep generations close to reference policy)
        kl_coef=0.05,
    )

    # ── Reward wrapper ────────────────────────────────────────────────────────
    # TRL's GRPOTrainer passes extra dataset columns to the reward function.
    # We include task_id in the dataset so the reward fn knows which grader to call.

    def reward_fn(prompts, completions, **kwargs):
        # kwargs contains extra dataset columns — extract task_ids
        task_ids = kwargs.get("task_id", ["task_easy"] * len(completions))
        return sql_repair_reward(prompts, completions, task_ids)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=args.model,
        config=config,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting GRPO training...")
    print("Watch for reward climbing from ~0.2 (syntax) toward 1.0 (exact match)\n")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    trainer.save_model(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")

    if args.push_to_hub:
        print(f"Pushing to HF Hub: {args.hub_model_id}")
        trainer.push_to_hub(args.hub_model_id)
        print("Done.")

    # ── Quick eval after training ─────────────────────────────────────────────
    print("\n── Post-training evaluation ──")
    _quick_eval(trainer, tokenizer, task_ids)


def _quick_eval(trainer, tokenizer, task_ids: list[str]):
    """
    Run the trained model on each task and print the reward.
    Not a substitute for the full baseline script, but useful for
    a quick sanity check during development.
    """
    from transformers import pipeline
    import torch

    pipe = pipeline(
        "text-generation",
        model=trainer.model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )

    for task_id in task_ids:
        task = TASKS[task_id]
        prompt = make_prompt(task)
        output = pipe(prompt, max_new_tokens=args.max_new_tokens, do_sample=False)
        generated = output[0]["generated_text"][len(prompt):]
        query = _clean_sql(generated)

        result = grade(task_id=task_id, query=query)
        print(f"  {task_id}: score={result['score']:.3f}  exact={result['exact_match']}")
        if not result["exact_match"]:
            print(f"    Generated: {query[:120]}...")
            print(f"    Error:     {result['error']}")


if __name__ == "__main__":
    train()
