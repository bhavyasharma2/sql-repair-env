"""
client.py — SQL Repair Env typed client.

Install from HF Space:
    pip install git+https://huggingface.co/spaces/iambhavyaa/sql-repair-env

Usage (async):
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

Usage (sync):
    from sql_repair_env import SQLRepairEnv, SQLRepairAction

    with SQLRepairEnv(base_url="https://iambhavyaa-sql-repair-env.hf.space").sync() as env:
        result = env.reset(task_id="task_medium")
        result = env.step(SQLRepairAction(query="SELECT ..."))
        print(result.reward)
"""

try:
    from openenv.core.env_client import EnvClient
    from .models import SQLRepairAction, SQLRepairObservation, SQLRepairState
except ImportError:
    from openenv.core.env_client import EnvClient
    from models import SQLRepairAction, SQLRepairObservation, SQLRepairState


class SQLRepairEnv(EnvClient[SQLRepairAction, SQLRepairObservation]):
    """
    Typed client for the SQL Repair OpenEnv environment.

    Connects via WebSocket for persistent sessions (efficient multi-step).
    Falls back to HTTP for stateless use cases.
    """

    action_type = SQLRepairAction
    observation_type = SQLRepairObservation
    state_type = SQLRepairState
