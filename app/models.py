from pydantic import BaseModel, Field
from typing import Any, Optional


#action:

class Action(BaseModel):
    """
    The agent submits a repaired SQL query as its action.
    Optionally includes a free-text explanation (used for info only, not graded).
    """
    query: str = Field(..., description="The repaired SQL query the agent proposes")
    explanation: Optional[str] = Field(
        None, description="Optional explanation of what was wrong and what was fixed"
    )


#observation:

class Observation(BaseModel):
    """
    Everything the agent sees at each step.
    Includes the schema, the broken query, and execution feedback from the last action.
    """
    task_id: str = Field(..., description="Unique identifier for the current task")
    difficulty: str = Field(..., description="easy | medium | hard")
    schema_sql: str = Field(..., description="CREATE TABLE statements defining the database")
    sample_data: str = Field(..., description="INSERT statements with a small sample of rows")
    broken_query: str = Field(..., description="The original broken SQL query to fix")
    objective: str = Field(..., description="Natural language description of what the query should return")
    last_action: Optional[str] = Field(None, description="The query submitted in the previous step")
    last_execution_result: Optional[str] = Field(
        None, description="Output or error from executing the last submitted query"
    )
    last_reward: Optional[float] = Field(None, description="Reward received for the last action")
    step_number: int = Field(0, description="Current step within the episode")
    max_steps: int = Field(5, description="Maximum steps allowed per episode")


#reward:

class Reward(BaseModel):
    """
    Shaped reward breakdown — partial credit at every step so the agent
    gets gradient signal even before solving the task completely.
    """
    total: float = Field(..., description="Total reward for this step (0.0 – 1.0)")
    syntax_valid: bool = Field(..., description="True if the query parsed without error")
    executes: bool = Field(..., description="True if the query ran without runtime error")
    row_count_correct: bool = Field(..., description="True if result has the expected number of rows")
    exact_match: bool = Field(..., description="True if result matches ground truth exactly")
    breakdown: dict[str, float] = Field(
        ..., description="Score contribution of each component"
    )


#state (internal, returned by /state endpoint)

class EnvState(BaseModel):
    task_id: str
    difficulty: str
    step_number: int
    max_steps: int
    done: bool
    best_reward_so_far: float
    current_query: Optional[str]
