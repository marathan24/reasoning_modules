from pydantic import BaseModel
from typing import Optional

class ReasoningInput(BaseModel):
    func_name: str
    problem: str
    num_thoughts: Optional[int] = 3  # Default to 3 thoughts if not specified

class SystemPromptSchema(BaseModel):
    role: str