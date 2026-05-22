from pydantic import BaseModel, Field
from typing import Any, Optional


class InputSchema(BaseModel):
    input: float = Field(..., description="The input value (int or float)")
