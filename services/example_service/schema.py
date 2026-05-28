from pydantic import BaseModel, Field, field_validator
from typing import Any, Optional

from .config import N_TH_ROOT


class InputSchema(BaseModel):
    value: float = Field(..., description="The input value (int or float)")
    root: float = Field(default=N_TH_ROOT, description="The root to calculate (default is 2 for square root)")

    @field_validator("root")
    @classmethod
    def exclude_invalid(cls, v):
        if v == 0 or float(v) == 0:
            raise ValueError("root cannot be 0")
        return v
