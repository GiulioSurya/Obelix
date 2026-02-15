# src/domain/model/usage.py
from pydantic import BaseModel, Field


class Usage(BaseModel):
    """Model for tracking LLM token usage"""
    input_tokens: int = Field(..., ge=0, description="Number of input tokens (prompt)")
    output_tokens: int = Field(..., ge=0, description="Number of output tokens (completion)")
    total_tokens: int = Field(..., ge=0, description="Total number of tokens used")

    class Config:
        # Strict validation for assignment
        validate_assignment = True


class AgentUsage(BaseModel):
    """Accumulates total token usage for an agent"""
    model_id: str = Field(..., description="ID of the LLM model")
    total_input_tokens: int = Field(default=0, ge=0, description="Sum of all input tokens")
    total_output_tokens: int = Field(default=0, ge=0, description="Sum of all output tokens")
    total_tokens: int = Field(default=0, ge=0, description="Total sum of all tokens")
    call_count: int = Field(default=0, ge=0, description="Number of LLM calls made")

    class Config:
        # Strict validation for assignment
        validate_assignment = True

    def add_usage(self, usage: Usage) -> None:
        """
        Add tokens from a new LLM call to the total accumulation

        Args:
            usage: Usage object with tokens from the single call
        """
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_tokens += usage.total_tokens
        self.call_count += 1

    def reset(self) -> None:
        """Reset counters to zero"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
