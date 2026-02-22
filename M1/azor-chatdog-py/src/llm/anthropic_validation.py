from pydantic import BaseModel, Field, validator
from typing import Optional, Literal


class AnthropicConfig(BaseModel):
    engine: Literal["ANTHROPIC"] = Field(default="ANTHROPIC")
    model_name: str = Field(..., description="Nazwa modelu Anthropic")
    anthropic_api_key: str = Field(..., min_length=1, description="Klucz API Anthropic")
    max_tokens: int = Field(default=4096, ge=1, description="Maksymalna liczba tokenów odpowiedzi")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Temperature (0.0–1.0)")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Top P / nucleus sampling (0.0–1.0)")
    top_k: Optional[int] = Field(default=None, ge=1, description="Top K (≥ 1)")

    @validator('anthropic_api_key')
    def validate_api_key(cls, v):
        if not v or v.strip() == "":
            raise ValueError("ANTHROPIC_API_KEY nie może być pusty")
        return v.strip()

    @validator('model_name')
    def validate_model_name(cls, v):
        if not v or v.strip() == "":
            raise ValueError("ANTHROPIC_MODEL_NAME nie może być pusty")
        return v.strip()


