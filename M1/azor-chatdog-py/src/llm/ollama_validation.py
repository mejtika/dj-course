from pydantic import BaseModel, Field, validator
from typing import Literal


class OllamaConfig(BaseModel):
    engine: Literal["OLLAMA"] = Field(default="OLLAMA")
    model_name: str = Field(..., description="Nazwa modelu Ollama")
    ollama_host: str = Field(default="http://localhost:11434", description="Adres hosta Ollama")

    @validator('model_name')
    def validate_model_name(cls, v):
        if not v or v.strip() == "":
            raise ValueError("OLLAMA_MODEL_NAME nie może być pusty")
        return v.strip()

    @validator('ollama_host')
    def validate_host(cls, v):
        if not v or v.strip() == "":
            raise ValueError("OLLAMA_HOST nie może być pusty")
        if not v.startswith("http://") and not v.startswith("https://"):
            raise ValueError("OLLAMA_HOST musi zaczynać się od http:// lub https://")
        return v.strip().rstrip('/')


