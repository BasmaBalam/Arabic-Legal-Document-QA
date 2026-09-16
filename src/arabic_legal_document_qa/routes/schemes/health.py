from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="healthy")
    documents_indexed: int = Field(..., example=1149)
