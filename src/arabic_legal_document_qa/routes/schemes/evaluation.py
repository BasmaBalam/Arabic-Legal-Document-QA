from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):
    chunk_size: int = Field(
        default=1000, description="Character size for chunking documents", gt=0
    )

    overlap_size: int = Field(
        default=100, description="Overlap size between chunks", ge=0
    )
