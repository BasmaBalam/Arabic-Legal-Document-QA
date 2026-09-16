from pydantic import BaseModel, Field


class ProcessRequest(BaseModel):
    chunk_size: int | None = 1000
    overlap_size: int | None = 20


class IndexRequest(BaseModel):
    chunk_size: int = Field(
        default=1000, description="Character size for chunking documents", gt=0
    )
    overlap_size: int = Field(
        default=100, description="Overlap size between chunks", ge=0
    )
    do_reset: bool = Field(
        default=False, description="Reset collection before indexing"
    )
