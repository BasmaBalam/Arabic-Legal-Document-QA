from typing import Any

from pydantic import BaseModel


class RetrievedDocument(BaseModel):
    text: str
    score: float
    metadata: dict[str, Any] | None = None
