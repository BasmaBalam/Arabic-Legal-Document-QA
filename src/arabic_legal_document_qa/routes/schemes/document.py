from pydantic import BaseModel
from typing import Optional

class ProcessRequest(BaseModel):
    chunk_size: Optional[int] = 1000
    overlap_size: Optional[int] = 20