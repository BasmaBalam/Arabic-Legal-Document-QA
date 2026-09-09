from pydantic import BaseModel, Field, validator

class RetrievedDocument(BaseModel):
    text: str
    score: float