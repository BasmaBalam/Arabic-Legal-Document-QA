from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Legal question, in Arabic or English.",
    )
    limit: int | None = Field(
        default=10, description="Maximum number of context chunks to retrieve", gt=0
    )

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            # min_length=1 alone lets a whitespace-only string ("   ") through —
            # this catches that case too, so both "" and "   " return 422.
            raise ValueError("question must not be empty or whitespace-only")

        return stripped


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
