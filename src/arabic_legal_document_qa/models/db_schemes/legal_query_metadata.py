from pydantic import BaseModel, field_validator


class LegalQueryMetadata(BaseModel):
    book: str | None = None
    chapter: str | None = None
    section: str | None = None
    topic: str | None = None
    article_number: int | None = None


class ExtractedLegalMetadata(BaseModel):
    metadata: list[LegalQueryMetadata]

    @field_validator("metadata", mode="before")
    @classmethod
    def remove_duplicates(cls, metadata):
        unique_metadata = []
        seen = set()

        for item in metadata:
            item_tuple = tuple(sorted(item.items()))

            if item_tuple not in seen:
                seen.add(item_tuple)
                unique_metadata.append(dict(item_tuple))

        return unique_metadata
