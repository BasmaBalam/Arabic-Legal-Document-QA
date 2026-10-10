from enum import Enum


class ResponseStatus(Enum):
    """Defines standardized response status values for application operations."""

    HEALTH_SUCCESS = "health_success"
    HEALTH_FAILED = "health_failed"

    PROCESSING_SUCCESS = "processing_success"
    PROCESSING_FAILED = "processing_failed"

    CHUNKING_SUCCESS = "chunking_success"
    CHUNKING_FAILED = "chunking_failed"

    INDEXING_SUCCESS = "indexing_success"
    INDEXING_FAILED = "indexing_failed"

    ANSWERING_SUCCESS = "answering_success"
    ANSWERING_FAILED = "answering_failed"

    EVALUATION_SUCCESS = "evaluation_success"
    EVALUATION_FAILED = "evaluation_failed"
