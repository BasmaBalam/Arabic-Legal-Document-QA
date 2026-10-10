from enum import Enum


class LLMEnum(Enum):
    QWEN = "QWEN"
    COHERE = "COHERE"
    GEMINI = "GEMINI"
    OLLAMA = "OLLAMA"


class CohereEnums(Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "CHATBOT"

    DOCUMENT = "search_document"
    QUERY = "search_query"


class QwenEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class GeminiEnums(Enum):
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"


class DocumentTypeEnum(Enum):
    QUERY = "query"
    DOCUMENT = "document"
    PASSAGE = "passage"
