from enum import Enum

class LLMEnum(Enum):
    QWEN = "QWEN"
    OPENAI = "OPENAI"

class DocumentTypeEnum(Enum):
    DOCUMENT = "document"
    QUERY = "query"

class QwenEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class DocumentTypeEnum(Enum):
    QUERY = "query"
    PASSAGE = "passage"
