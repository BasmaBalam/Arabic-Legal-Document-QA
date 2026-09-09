from pydantic_settings import SettingsConfigDict, BaseSettings


class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    MAX_CIVIL_CODE_ARTICLE_NUMBER: int
    MAX_CHAR_LENGTH_THRESHOLD: int
    GENERATION_BACKEND: str
    EMBEDDING_BACKEND: str

    GENERATION_MODEL_ID: str
    EMBEDDING_MODEL_ID: str
    EMBEDDING_MODEL_SIZE: int

    DEFAULT_INPUT_MAX_CHARACTERS: int,
    DEFAULT_GENERATION_MAX_OUTPUT_TOKENS: int,
    DEFAULT_GENERATION_TEMPERATURE: int,
    ENABLE_THINKING: bool

    VECTOR_DB_BACKEND: str
    VECTOR_DB_PATH: str
    VECTOR_DB_DISTANCE_METHOD: str

    
    model_config = SettingsConfigDict(
        env_file="src/arabic_legal_document_qa/.env",
        env_file_encoding="utf-8",
    )


def get_settings():
    return Settings()