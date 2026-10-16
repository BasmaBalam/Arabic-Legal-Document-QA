from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    MAX_CIVIL_CODE_ARTICLE_NUMBER: int
    MAX_CHAR_LENGTH_THRESHOLD: int
    GENERATION_BACKEND: str
    EMBEDDING_BACKEND: str

    COHERE_API_KEY: str
    GEMINI_API_KEY: str
    GENERATION_MODEL_ID: str
    EMBEDDING_MODEL_ID: str
    EMBEDDING_MODEL_SIZE: int

    DEFAULT_INPUT_MAX_CHARACTERS: int
    DEFAULT_GENERATION_MAX_OUTPUT_TOKENS: int
    DEFAULT_GENERATION_TEMPERATURE: float
    ENABLE_THINKING: bool

    QUANTIZED_GENERATION_MODEL_ID: str

    OLLAMA_HOST: str
    OLLAMA_KEEP_ALIVE: str | int

    VECTOR_DB_BACKEND: str
    VECTOR_DB_PATH: str
    VECTOR_DB_DISTANCE_METHOD: str

    EVALUATION_BACKEND: str
    EVALUATION_MODEL_ID: str

    EVALUATION_EMBEDDING_BACKEND: str
    EVALUATION_EMBEDDING_MODEL_ID: str
    EVALUATION_EMBEDDING_MODEL_SIZE: int

    PRIMARY_LANG: str
    DEFAULT_LANG: str

    MLFLOW_EXPERIMENT_NAME: str
    MLFLOW_TRACKING_URI: str

    VLLM_BASE_URL: str
    VLLM_API_KEY: str
    VLLM_MODEL_ID: str

    model_config = SettingsConfigDict(
        env_file="src/arabic_legal_document_qa/.env",
        env_file_encoding="utf-8",
    )


def get_settings():
    return Settings()
