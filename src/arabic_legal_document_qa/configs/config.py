from pydantic_settings import SettingsConfigDict, BaseSettings


class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    MAX_CIVIL_CODE_ARTICLE_NUMBER: int
    MAX_CHAR_LENGTH_THRESHOLD: int


    model_config = SettingsConfigDict(
        env_file="src/arabic_legal_document_qa/.env",
        env_file_encoding="utf-8",
    )


def get_settings():
    return Settings()