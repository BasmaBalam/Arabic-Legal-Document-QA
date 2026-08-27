from pydantic_settings import SettingsConfigDict, BaseSettings


class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str

    model_config = SettingsConfigDict(
        env_file="arabic_legal_document_qa/.env",
        env_file_encoding="utf-8",
    )


def get_settings():
    return Settings()