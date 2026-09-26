from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    gemini_api_key: str = ""
    cohere_api_key: str = ""
    database_url: str = "postgresql://postgres:postgres@localhost:5432/bankpolicy"
    log_level: str = "INFO"


settings = Settings()