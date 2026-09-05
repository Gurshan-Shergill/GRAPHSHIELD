from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "GraphShield Plagiarism Engine"
    app_version: str = "2.4.0"
    debug: bool = Field(default=False, alias="DEBUG")
    api_prefix: str = "/api/v1"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Security
    api_keys: list[str] = Field(default_factory=list, alias="API_KEYS")
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    # Database
    use_cloud_sql: bool = Field(default=False, alias="USE_CLOUD_SQL")
    cloud_sql_instance: Optional[str] = Field(default=None, alias="CLOUD_SQL_INSTANCE")
    db_user: str = Field(default="postgres", alias="DB_USER")
    db_pass: str = Field(default="", alias="DB_PASS")
    db_name: str = Field(default="graphshield", alias="DB_NAME")
    private_ip: bool = Field(default=False, alias="PRIVATE_IP")
    google_application_credentials: Optional[str] = Field(default=None, alias="GOOGLE_APPLICATION_CREDENTIALS")

    # SQLite fallback
    sqlite_path: str = "graphshield.db"

    # Matching thresholds
    phash_threshold: int = 12
    dhash_threshold: int = 12
    hybrid_threshold: int = 12
    text_similarity_threshold: float = 0.75

    # External services
    google_vision_api_key: Optional[str] = Field(default=None, alias="GOOGLE_VISION_API_KEY")
    enable_external_text_check: bool = Field(default=False, alias="ENABLE_EXTERNAL_TEXT_CHECK")

    # Storage
    exports_dir: str = "exports"
    max_file_size_mb: int = 50
    allowed_extensions: list[str] = Field(default_factory=lambda: [".pdf"])

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    @property
    def api_key_set(self) -> set[str]:
        return set(self.api_keys) if self.api_keys else set()


@lru_cache
def get_settings() -> Settings:
    return Settings()