from functools import lru_cache
from typing import Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import json


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
    )

    # App
    app_name: str = 'GraphShield'
    app_version: str = '3.0.0'
    debug: bool = Field(default=False, alias='DEBUG')
    api_prefix: str = '/api/v1'

    # Server
    host: str = '0.0.0.0'
    port: int = 8000
    workers: int = 1

    # Security
    secret_key: str = Field(alias='SECRET_KEY')
    jwt_algorithm: str = 'HS256'
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days: int = 7
    api_key_prefix: str = 'gs_'
    bcrypt_rounds: int = 12

    # API Keys & Rate Limiting
    api_keys: List[str] = Field(default_factory=list, alias='API_KEYS')
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    # Database
    use_cloud_sql: bool = Field(default=False, alias='USE_CLOUD_SQL')
    cloud_sql_instance: Optional[str] = Field(default=None, alias='CLOUD_SQL_INSTANCE')
    db_user: str = Field(default='postgres', alias='DB_USER')
    db_pass: str = Field(default='', alias='DB_PASS')
    db_name: str = Field(default='graphshield', alias='DB_NAME')
    private_ip: bool = Field(default=False, alias='PRIVATE_IP')
    google_application_credentials: Optional[str] = Field(default=None, alias='GOOGLE_APPLICATION_CREDENTIALS')
    database_url: Optional[str] = Field(default=None, alias='DATABASE_URL')
    sqlite_path: str = 'graphshield.db'

    # Redis / Celery
    redis_url: str = Field(default='redis://redis:6379/0', alias='REDIS_URL')
    celery_broker_url: str = Field(default='redis://redis:6379/0', alias='CELERY_BROKER_URL')
    celery_result_backend: str = Field(default='redis://redis:6379/1', alias='CELERY_RESULT_BACKEND')

    # MinIO / S3
    minio_endpoint: str = Field(default='minio:9000', alias='MINIO_ENDPOINT')
    minio_root_user: str = Field(default='minioadmin', alias='MINIO_ROOT_USER')
    minio_root_password: str = Field(default='minioadmin', alias='MINIO_ROOT_PASSWORD')
    minio_bucket: str = Field(default='graphshield', alias='MINIO_BUCKET')
    minio_secure: bool = Field(default=False, alias='MINIO_SECURE')
    s3_endpoint: Optional[str] = Field(default=None, alias='S3_ENDPOINT')
    s3_access_key: Optional[str] = Field(default=None, alias='S3_ACCESS_KEY')
    s3_secret_key: Optional[str] = Field(default=None, alias='S3_SECRET_KEY')
    s3_bucket: Optional[str] = Field(default=None, alias='S3_BUCKET')
    s3_region: str = Field(default='us-east-1', alias='S3_REGION')

    # ML Models
    clip_model: str = Field(default='ViT-B-32', alias='CLIP_MODEL')
    clip_pretrained: str = Field(default='openai', alias='CLIP_PRETRAINED')
    easyocr_languages: str = Field(default='en,hi', alias='EASYOCR_LANGUAGES')
    embedding_dim: int = Field(default=512, alias='EMBEDDING_DIM')

    # Vector Search
    pgvector_hnsw_m: int = Field(default=16, alias='PGVECTOR_HNSW_M')
    pgvector_hnsw_ef_construction: int = Field(default=64, alias='PGVECTOR_HNSW_EF_CONSTRUCTION')
    faiss_nlist: int = Field(default=100, alias='FAISS_NLIST')
    similarity_thresholds: dict = Field(default_factory=lambda: {'exact': 5, 'near_duplicate': 0.85, 'semantic': 0.70})

    # Processing
    max_file_size_mb: int = Field(default=100, alias='MAX_FILE_SIZE_MB')
    max_figures_per_paper: int = Field(default=50, alias='MAX_FIGURES_PER_PAPER')
    worker_concurrency_high: int = Field(default=2, alias='WORKER_CONCURRENCY_HIGH')
    worker_concurrency_default: int = Field(default=4, alias='WORKER_CONCURRENCY_DEFAULT')
    worker_concurrency_low: int = Field(default=4, alias='WORKER_CONCURRENCY_LOW')

    # Shodhganga
    shodhganga_sync_enabled: bool = Field(default=True, alias='SHODHGANGA_SYNC_ENABLED')
    shodhganga_sync_cron: str = Field(default='0 2 * * 0', alias='SHODHGANGA_SYNC_CRON')

    # Webhooks
    webhook_timeout_seconds: int = Field(default=10, alias='WEBHOOK_TIMEOUT_SECONDS')
    webhook_max_retries: int = Field(default=3, alias='WEBHOOK_MAX_RETRIES')

    # Logging
    log_level: str = Field(default='INFO', alias='LOG_LEVEL')
    log_format: str = Field(default='json', alias='LOG_FORMAT')

    # Exports
    exports_dir: str = Field(default='exports', alias='EXPORTS_DIR')

    @property
    def api_key_set(self) -> set:
        return set(self.api_keys) if self.api_keys else set()

    @field_validator('similarity_thresholds', mode='before')
    @classmethod
    def parse_thresholds(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
