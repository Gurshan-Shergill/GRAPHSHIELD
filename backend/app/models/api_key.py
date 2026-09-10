from datetime import datetime
from sqlalchemy import String, ForeignKey, Integer, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class APIKey(Base, TimestampMixin):
    __tablename__ = 'api_keys'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    org_id: Mapped[int] = mapped_column(ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    scopes: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    rate_limit: Mapped[int] = mapped_column(Integer, default=100, nullable=False)
    rate_limit_window: Mapped[int] = mapped_column(Integer, default=60, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)

    # Relationships
    organization: Mapped['Organization'] = relationship(back_populates='api_keys')

    __table_args__ = (
        Index('ix_api_keys_org_active', 'org_id', 'is_active'),
    )
