from sqlalchemy import String, ForeignKey, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class WebhookEndpoint(Base, TimestampMixin):
    __tablename__ = 'webhook_endpoints'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    org_id: Mapped[int] = mapped_column(ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False, index=True)
    url: Mapped[str] = mapped_column(String(500), nullable=False)
    events: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    secret: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    failure_count: Mapped[int] = mapped_column(default=0, nullable=False)
    last_triggered_at: Mapped[datetime | None] = mapped_column(nullable=True)
    last_success_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Relationships
    organization: Mapped['Organization'] = relationship(back_populates='webhooks')

    __table_args__ = (
        Index('ix_webhooks_org_active', 'org_id', 'is_active'),
    )

from datetime import datetime
