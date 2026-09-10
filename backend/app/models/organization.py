from sqlalchemy import String, Text, Boolean, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Organization(Base, TimestampMixin):
    __tablename__ = 'organizations'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    logo_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    settings: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    repo_opt_in: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    shodhganga_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Relationships
    users: Mapped[list['User']] = relationship(back_populates='organization', lazy='dynamic')
    papers: Mapped[list['Paper']] = relationship(back_populates='organization', lazy='dynamic')
    api_keys: Mapped[list['APIKey']] = relationship(back_populates='organization', lazy='dynamic')
    webhooks: Mapped[list['WebhookEndpoint']] = relationship(back_populates='organization', lazy='dynamic')

    __table_args__ = (
        Index('ix_organizations_slug_active', 'slug', 'is_active'),
    )
