from enum import Enum as PyEnum
from sqlalchemy import String, ForeignKey, Boolean, Enum, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class UserRole(str, PyEnum):
    ORG_ADMIN = 'org_admin'
    INSTRUCTOR = 'instructor'
    RESEARCHER = 'researcher'
    VIEWER = 'viewer'


class User(Base, TimestampMixin):
    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    org_id: Mapped[int] = mapped_column(ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.RESEARCHER, nullable=False)
    api_key_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_login: Mapped[datetime | None] = mapped_column(nullable=True)

    # Relationships
    organization: Mapped['Organization'] = relationship(back_populates='users')
    papers: Mapped[list['Paper']] = relationship(back_populates='uploaded_by', lazy='dynamic')

    __table_args__ = (
        Index('ix_users_org_email', 'org_id', 'email'),
    )

# Import datetime for type hint
from datetime import datetime
