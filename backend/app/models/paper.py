from enum import Enum as PyEnum
from datetime import datetime
from sqlalchemy import String, Text, ForeignKey, Enum, JSON, Integer, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class PaperStatus(str, PyEnum):
    UPLOADED = 'uploaded'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'


class Paper(Base, TimestampMixin):
    __tablename__ = 'papers'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    org_id: Mapped[int] = mapped_column(ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False, index=True)
    uploaded_by_id: Mapped[int] = mapped_column(ForeignKey('users.id', ondelete='SET NULL'), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    doi: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    authors: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    source: Mapped[str | None] = mapped_column(String(200), nullable=True)
    status: Mapped[PaperStatus] = mapped_column(Enum(PaperStatus), default=PaperStatus.UPLOADED, nullable=False, index=True)
    minio_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    total_figures: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    flagged_figures: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    overall_similarity: Mapped[float | None] = mapped_column(nullable=True)
    risk_level: Mapped[str | None] = mapped_column(String(50), nullable=True)
    processing_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    organization: Mapped['Organization'] = relationship(back_populates='papers')
    uploaded_by: Mapped['User'] = relationship(back_populates='papers')
    figures: Mapped[list['Figure']] = relationship(back_populates='paper', lazy='dynamic', cascade='all, delete-orphan')
    reports: Mapped[list['Report']] = relationship(back_populates='paper', lazy='dynamic', cascade='all, delete-orphan')
    jobs: Mapped[list['Job']] = relationship(back_populates='paper', lazy='dynamic', cascade='all, delete-orphan')

    __table_args__ = (
        Index('ix_papers_org_status', 'org_id', 'status'),
        Index('ix_papers_org_created', 'org_id', 'created_at'),
    )
