from enum import Enum as PyEnum
from sqlalchemy import String, Text, ForeignKey, JSON, Enum, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class ReportStatus(str, PyEnum):
    PENDING = 'pending'
    GENERATING = 'generating'
    COMPLETED = 'completed'
    FAILED = 'failed'


class Report(Base, TimestampMixin):
    __tablename__ = 'reports'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey('papers.id', ondelete='CASCADE'), nullable=False, index=True)
    status: Mapped[ReportStatus] = mapped_column(Enum(ReportStatus), default=ReportStatus.PENDING, nullable=False)
    summary: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    report_minio_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    deplagiarized_minio_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Relationships
    paper: Mapped['Paper'] = relationship(back_populates='reports')

    __table_args__ = (
        Index('ix_reports_paper_status', 'paper_id', 'status'),
    )

# Import datetime
from datetime import datetime
