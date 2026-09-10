from enum import Enum as PyEnum
from sqlalchemy import String, ForeignKey, Integer, JSON, Enum, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class JobStatus(str, PyEnum):
    QUEUED = 'queued'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


class JobStep(str, PyEnum):
    EXTRACTION = 'extraction'
    FINGERPRINTING = 'fingerprinting'
    VECTOR_SEARCH = 'vector_search'
    MANIPULATION_DETECTION = 'manipulation_detection'
    GRAPH_SEMANTICS = 'graph_semantics'
    VALIDITY_CHECK = 'validity_check'
    AMBIGUITY_DETECTION = 'ambiguity_detection'
    REPORT_GENERATION = 'report_generation'
    INDEXING = 'indexing'


class Job(Base, TimestampMixin):
    __tablename__ = 'jobs'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey('papers.id', ondelete='CASCADE'), nullable=False, index=True)
    org_id: Mapped[int] = mapped_column(ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False, index=True)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.QUEUED, nullable=False, index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    current_step: Mapped[JobStep | None] = mapped_column(Enum(JobStep), nullable=True)
    error: Mapped[str | None] = mapped_column(nullable=True)
    result: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    celery_task_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Relationships
    paper: Mapped['Paper'] = relationship(back_populates='jobs')
    organization: Mapped['Organization'] = relationship()

    __table_args__ = (
        Index('ix_jobs_paper_status', 'paper_id', 'status'),
        Index('ix_jobs_org_status', 'org_id', 'status'),
    )

# Import datetime
from datetime import datetime
