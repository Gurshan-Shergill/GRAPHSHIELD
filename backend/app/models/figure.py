from sqlalchemy import String, Text, ForeignKey, Integer, Float, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.models.base import Base, TimestampMixin


class Figure(Base, TimestampMixin):
    __tablename__ = 'figures'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey('papers.id', ondelete='CASCADE'), nullable=False, index=True)
    page_num: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    minio_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    phash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    dhash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    clip_embedding: Mapped[list[float] | None] = mapped_column(Vector(512), nullable=True)
    manipulation_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ocr_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    figure_type: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Relationships
    paper: Mapped['Paper'] = relationship(back_populates='figures')
    matches_as_query: Mapped[list['SimilarityMatch']] = relationship(
        back_populates='query_figure',
        foreign_keys='SimilarityMatch.query_figure_id',
        lazy='dynamic'
    )
    matches_as_matched: Mapped[list['SimilarityMatch']] = relationship(
        back_populates='matched_figure',
        foreign_keys='SimilarityMatch.matched_figure_id',
        lazy='dynamic'
    )

    __table_args__ = (
        Index('ix_figures_paper_page', 'paper_id', 'page_num'),
        Index('ix_figures_phash_paper', 'phash', 'paper_id'),
    )
