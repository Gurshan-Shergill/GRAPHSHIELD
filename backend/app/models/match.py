from enum import Enum as PyEnum
from sqlalchemy import String, ForeignKey, Float, JSON, Integer, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class MatchType(str, PyEnum):
    EXACT = 'exact'
    NEAR_DUPLICATE = 'near_duplicate'
    SEMANTIC = 'semantic'
    MANIPULATED = 'manipulated'
    INTERNAL_DUPLICATE = 'internal_duplicate'
    SHODHGANGA = 'shodhganga'
    EXTERNAL = 'external'


class SimilarityMatch(Base, TimestampMixin):
    __tablename__ = 'similarity_matches'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    query_figure_id: Mapped[int] = mapped_column(ForeignKey('figures.id', ondelete='CASCADE'), nullable=False, index=True)
    matched_figure_id: Mapped[int] = mapped_column(ForeignKey('figures.id', ondelete='CASCADE'), nullable=False, index=True)
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    match_type: Mapped[MatchType] = mapped_column(default=MatchType.SEMANTIC, nullable=False)
    details: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    # Relationships
    query_figure: Mapped['Figure'] = relationship(
        back_populates='matches_as_query',
        foreign_keys=[query_figure_id]
    )
    matched_figure: Mapped['Figure'] = relationship(
        back_populates='matches_as_matched',
        foreign_keys=[matched_figure_id]
    )

    __table_args__ = (
        Index('ix_matches_query_score', 'query_figure_id', 'similarity_score'),
        Index('ix_matches_matched_score', 'matched_figure_id', 'similarity_score'),
    )
