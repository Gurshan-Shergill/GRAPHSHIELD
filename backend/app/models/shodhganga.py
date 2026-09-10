from sqlalchemy import String, Text, ForeignKey, JSON, ARRAY, Index
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from app.models.base import Base, TimestampMixin


class ShodhgangaThesis(Base, TimestampMixin):
    __tablename__ = 'shodhganga_theses'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    thesis_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    author: Mapped[str] = mapped_column(String(255), nullable=False)
    year: Mapped[int] = mapped_column(nullable=False)
    university: Mapped[str] = mapped_column(String(255), nullable=False)
    department: Mapped[str | None] = mapped_column(String(255), nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    pdf_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    figure_embeddings: Mapped[list[list[float]] | None] = mapped_column(ARRAY(Vector(512)), nullable=True)
    figure_count: Mapped[int] = mapped_column(default=0, nullable=False)
    metadata: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    indexed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    __table_args__ = (
        Index('ix_shodhganga_university_year', 'university', 'year'),
        Index('ix_shodhganga_author', 'author'),
    )

from datetime import datetime
