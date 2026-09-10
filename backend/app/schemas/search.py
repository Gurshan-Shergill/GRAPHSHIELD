from typing import Optional, List
from pydantic import BaseModel, Field


class FigureMatchResult(BaseModel):
    figure_id: int
    paper_id: int
    paper_title: str
    similarity_score: float
    match_type: str
    page_num: int
    minio_path: Optional[str] = None
    details: dict = {}


class FigureSearchRequest(BaseModel):
    # For multipart/form-data, file is handled separately
    scopes: List[str] = Field(default=['my_org', 'all_opted_in', 'shodhganga'])
    top_k: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.70, ge=0.0, le=1.0)


class FigureSearchResponse(BaseModel):
    query_figure_id: Optional[int] = None
    matches: List[FigureMatchResult]
    total_matches: int
    search_time_ms: float
