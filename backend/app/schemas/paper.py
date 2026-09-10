from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class PaperUploadResponse(BaseModel):
    job_id: str
    paper_id: int
    status: str = 'queued'
    message: str = 'Paper uploaded successfully. Processing started.'


class FigureMatchDetail(BaseModel):
    index: int
    page: int
    similarity: float
    match_type: str
    source: str
    matched_url: Optional[str] = None
    flags: List[str] = []


class PaperResultsResponse(BaseModel):
    paper_id: int
    title: str
    overall_similarity: float
    risk_level: str
    total_figures: int
    flagged_figures: int
    figures: List[FigureMatchDetail]
    manipulation_findings: int
    graph_semantics_findings: int
    validity_issues: int
    ambiguity_findings: int
    created_at: datetime


class PaperStatusResponse(BaseModel):
    paper_id: int
    job_id: Optional[str] = None
    status: str
    progress: int = 0
    current_step: Optional[str] = None
    error: Optional[str] = None
    result: dict = {}


class PaperListResponse(BaseModel):
    papers: List['PaperResponse']
    total: int
    page: int
    page_size: int


class PaperResponse(BaseModel):
    id: int
    title: str
    doi: Optional[str] = None
    authors: List[str] = []
    source: Optional[str] = None
    status: str
    total_figures: int
    flagged_figures: int
    overall_similarity: Optional[float] = None
    risk_level: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


PaperListResponse.model_rebuild()
