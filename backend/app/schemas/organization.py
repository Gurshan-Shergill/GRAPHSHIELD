from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class OrganizationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=100, pattern='^[a-z0-9-]+$')
    admin_email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    admin_password: str = Field(..., min_length=8, max_length=100)


class OrganizationResponse(BaseModel):
    id: int
    name: str
    slug: str
    logo_url: Optional[str] = None
    settings: Dict[str, Any] = {}
    repo_opt_in: bool
    shodhganga_enabled: bool
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class OrganizationSettingsUpdate(BaseModel):
    repo_opt_in: Optional[bool] = None
    shodhganga_enabled: Optional[bool] = None
    similarity_thresholds: Optional[Dict[str, float]] = None
    max_file_size_mb: Optional[int] = Field(default=None, ge=1, le=500)
    webhook_url: Optional[str] = None
    webhook_events: Optional[List[str]] = None


class OrganizationStats(BaseModel):
    total_papers: int
    total_figures: int
    total_users: int
    total_reports: int
    storage_used_mb: float
    papers_this_month: int
    avg_similarity: float
    flagged_papers: int
