from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel


class JobCreate(BaseModel):
    paper_id: int
    options: Dict[str, Any] = {}


class JobResponse(BaseModel):
    id: int
    paper_id: int
    org_id: int
    status: str
    progress: int
    current_step: Optional[str] = None
    error: Optional[str] = None
    result: Dict[str, Any] = {}
    celery_task_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True
