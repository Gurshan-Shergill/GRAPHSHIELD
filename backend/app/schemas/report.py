from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class ReportResponse(BaseModel):
    id: int
    paper_id: int
    status: str
    summary: dict
    report_minio_path: Optional[str] = None
    deplagiarized_minio_path: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True
