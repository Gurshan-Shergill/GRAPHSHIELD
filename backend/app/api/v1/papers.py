import os
import uuid
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.security import get_current_user_org_id, verify_api_key_dependency
from app.core.database import get_session
from app.core.celery_app import celery_app
from app.models import Paper, PaperStatus, Report, Job, JobStatus
from app.schemas.paper import (
    PaperUploadResponse, PaperStatusResponse, PaperResultsResponse,
    PaperListResponse, PaperResponse
)
from app.workers.tasks import process_paper_job

settings = get_settings()
router = APIRouter(prefix='/papers', tags=['Papers'])


@router.post('/upload', response_model=PaperUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_paper(
    file: UploadFile = File(..., description='PDF file to scan'),
    org_id: int = Depends(get_current_user_org_id),
    api_key: str = Depends(verify_api_key_dependency),
    db: AsyncSession = Depends(get_session),
):
    '''Upload PDF for plagiarism detection - returns job_id for polling'''
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Only PDF files allowed')

    # Read file content
    content = await file.read()
    if len(content) > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f'File too large. Max {settings.max_file_size_mb}MB')

    # Create paper record
    paper = Paper(
        org_id=org_id,
        title=file.filename.replace('.pdf', ''),
        status=PaperStatus.UPLOADED,
    )
    db.add(paper)
    await db.flush()

    # Create job record
    job = Job(
        paper_id=paper.id,
        org_id=org_id,
        status=JobStatus.QUEUED,
        current_step=None,
    )
    db.add(job)
    await db.commit()
    await db.refresh(paper)
    await db.refresh(job)

    # Upload PDF to MinIO
    from app.core.storage import storage
    minio_path = f'orgs/{org_id}/papers/{paper.id}/{file.filename}'
    storage.upload_bytes(minio_path, content, 'application/pdf')
    paper.minio_path = minio_path
    await db.commit()

    # Queue async processing job
    task = process_paper_job.delay(str(job.id), paper.id, content, {})
    job.celery_task_id = task.id
    job.status = JobStatus.PROCESSING
    job.started_at = datetime.utcnow()
    paper.status = PaperStatus.PROCESSING
    await db.commit()

    return PaperUploadResponse(job_id=str(job.id), paper_id=paper.id)


@router.get('', response_model=PaperListResponse)
async def list_papers(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None),
    org_id: int = Depends(get_current_user_org_id),
    db: AsyncSession = Depends(get_session),
):
    '''List papers for organization with pagination'''
    query = select(Paper).where(Paper.org_id == org_id)
    if status_filter:
        query = query.where(Paper.status == status_filter)
    query = query.order_by(desc(Paper.created_at))

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    # Paginate
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    papers = result.scalars().all()

    return PaperListResponse(
        papers=[PaperResponse.model_validate(p) for p in papers],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get('/{paper_id}', response_model=PaperResponse)
async def get_paper(
    paper_id: int,
    org_id: int = Depends(get_current_user_org_id),
    db: AsyncSession = Depends(get_session),
):
    '''Get paper details'''
    result = await db.execute(select(Paper).where(Paper.id == paper_id, Paper.org_id == org_id))
    paper = result.scalar_one_or_none()
    if not paper:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Paper not found')
    return PaperResponse.model_validate(paper)


@router.get('/{paper_id}/status', response_model=PaperStatusResponse)
async def get_paper_status(
    paper_id: int,
    org_id: int = Depends(get_current_user_org_id),
    db: AsyncSession = Depends(get_session),
):
    '''Get processing status for paper'''
    result = await db.execute(select(Paper).where(Paper.id == paper_id, Paper.org_id == org_id))
    paper = result.scalar_one_or_none()
    if not paper:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Paper not found')

    # Get latest job
    job_result = await db.execute(
        select(Job).where(Job.paper_id == paper_id).order_by(desc(Job.created_at))
    )
    job = job_result.scalar_one_or_none()

    return PaperStatusResponse(
        paper_id=paper.id,
        job_id=str(job.id) if job else None,
        status=paper.status.value,
        progress=job.progress if job else 0,
        current_step=job.current_step.value if job and job.current_step else None,
        error=job.error if job else None,
        result=job.result if job else {},
    )


@router.get('/{paper_id}/results', response_model=PaperResultsResponse)
async def get_paper_results(
    paper_id: int,
    org_id: int = Depends(get_current_user_org_id),
    db: AsyncSession = Depends(get_session),
):
    '''Get detailed similarity results for paper'''
    result = await db.execute(select(Paper).where(Paper.id == paper_id, Paper.org_id == org_id))
    paper = result.scalar_one_or_none()
    if not paper:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Paper not found')

    if paper.status != PaperStatus.COMPLETED:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Paper not yet processed')

    # Get figures with matches
    from app.models import Figure, SimilarityMatch, MatchType
    fig_result = await db.execute(select(Figure).where(Figure.paper_id == paper_id).order_by(Figure.page_num))
    figures = fig_result.scalars().all()

    figure_details = []
    for fig in figures:
        match_result = await db.execute(
            select(SimilarityMatch)
            .where(SimilarityMatch.query_figure_id == fig.id)
            .order_by(desc(SimilarityMatch.similarity_score))
            .limit(1)
        )
        best_match = match_result.scalar_one_or_none()

        match_detail = FigureMatchDetail(
            index=fig.id,
            page=fig.page_num,
            similarity=best_match.similarity_score if best_match else 0.0,
            match_type=best_match.match_type.value if best_match else 'unique',
            source=best_match.details.get('source', 'Original') if best_match else 'Original',
            matched_url=best_match.details.get('url') if best_match else None,
            flags=best_match.details.get('flags', []) if best_match else [],
        )
        figure_details.append(match_detail)

    # Count findings
    from app.models import Report
    report_result = await db.execute(select(Report).where(Report.paper_id == paper_id))
    report = report_result.scalar_one_or_none()

    return PaperResultsResponse(
        paper_id=paper.id,
        title=paper.title,
        overall_similarity=paper.overall_similarity or 0.0,
        risk_level=paper.risk_level or 'LOW',
        total_figures=paper.total_figures,
        flagged_figures=paper.flagged_figures,
        figures=figure_details,
        manipulation_findings=report.summary.get('manipulation_findings', 0) if report else 0,
        graph_semantics_findings=report.summary.get('graph_semantics_findings', 0) if report else 0,
        validity_issues=report.summary.get('validity_issues', 0) if report else 0,
        ambiguity_findings=report.summary.get('ambiguity_findings', 0) if report else 0,
        created_at=paper.created_at,
    )


@router.get('/{paper_id}/report')
async def download_report(
    paper_id: int,
    org_id: int = Depends(get_current_user_org_id),
    db: AsyncSession = Depends(get_session),
):
    '''Download UGC-compliant PDF report'''
    from fastapi.responses import StreamingResponse
    result = await db.execute(select(Report).where(Report.paper_id == paper_id))
    report = result.scalar_one_or_none()
    if not report or not report.report_minio_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Report not found')

    from app.core.storage import storage
    try:
        file_stream = storage.download_stream(report.report_minio_path.replace(f's3://{storage.bucket}/', ''))
        return StreamingResponse(
            file_stream,
            media_type='application/pdf',
            headers={'Content-Disposition': f'attachment; filename= GraphShield_Report_ '}
        )
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Failed to download report')


@router.get('/{paper_id}/deplagiarized')
async def download_deplagiarized(
    paper_id: int,
    org_id: int = Depends(get_current_user_org_id),
    db: AsyncSession = Depends(get_session),
):
    '''Download deplagiarized PDF (flagged figures removed)'''
    from fastapi.responses import StreamingResponse
    result = await db.execute(select(Report).where(Report.paper_id == paper_id))
    report = result.scalar_one_or_none()
    if not report or not report.deplagiarized_minio_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Deplagiarized PDF not found')

    from app.core.storage import storage
    try:
        file_stream = storage.download_stream(report.deplagiarized_minio_path.replace(f's3://{storage.bucket}/', ''))
        return StreamingResponse(
            file_stream,
            media_type='application/pdf',
            headers={'Content-Disposition': f'attachment; filename= Deplagiarized_ '}
        )
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Failed to download deplagiarized PDF')
