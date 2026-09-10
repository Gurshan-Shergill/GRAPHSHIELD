from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.security import get_current_user_id, get_current_user_org_id
from app.core.database import get_session
from app.models import Organization, User, Paper, Report, Figure, UserRole
from app.schemas.organization import OrganizationCreate, OrganizationResponse, OrganizationSettingsUpdate, OrganizationStats

settings = get_settings()
router = APIRouter(prefix='/organizations', tags=['Organizations'])


@router.post('', response_model=OrganizationResponse, status_code=status.HTTP_201_CREATED)
async def create_organization(
    org_data: OrganizationCreate,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_session),
):
    '''Create new organization (superadmin only in production)'''
    # Check if user is superadmin (simplified for now)
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user or user.role != UserRole.ORG_ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Not authorized')

    # Check slug
    existing = await db.execute(select(Organization).where(Organization.slug == org_data.slug))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Slug already taken')

    org = Organization(
        name=org_data.name,
        slug=org_data.slug,
    )
    db.add(org)
    await db.flush()

    # Create admin user
    from app.core.security import get_password_hash
    admin = User(
        org_id=org.id,
        email=org_data.admin_email,
        hashed_password=get_password_hash(org_data.admin_password),
        role=UserRole.ORG_ADMIN,
    )
    db.add(admin)
    await db.commit()
    await db.refresh(org)

    return OrganizationResponse.model_validate(org)


@router.get('/me', response_model=OrganizationResponse)
async def get_my_organization(
    org_id: int = Depends(get_current_user_org_id),
    db: AsyncSession = Depends(get_session),
):
    '''Get current user organization'''
    result = await db.execute(select(Organization).where(Organization.id == org_id))
    org = result.scalar_one_or_none()
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Organization not found')
    return OrganizationResponse.model_validate(org)


@router.patch('/me', response_model=OrganizationResponse)
async def update_organization_settings(
    settings_update: OrganizationSettingsUpdate,
    org_id: int = Depends(get_current_user_org_id),
    db: AsyncSession = Depends(get_session),
):
    '''Update organization settings'''
    result = await db.execute(select(Organization).where(Organization.id == org_id))
    org = result.scalar_one_or_none()
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Organization not found')

    update_data = settings_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if hasattr(org, key):
            setattr(org, key, value)
        else:
            # Store in settings JSON
            org.settings[key] = value

    await db.commit()
    await db.refresh(org)
    return OrganizationResponse.model_validate(org)


@router.get('/me/stats', response_model=OrganizationStats)
async def get_organization_stats(
    org_id: int = Depends(get_current_user_org_id),
    db: AsyncSession = Depends(get_session),
):
    '''Get organization statistics'''
    # Total papers
    total_papers = await db.scalar(select(func.count(Paper.id)).where(Paper.org_id == org_id))
    total_figures = await db.scalar(select(func.count(Figure.id)).join(Paper).where(Paper.org_id == org_id))
    total_users = await db.scalar(select(func.count(User.id)).where(User.org_id == org_id))
    total_reports = await db.scalar(select(func.count(Report.id)).join(Paper).where(Paper.org_id == org_id))

    # Storage used (approximate)
    from app.core.storage import storage
    storage_used = 0  # TODO: Calculate from MinIO

    # Papers this month
    from datetime import datetime, timedelta
    month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    papers_this_month = await db.scalar(
        select(func.count(Paper.id)).where(Paper.org_id == org_id, Paper.created_at >= month_start)
    )

    # Average similarity
    avg_similarity_result = await db.execute(
        select(func.avg(Paper.overall_similarity)).where(Paper.org_id == org_id, Paper.overall_similarity.is_not(None))
    )
    avg_similarity = avg_similarity_result.scalar() or 0.0

    # Flagged papers (similarity > 25%)
    flagged_papers = await db.scalar(
        select(func.count(Paper.id)).where(Paper.org_id == org_id, Paper.overall_similarity > 25.0)
    )

    return OrganizationStats(
        total_papers=total_papers or 0,
        total_figures=total_figures or 0,
        total_users=total_users or 0,
        total_reports=total_reports or 0,
        storage_used_mb=storage_used,
        papers_this_month=papers_this_month or 0,
        avg_similarity=round(avg_similarity, 1),
        flagged_papers=flagged_papers or 0,
    )
