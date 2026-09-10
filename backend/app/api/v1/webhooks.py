from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import secrets

from app.core.config import get_settings
from app.core.security import get_current_user_org_id
from app.core.database import get_session
from app.models import WebhookEndpoint, Organization
from app.schemas.organization import OrganizationSettingsUpdate
from pydantic import BaseModel, HttpUrl

settings = get_settings()
router = APIRouter(prefix='/webhooks', tags=['Webhooks'])


class WebhookCreate(BaseModel):
    url: HttpUrl
    events: List[str] = ['paper.processed', 'report.ready', 'high_similarity.alert']
    secret: Optional[str] = None


class WebhookResponse(BaseModel):
    id: int
    url: str
    events: List[str]
    is_active: bool
    failure_count: int
    last_triggered_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


@router.post('', response_model=WebhookResponse, status_code=status.HTTP_201_CREATED)
async def register_webhook(
    webhook: WebhookCreate,
    org_id: int = Depends(get_current_user_org_id),
    db: AsyncSession = Depends(get_session),
):
    '''Register webhook endpoint for organization'''
    secret = webhook.secret or secrets.token_urlsafe(32)
    endpoint = WebhookEndpoint(
        org_id=org_id,
        url=str(webhook.url),
        events=webhook.events,
        secret=secret,
    )
    db.add(endpoint)
    await db.commit()
    await db.refresh(endpoint)
    return WebhookResponse.model_validate(endpoint)


@router.get('', response_model=List[WebhookResponse])
async def list_webhooks(
    org_id: int = Depends(get_current_user_org_id),
    db: AsyncSession = Depends(get_session),
):
    '''List webhook endpoints for organization'''
    result = await db.execute(select(WebhookEndpoint).where(WebhookEndpoint.org_id == org_id))
    endpoints = result.scalars().all()
    return [WebhookResponse.model_validate(e) for e in endpoints]


@router.delete('/{webhook_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_webhook(
    webhook_id: int,
    org_id: int = Depends(get_current_user_org_id),
    db: AsyncSession = Depends(get_session),
):
    '''Delete webhook endpoint'''
    result = await db.execute(select(WebhookEndpoint).where(WebhookEndpoint.id == webhook_id, WebhookEndpoint.org_id == org_id))
    endpoint = result.scalar_one_or_none()
    if not endpoint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Webhook not found')
    await db.delete(endpoint)
    await db.commit()
