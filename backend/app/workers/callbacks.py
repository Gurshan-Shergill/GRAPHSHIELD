import httpx
import logging
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import async_session_factory
from app.models import WebhookEndpoint

settings = get_settings()
logger = logging.getLogger(__name__)


async def fire_webhook(db: AsyncSession, org_id: int, event: str, payload: dict):
    """Fire webhooks for an organization and event"""
    result = await db.execute(
        select(WebhookEndpoint).where(
            WebhookEndpoint.org_id == org_id,
            WebhookEndpoint.is_active == True,
            WebhookEndpoint.events.contains([event])
        )
    )
    webhooks = result.scalars().all()

    for webhook in webhooks:
        # Queue webhook delivery
        from app.workers.tasks import deliver_webhook_task
        deliver_webhook_task.delay(webhook.id, {'event': event, 'data': payload, 'timestamp': datetime.utcnow().isoformat()})


async def deliver_webhook(webhook_id: int, payload: dict):
    """Deliver a single webhook with retries"""
    async with async_session_factory() as db:
        webhook = await db.get(WebhookEndpoint, webhook_id)
        if not webhook or not webhook.is_active:
            return

        for attempt in range(settings.webhook_max_retries):
            try:
                async with httpx.AsyncClient(timeout=settings.webhook_timeout_seconds) as client:
                    response = await client.post(
                        webhook.url,
                        json=payload,
                        headers={
                            'X-Webhook-Signature': webhook.secret,
                            'Content-Type': 'application/json',
                        }
                    )
                    response.raise_for_status()
                    webhook.last_success_at = datetime.utcnow()
                    webhook.failure_count = 0
                    await db.commit()
                    logger.info(f"Webhook {webhook_id} delivered successfully")
                    return
            except Exception as e:
                logger.warning(f"Webhook {webhook_id} attempt {attempt + 1} failed: {e}")
                webhook.failure_count += 1
                await db.commit()

        webhook.last_triggered_at = datetime.utcnow()
        await db.commit()
        logger.error(f"Webhook {webhook_id} failed after {settings.webhook_max_retries} attempts")


def task_success_handler(sender=None, result=None, **kwargs):
    """Celery task success handler"""
    logger.info(f"Task {sender.name} succeeded")


def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwargs):
    """Celery task failure handler"""
    logger.error(f"Task {sender.name} failed: {exception}")