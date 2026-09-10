from celery import Celery
from celery.schedules import crontab

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery('graphshield')
celery_app.conf.update(
    broker_url=settings.celery_broker_url,
    result_backend=settings.celery_result_backend,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    task_soft_time_limit=3000,  # 50 min soft limit
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_routes={
        'app.workers.tasks.process_paper_job': {'queue': 'high'},
        'app.workers.tasks.generate_report_task': {'queue': 'default'},
        'app.workers.tasks.index_figure_task': {'queue': 'default'},
        'app.workers.tasks.sync_shodhganga_task': {'queue': 'low'},
        'app.workers.tasks.cleanup_temp_files_task': {'queue': 'low'},
        'app.workers.tasks.deliver_webhook_task': {'queue': 'high'},
    },
    beat_schedule={
        'sync-shodhganga-weekly': {
            'task': 'app.workers.tasks.sync_shodhganga_task',
            'schedule': crontab(hour=2, minute=0, day_of_week=0),  # Sunday 2 AM
        },
        'cleanup-temp-files-daily': {
            'task': 'app.workers.tasks.cleanup_temp_files_task',
            'schedule': crontab(hour=3, minute=0),  # Daily 3 AM
        },
        'reindex-embeddings-weekly': {
            'task': 'app.workers.tasks.reindex_embeddings_task',
            'schedule': crontab(hour=4, minute=0, day_of_week=0),  # Sunday 4 AM
        },
    },
)

celery_app.autodiscover_tasks(['app.workers'])

# Task annotations for retries
celery_app.conf.task_annotations = {
    'app.workers.tasks.process_paper_job': {
        'rate_limit': '10/m',
        'autoretry_for': (Exception,),
        'retry_backoff': True,
        'retry_backoff_max': 600,
        'retry_jitter': True,
        'max_retries': 3,
    },
    'app.workers.tasks.deliver_webhook_task': {
        'rate_limit': '100/m',
        'autoretry_for': (Exception,),
        'retry_backoff': True,
        'retry_backoff_max': 300,
        'max_retries': 5,
    },
}
