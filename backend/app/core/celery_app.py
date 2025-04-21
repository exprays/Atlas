from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.task_routes = {
    "app.ml.tasks.*": {"queue": "ml-tasks"},
}

# Optional configuration
celery_app.conf.update(
    worker_max_tasks_per_child=1,  # Restart worker after each task (prevent memory leaks)
    task_acks_late=True,  # Task acknowledged after execution (not when received)
    task_reject_on_worker_lost=True,  # Requeue tasks if worker dies
    worker_prefetch_multiplier=1,  # Only prefetch one task at a time
)