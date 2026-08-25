import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "immigration_algoflow_APIs.settings")

from celery import Celery
from case_jobs.tasks.routing import TASK_ROUTES


app = Celery("immigration_algoflow_APIs")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.conf.task_routes = TASK_ROUTES
app.conf.imports = (
    "case_jobs.tasks.pipeline_tasks",
    "case_jobs.tasks.webhook_tasks",
)
app.autodiscover_tasks()
