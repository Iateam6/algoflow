from django.core.management.base import BaseCommand, CommandError

from case_jobs.jobs.job_store import JobStore
from case_jobs.tasks.webhook_tasks import queue_job_webhook


class Command(BaseCommand):
    help = "Replay the current webhook event for a Redis-backed generation job."

    def add_arguments(self, parser):
        parser.add_argument("tenant_id")
        parser.add_argument("job_id")

    def handle(self, *args, **options):
        record = JobStore().get(options["tenant_id"], options["job_id"])
        if not record:
            raise CommandError("Job was not found in Redis")
        queue_job_webhook(record)
        self.stdout.write(self.style.SUCCESS("Webhook replay queued"))

