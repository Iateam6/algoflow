import logging

from celery import shared_task

from case_jobs.jobs.rate_limits import CapacityLimiter
from case_jobs.pipeline.orchestrator import execute_generation_job


logger = logging.getLogger(__name__)


@shared_task(name="case_jobs.run_generation_pipeline", acks_late=True)
def run_generation_pipeline(job: dict):
    job_id = job.get("job_id")
    tenant_id = job.get("tenant_id")
    visa_type = job.get("visa_type")
    logger.info(
        "pipeline task received job_id=%s tenant_id=%s visa_type=%s",
        job_id,
        tenant_id,
        visa_type,
    )
    try:
        result = execute_generation_job(job)
        logger.info(
            "pipeline task finished job_id=%s tenant_id=%s visa_type=%s status=%s stage=%s",
            job_id,
            tenant_id,
            visa_type,
            result.get("status"),
            result.get("stage"),
        )
        return result
    except Exception as exc:
        logger.exception(
            "pipeline task failed job_id=%s tenant_id=%s visa_type=%s error=%s",
            job_id,
            tenant_id,
            visa_type,
            type(exc).__name__,
        )
        raise
    finally:
        CapacityLimiter().release(job["tenant_id"])
