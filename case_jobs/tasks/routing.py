TASK_ROUTES = {
    "case_jobs.run_generation_pipeline": {"queue": "file_reading"},
    "case_jobs.deliver_webhook": {"queue": "webhooks"},
}

