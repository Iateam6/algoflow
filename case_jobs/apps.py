from django.apps import AppConfig


class CaseJobsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "case_jobs"

    def ready(self) -> None:
        from importlib import import_module

        for module_name in (
            "aap",
            "aea",
            "ds_160",
            "ds_260",
            "eb_1aA",
            "eb_1aB",
            "naturalization",
            "reentry_permit",
        ):
            import_module(f"{module_name}.adapter").register_adapter()
