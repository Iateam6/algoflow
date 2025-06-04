#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
<<<<<<< HEAD
    os.environ.setdefault(
        "DJANGO_SETTINGS_MODULE", "immigration_algoflow_APIs.settings"
    )
=======
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "immigration_algoai_APIs.settings")
>>>>>>> 298b180593eec7888be759fd446b324f2cd91908
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
