import logging
import shutil


logger = logging.getLogger(__name__)


def cleanup_job_directory(path: str) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except OSError:
        logger.exception("Could not clean job directory %s", path)

