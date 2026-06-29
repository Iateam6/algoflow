import logging
import os

from django.conf import settings


logger = logging.getLogger(__name__)


def delete_file_if_exists(file_path: str) -> None:
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.remove(file_path)
        logger.info("Deleted existing media file %s", file_path)


def delete_files_in_directory(directory_path: str) -> None:
    os.makedirs(directory_path, exist_ok=True)
    for entry_name in os.listdir(directory_path):
        entry_path = os.path.join(directory_path, entry_name)
        delete_file_if_exists(entry_path)


def clean_media_files_for_generation() -> None:
    """
    Remove stale files before vectorization and output generation.

    Only files directly inside MEDIA_ROOT and MEDIA_ROOT/generated are removed.
    Subdirectories, including vector cache folders, are left intact.
    """
    media_root = str(settings.MEDIA_ROOT)
    generated_dir = os.path.join(media_root, "generated")

    delete_files_in_directory(media_root)
    delete_files_in_directory(generated_dir)
