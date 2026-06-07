import os

from django.conf import settings
from openai import OpenAI


def get_openai_api_key() -> str:
    api_key = getattr(settings, "OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    return api_key


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=get_openai_api_key())
