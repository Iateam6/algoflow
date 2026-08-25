from __future__ import annotations

from functools import lru_cache

import redis
from django.conf import settings


@lru_cache(maxsize=1)
def get_redis_client() -> redis.Redis:
    return redis.Redis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        health_check_interval=30,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
        socket_keepalive=True,
    )


def check_redis_connection() -> bool:
    return bool(get_redis_client().ping())
