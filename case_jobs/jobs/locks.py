from contextlib import contextmanager

from case_jobs.integrations.redis_client import get_redis_client


@contextmanager
def distributed_lock(
    name: str,
    *,
    timeout: int = 900,
    blocking_timeout: int = 5,
    client=None,
):
    redis_client = client or get_redis_client()
    lock = redis_client.lock(
        f"case_jobs:lock:{name}",
        timeout=timeout,
        blocking_timeout=blocking_timeout,
    )
    acquired = lock.acquire(blocking=True)
    if not acquired:
        raise TimeoutError(f"Could not acquire lock: {name}")
    try:
        yield lock
    finally:
        if lock.owned():
            lock.release()

