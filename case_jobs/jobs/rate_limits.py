from __future__ import annotations

from django.conf import settings

from case_jobs.exceptions import CapacityExceeded
from case_jobs.integrations.redis_client import get_redis_client


ACQUIRE_SCRIPT = """
local tenant = tonumber(redis.call('GET', KEYS[1]) or '0')
local global = tonumber(redis.call('GET', KEYS[2]) or '0')
if tenant >= tonumber(ARGV[1]) or global >= tonumber(ARGV[2]) then
  return 0
end
redis.call('INCR', KEYS[1])
redis.call('INCR', KEYS[2])
redis.call('EXPIRE', KEYS[1], ARGV[3])
redis.call('EXPIRE', KEYS[2], ARGV[3])
return 1
"""

RELEASE_SCRIPT = """
for _, key in ipairs(KEYS) do
  local current = tonumber(redis.call('GET', key) or '0')
  if current > 0 then redis.call('DECR', key) end
end
return 1
"""


class CapacityLimiter:
    def __init__(self, client=None):
        self.client = client or get_redis_client()

    @staticmethod
    def _keys(tenant_id: str) -> tuple[str, str]:
        return (
            f"case_jobs:capacity:tenant:{tenant_id}",
            "case_jobs:capacity:global",
        )

    def acquire(self, tenant_id: str) -> None:
        acquired = self.client.eval(
            ACQUIRE_SCRIPT,
            2,
            *self._keys(tenant_id),
            settings.MAX_ACTIVE_JOBS_PER_TENANT,
            settings.MAX_ACTIVE_JOBS_GLOBAL,
            settings.CELERY_TASK_TIME_LIMIT + 300,
        )
        if not acquired:
            raise CapacityExceeded("Generation capacity is currently full")

    def release(self, tenant_id: str) -> None:
        self.client.eval(RELEASE_SCRIPT, 2, *self._keys(tenant_id))

