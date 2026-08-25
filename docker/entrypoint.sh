#!/bin/sh
set -eu

# Note: `.env` is intentionally not baked into the Docker image (see `.dockerignore`).
# Pass required environment variables at runtime (e.g. `--env-file` or `-e` flags).
#
# If you want a single-container setup, set `RUN_LOCAL_REDIS=1` so the worker/web
# can connect to the in-container Redis at `redis://localhost:6379/0`.
if [ "${RUN_LOCAL_REDIS:-0}" = "1" ]; then
    redis-server /app/docker/redis.conf --daemonize yes
fi

exec "$@"
