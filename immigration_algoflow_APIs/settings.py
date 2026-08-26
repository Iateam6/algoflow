import os
import environ
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from typing import Any

try:
    import redis.connection as _redis_connection

    # Some redis-py versions/typeshed stubs don't expose DEFAULT_RESP_VERSION.
    # Use getattr/setattr to keep this compatible with static type checkers.
    if getattr(_redis_connection, "DEFAULT_RESP_VERSION", None) is not None:
        setattr(_redis_connection, "DEFAULT_RESP_VERSION", 2)
except Exception:
    _redis_connection = None

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

env = environ.Env()
env.read_env(BASE_DIR / '.env')

OPENAI_API_KEY = env('OPENAI_API_KEY')

SECURE_SSL_REDIRECT = False


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "django-insecure-0p+wz2#5h1bsxcb84+7b)i9)gp)+r1bx#)-b+8mr-2k6ip_(($"

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

#ALLOWED_HOSTS = ["*"]

# #CORS configuration
# CORS_ALLOWED_ORIGINS = [
#     "https://algoflow.visa26.com/",
#     "https://app.visa26.com",
# ]

# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "aap",
    "aea",
    "ds_160",
    "ds_260",
    "eb_1aA",
    "eb_1aB",
    "final_copy",
    "naturalization",
    "reentry_permit",

]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "corsheaders.middleware.CorsMiddleware",
]

ROOT_URLCONF = "immigration_algoflow_APIs.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "immigration_algoflow_APIs.wsgi.application"


# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.dummy'
    }
}


# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/

STATIC_URL = "static/"

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'
PUBLIC_BASE_URL = env.str("PUBLIC_BASE_URL", default="http://127.0.0.1:8000")

# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# Shared generation infrastructure. Secrets are supplied through the
# environment; job state is never written to Django's database.
def _ensure_redis_resp2(url: Any) -> str:
    url_str = str(url) if url is not None else ""
    parsed = urlsplit(url_str)
    if parsed.scheme.lower() not in {"redis", "rediss"}:
        return url_str

    params = parse_qsl(parsed.query, keep_blank_values=True)
    filtered_params: list[tuple[str, str]] = []
    for item in params:
        if not item:
            continue
        key = str(item[0])
        value = str(item[1]) if len(item) > 1 else ""
        if key != "protocol":
            filtered_params.append((key, value))
    return urlunsplit(parsed._replace(query=urlencode(filtered_params)))

REDIS_URL = _ensure_redis_resp2(
    env.str("REDIS_URL", default="redis://algoai-redis:6379/2")
)
CELERY_BROKER_URL = _ensure_redis_resp2(
    env.str("CELERY_BROKER_URL", default=REDIS_URL)
)
CELERY_RESULT_BACKEND = _ensure_redis_resp2(
    env.str("CELERY_RESULT_BACKEND", default=REDIS_URL)
)
CELERY_TASK_TIME_LIMIT = env.int("CELERY_TASK_TIME_LIMIT", default=3600)
CELERY_TASK_SOFT_TIME_LIMIT = env.int("CELERY_TASK_SOFT_TIME_LIMIT", default=3300)
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = True
CELERY_WORKER_DISABLE_MINGLE = env.bool("CELERY_WORKER_DISABLE_MINGLE", default=False)
CELERY_WORKER_POOL = env.str("CELERY_WORKER_POOL", default="solo")
CELERY_WORKER_CONCURRENCY = env.int("CELERY_WORKER_CONCURRENCY", default=1)

JWT_SECRET = env.str("JWT_SECRET", default="")
JWT_ALGORITHM = env.str("JWT_ALGORITHM", default="HS256")
JWT_AUDIENCE = env.str("JWT_AUDIENCE", default="")
JWT_ISSUER = env.str("JWT_ISSUER", default="")
GENERATION_AUTH_ENABLED = env.bool("GENERATION_AUTH_ENABLED", default=False)
DEFAULT_TENANT_ID = env.str("DEFAULT_TENANT_ID", default="public")


WEBHOOK_ROOT_URL = env.str(
    "WEBHOOK_ROOT_URL",
    default="https://api.visa26.com/webhooks/documents",
)
AI_DOC_WEBHOOK_SECRET = env.str("AI_DOC_WEBHOOK_SECRET", default="")
WEBHOOK_TIMEOUT_SECONDS = env.int("WEBHOOK_TIMEOUT_SECONDS", default=10)
WEBHOOK_MAX_RETRIES = env.int("WEBHOOK_MAX_RETRIES", default=8)

JOB_STATE_TTL_SECONDS = env.int("JOB_STATE_TTL_SECONDS", default=604800)
IDEMPOTENCY_TTL_SECONDS = env.int("IDEMPOTENCY_TTL_SECONDS", default=604800)
MAX_ACTIVE_JOBS_PER_TENANT = env.int("MAX_ACTIVE_JOBS_PER_TENANT", default=10)
MAX_ACTIVE_JOBS_GLOBAL = env.int("MAX_ACTIVE_JOBS_GLOBAL", default=100)
MAX_FILES_PER_JOB = env.int("MAX_FILES_PER_JOB", default=1000)
MAX_FILE_SIZE_BYTES = env.int("MAX_FILE_SIZE_BYTES", default=524288000)
MAX_TOTAL_DOWNLOAD_BYTES = env.int("MAX_TOTAL_DOWNLOAD_BYTES", default=2621440000)
MAX_PAGES_PER_JOB = env.int("MAX_PAGES_PER_JOB", default=10000)
FILE_DOWNLOAD_CONNECT_TIMEOUT = env.int("FILE_DOWNLOAD_CONNECT_TIMEOUT", default=5)
FILE_DOWNLOAD_READ_TIMEOUT = env.int("FILE_DOWNLOAD_READ_TIMEOUT", default=30)
FILE_DOWNLOAD_MAX_REDIRECTS = env.int("FILE_DOWNLOAD_MAX_REDIRECTS", default=3)
allowed_domains_raw = env.str("FILE_DOWNLOAD_ALLOWED_DOMAINS", default="")
FILE_DOWNLOAD_ALLOWED_DOMAINS = tuple(
    item.strip().lower()
    for item in allowed_domains_raw.split(",")
    if item.strip()
)

AWS_S3_BUCKET_NAME = env.str("AWS_S3_BUCKET_NAME", default="")
AWS_S3_REGION_NAME = env.str("AWS_S3_REGION_NAME", default="us-east-1")
AWS_S3_PRESIGNED_URL_EXPIRY_SECONDS = env.int(
    "AWS_S3_PRESIGNED_URL_EXPIRY_SECONDS", default=900
)

VERIFICATION_MODEL = env.str("VERIFICATION_MODEL",default="gpt-5.6")
ENABLE_MODEL_VERIFICATION = env.bool("ENABLE_MODEL_VERIFICATION", default=False)
ENABLE_DOCUMENT_VERIFICATION = env.bool("ENABLE_DOCUMENT_VERIFICATION", default=False)
OCR_MODEL = env.str("OCR_MODEL", default="gpt-5.4")
AUXILIARY_CONTEXT_MODEL = env.str(
    "AUXILIARY_CONTEXT_MODEL",
    default="gpt-5.6",
)
RAG_EMBEDDING_MODEL = env.str(
    "RAG_EMBEDDING_MODEL",
    default="text-embedding-3-small",
)
HYBRID_RETRIEVAL_CANDIDATE_K = env.int(
    "HYBRID_RETRIEVAL_CANDIDATE_K",
    default=30,
)
HYBRID_RETRIEVAL_FINAL_K = env.int(
    "HYBRID_RETRIEVAL_FINAL_K",
    default=20,
)
HYBRID_RETRIEVAL_MMR_FETCH_K = env.int(
    "HYBRID_RETRIEVAL_MMR_FETCH_K",
    default=60,
)
if HYBRID_RETRIEVAL_CANDIDATE_K < HYBRID_RETRIEVAL_FINAL_K:
    raise ValueError(
        "HYBRID_RETRIEVAL_CANDIDATE_K cannot be lower than "
        "HYBRID_RETRIEVAL_FINAL_K"
    )
if HYBRID_RETRIEVAL_MMR_FETCH_K < HYBRID_RETRIEVAL_CANDIDATE_K:
    raise ValueError(
        "HYBRID_RETRIEVAL_MMR_FETCH_K cannot be lower than "
        "HYBRID_RETRIEVAL_CANDIDATE_K"
    )
IDENTITY_EVIDENCE_POLICY = env.str("IDENTITY_EVIDENCE_POLICY", default="warn").strip().lower()
GENERATION_LOG_PREVIEW_CHARS = env.int("GENERATION_LOG_PREVIEW_CHARS", default=2000)
PRINT_GENERATION_MARKDOWN = env.bool("PRINT_GENERATION_MARKDOWN", default=False)