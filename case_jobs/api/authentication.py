from __future__ import annotations

import re

import jwt
from django.conf import settings

from case_jobs.api.schemas import TenantPrincipal
from case_jobs.exceptions import AuthenticationError, ServiceConfigurationError


TENANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def authenticate_bearer_header(header: str | None) -> TenantPrincipal:
    if not header or not header.startswith("Bearer "):
        raise AuthenticationError("Bearer token is required")
    token = header[7:].strip()
    if not token:
        raise AuthenticationError("Bearer token is required")
    if not settings.JWT_SECRET:
        raise AuthenticationError("JWT authentication is not configured")

    kwargs = {
        "algorithms": [settings.JWT_ALGORITHM],
        "options": {"require": ["exp", "sub", "tenant_id"]},
    }
    if settings.JWT_AUDIENCE:
        kwargs["audience"] = settings.JWT_AUDIENCE
    else:
        kwargs["options"]["verify_aud"] = False
    if settings.JWT_ISSUER:
        kwargs["issuer"] = settings.JWT_ISSUER

    try:
        claims = jwt.decode(token, settings.JWT_SECRET, **kwargs)
    except jwt.PyJWTError as exc:
        raise AuthenticationError("Bearer token is invalid or expired") from exc

    tenant_id = str(claims.get("tenant_id", "")).strip()
    subject = str(claims.get("sub", "")).strip()
    if not tenant_id or not subject or not TENANT_ID_PATTERN.fullmatch(tenant_id):
        raise AuthenticationError("Bearer token is missing required claims")
    return TenantPrincipal(tenant_id=tenant_id, subject=subject, claims=claims)


def resolve_generation_principal(header: str | None) -> TenantPrincipal:
    """Use JWT only when explicitly enabled; otherwise use one public tenant."""
    if bool(getattr(settings, "GENERATION_AUTH_ENABLED", False)):
        return authenticate_bearer_header(header)

    tenant_id = str(getattr(settings, "DEFAULT_TENANT_ID", "") or "").strip()
    if not tenant_id or not TENANT_ID_PATTERN.fullmatch(tenant_id):
        raise ServiceConfigurationError(
            "DEFAULT_TENANT_ID is missing or contains unsupported characters"
        )
    return TenantPrincipal(
        tenant_id=tenant_id,
        subject="public-api",
        claims={},
    )
