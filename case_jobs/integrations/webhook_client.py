from __future__ import annotations

import json
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

import requests

from case_jobs.exceptions import ValidationError, WebhookDeliveryError
from case_jobs.integrations.webhook_signing import sign_webhook_body


RETRYABLE_STATUS_CODES = frozenset({408, 429})
WEBHOOK_PATH = "/webhooks/documents"


def resolve_webhook_url(webhook_url: str) -> str:
    parsed = urlsplit(webhook_url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValidationError("WEBHOOK_ROOT_URL must be HTTPS")
    if parsed.username or parsed.password:
        raise ValidationError("WEBHOOK_ROOT_URL cannot contain credentials")
    if parsed.query or parsed.fragment:
        raise ValidationError("WEBHOOK_ROOT_URL cannot contain query or fragment")
    if parsed.path != WEBHOOK_PATH:
        raise ValidationError(
            f"WEBHOOK_ROOT_URL must use the exact {WEBHOOK_PATH} path"
        )
    return urlunsplit(("https", parsed.netloc, WEBHOOK_PATH, "", ""))


def _retry_after_seconds(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return max(0, int(value))
    except ValueError:
        try:
            target = parsedate_to_datetime(value)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            return max(0, int((target - datetime.now(timezone.utc)).total_seconds()))
        except (TypeError, ValueError, OverflowError):
            return None


@dataclass(frozen=True)
class WebhookDelivery:
    status_code: int
    event_id: str


class WebhookClient:
    def __init__(self, *, timeout: int = 10, session=None):
        self.timeout = timeout
        self.session = session or requests.Session()

    def deliver(self, url: str, event: dict, secret: str) -> WebhookDelivery:
        body = json.dumps(event, separators=(",", ":"), sort_keys=True).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "X-Signature": sign_webhook_body(body, secret),
        }
        try:
            response = self.session.post(
                url,
                data=body,
                headers=headers,
                timeout=self.timeout,
                allow_redirects=False,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise WebhookDeliveryError(str(exc), retryable=True) from exc
        except requests.RequestException as exc:
            raise WebhookDeliveryError(str(exc), retryable=False) from exc

        if response.status_code == 200:
            return WebhookDelivery(response.status_code, event["event_id"])
        retryable = (
            response.status_code in RETRYABLE_STATUS_CODES
            or response.status_code >= 500
        )
        if response.status_code == 400:
            message = "Webhook receiver rejected a malformed event payload (HTTP 400)"
        elif response.status_code == 401:
            message = "Webhook receiver rejected the request signature (HTTP 401)"
        else:
            message = f"Webhook receiver returned HTTP {response.status_code}"
        raise WebhookDeliveryError(
            message,
            retryable=retryable,
            retry_after=_retry_after_seconds(response.headers.get("Retry-After")),
            status_code=response.status_code,
        )
