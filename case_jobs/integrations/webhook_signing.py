import base64
import hashlib
import hmac


def sign_webhook_body(body: bytes, secret: str) -> str:
    if not secret:
        raise ValueError("webhook secret cannot be empty")
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


def verify_webhook_signature(
    body: bytes,
    secret: str,
    signature: str,
) -> bool:
    expected = sign_webhook_body(body, secret)
    return hmac.compare_digest(expected, signature)
