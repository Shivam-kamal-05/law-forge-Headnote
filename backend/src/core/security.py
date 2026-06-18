"""Stateless demo authentication.

A single hardcoded operator guards the studio. Rather than keep a session
store, the bearer token is a deterministic HMAC of the username keyed by the
password + a server secret. The server can therefore validate any token by
recomputing it — no DB, no in-memory session map, survives restarts.

This is a *demo gate*, not an identity system: one user, no expiry, no
refresh. For production, swap this for real sessions / JWT with rotation.
"""

from __future__ import annotations

import hmac
from hashlib import sha256

from src.core.config import AuthSettings


def derive_token(settings: AuthSettings) -> str:
    """Deterministic opaque bearer token for the configured operator."""
    key = (
        settings.password.get_secret_value()
        + "::"
        + settings.token_secret.get_secret_value()
    ).encode("utf-8")
    msg = settings.username.encode("utf-8")
    return hmac.new(key, msg, sha256).hexdigest()


def _eq(a: str, b: str) -> bool:
    # Compare as UTF-8 bytes so non-ASCII input can't raise TypeError
    # (hmac.compare_digest rejects non-ASCII str) — a bad password must
    # always yield a clean 401, never a 500.
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def verify_credentials(settings: AuthSettings, username: str, password: str) -> bool:
    """Constant-time check of submitted credentials."""
    user_ok = _eq(username.strip(), settings.username)
    pass_ok = _eq(password, settings.password.get_secret_value())
    return user_ok and pass_ok


def verify_token(settings: AuthSettings, token: str) -> bool:
    """Constant-time check of a presented bearer token."""
    if not token:
        return False
    return _eq(token, derive_token(settings))
