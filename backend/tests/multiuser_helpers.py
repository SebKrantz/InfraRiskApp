"""Shared by the multi-user tests: switch app.config into multi-user mode and
build the headers Caddy forwards for a signed-in user, or AGUI sends on an
internal call."""

from __future__ import annotations

import hashlib
from unittest import mock

from app import config

SECRET = "edge-secret-used-only-in-tests"
TOKEN = "service-token-used-only-in-tests"
HEADER = "X-Auth-Request-Email"


def multi_user(admins: tuple[str, ...] = ()):
    """Patch app.config into multi-user mode, as the deployment sets it."""
    return mock.patch.multiple(
        config,
        AEI_AUTH_HEADER=HEADER,
        AEI_EDGE_SECRET_SHA256=hashlib.sha256(SECRET.encode()).hexdigest(),
        AEI_SERVICE_TOKEN_SHA256=hashlib.sha256(TOKEN.encode()).hexdigest(),
        AEI_ADMIN_EMAILS=frozenset(admins),
    )


def signed_in(email: str, secret: str = SECRET) -> dict[str, str]:
    """The headers Caddy forwards for a signed-in user."""
    return {HEADER: email, "X-AEI-Edge": secret}


def internal(user: str | None = None, scope: str | None = None, token: str = TOKEN) -> dict[str, str]:
    """The headers AGUI sends on an internal call: the service token, the chat's
    owner and its scope when given."""
    headers = {"Authorization": f"Bearer {token}"}
    if user is not None:
        headers["X-AEI-User"] = user
    if scope is not None:
        headers["X-AEI-Scope"] = scope
    return headers
