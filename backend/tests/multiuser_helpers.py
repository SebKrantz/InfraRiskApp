"""Shared by the multi-user tests: switch app.config into multi-user mode and
build the headers Caddy forwards for a signed-in user."""

from __future__ import annotations

import hashlib
from unittest import mock

from app import config

SECRET = "edge-secret-used-only-in-tests"
HEADER = "X-Auth-Request-Email"


def multi_user(admins: tuple[str, ...] = ()):
    """Patch app.config into multi-user mode, as the deployment sets it."""
    return mock.patch.multiple(
        config,
        AEI_AUTH_HEADER=HEADER,
        AEI_EDGE_SECRET_SHA256=hashlib.sha256(SECRET.encode()).hexdigest(),
        AEI_ADMIN_EMAILS=frozenset(admins),
    )


def signed_in(email: str, secret: str = SECRET) -> dict[str, str]:
    """The headers Caddy forwards for a signed-in user."""
    return {HEADER: email, "X-AEI-Edge": secret}
