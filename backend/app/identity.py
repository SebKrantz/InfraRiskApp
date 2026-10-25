"""Who is asking: the signed-in user behind each request.

Behind Caddy, oauth2-proxy signs people in with Google; Caddy forwards the
email in the header named by AEI_AUTH_HEADER together with the X-AEI-Edge
secret, whose sha256 is AEI_EDGE_SECRET_SHA256. A request to /api/* without
both is refused with 401 — the header alone could come from any container on
the internal network.

AGUI's internal calls (/mcp, its file push, artifact downloads) do not come
through Caddy. They carry `Authorization: Bearer <token>`, whose sha256 is
AEI_SERVICE_TOKEN_SHA256, and name the chat's owner in X-AEI-User. In
multi-user mode /mcp refuses a call without the token (401), and X-AEI-User is
ignored on any request that does not carry it.

With AEI_AUTH_HEADER empty (single-user mode, the local fleet) nothing is
checked and every request, conversation and dataset belongs to AEI_LOCAL_USER,
so the app behaves as it did before it knew about users.

Principals: a person is their lower-cased email; an MCP scope is `mcp:<scope>`.
What a scope creates is the scope's; a scope working for a person (X-AEI-User)
also reads that person's own datasets (conversations.Conversation.readers).
"""

from __future__ import annotations

import hashlib
import hmac
import re

from fastapi import Depends, HTTPException, Request
from starlette.datastructures import Headers
from starlette.responses import JSONResponse

from . import config

EDGE_HEADER = "x-aei-edge"
SERVICE_USER_HEADER = "x-aei-user"
SCOPE_HEADER = "x-aei-scope"
SHARED_SCOPE = "mcp-shared-scope"
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+")
_WRITES = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def multi_user() -> bool:
    return bool(config.AEI_AUTH_HEADER)


def local_user() -> str:
    return config.AEI_LOCAL_USER


def internal(headers) -> bool:
    """An internal call from AGUI: `Authorization: Bearer <token>` whose sha256 is
    AEI_SERVICE_TOKEN_SHA256."""
    expected = config.AEI_SERVICE_TOKEN_SHA256
    scheme, _, token = (headers.get("authorization") or "").partition(" ")
    if not expected or scheme.lower() != "bearer" or not token:
        return False
    return hmac.compare_digest(hashlib.sha256(token.encode()).hexdigest(), expected)


def internal_request(headers) -> bool:
    """Multi-user mode and an internal call: AGUI acting for its scopes."""
    return multi_user() and internal(headers)


def service_user(headers) -> str | None:
    """The person AGUI names in X-AEI-User (the chat's owner), on an internal call
    in multi-user mode only; ignored everywhere else."""
    if not internal_request(headers):
        return None
    email = (headers.get(SERVICE_USER_HEADER) or "").strip().lower()
    return email if _EMAIL.fullmatch(email) else None


def resolve(headers) -> str | None:
    """The principal for a request's headers, or None when multi-user mode refuses it.

    An internal call acts for the scope it names in X-AEI-Scope, else for its
    X-AEI-User, else for the shared MCP scope."""
    if not multi_user():
        return config.AEI_LOCAL_USER
    if internal(headers):
        scope = (headers.get(SCOPE_HEADER) or "").strip()[:128]
        if scope:
            return f"mcp:{scope}"
        return service_user(headers) or scope_principal(SHARED_SCOPE)
    edge = headers.get(EDGE_HEADER) or ""
    digest = hashlib.sha256(edge.encode()).hexdigest()
    if not edge or not hmac.compare_digest(digest, config.AEI_EDGE_SECRET_SHA256):
        return None
    email = (headers.get(config.AEI_AUTH_HEADER) or "").strip().lower()
    return email if _EMAIL.fullmatch(email) else None


def current_user(request: Request) -> str:
    """FastAPI dependency: the request's principal (401 when there is none).

    The middleware has normally resolved it already; resolving again here keeps
    routers that are mounted without the middleware (tests) just as strict.
    """
    user = getattr(request.state, "user", None)
    if user is None:
        user = resolve(request.headers)
        if user is None:
            raise HTTPException(401, "sign-in required")
    return user


def is_admin(user: str) -> bool:
    return not multi_user() or user in config.AEI_ADMIN_EMAILS


def require_admin(user: str = Depends(current_user)) -> str:
    if not is_admin(user):
        raise HTTPException(403, "admin only")
    return user


def scope_principal(conversation_id: str) -> str:
    """Who owns what an MCP scope creates: the scope itself in multi-user mode."""
    if not multi_user():
        return config.AEI_LOCAL_USER
    return conversation_id if conversation_id.startswith("mcp:") else f"mcp:{conversation_id}"


def is_scope_principal(owner: str) -> bool:
    return owner.startswith("mcp:")


def reserved_conversation_id(conversation_id: str | None) -> bool:
    """Ids a browser may not choose in multi-user mode: the MCP scopes' own."""
    return multi_user() and bool(conversation_id) and conversation_id.startswith("mcp")


class IdentityMiddleware:
    """Resolves the principal for every /api/* request (401 without one) and, in
    multi-user mode, refuses cross-site writes: the sign-in cookie is shared by
    the sibling subdomains, and a multipart or text/plain POST needs no CORS
    preflight. In multi-user mode /mcp takes internal calls only (401 otherwise).
    Pure ASGI, so SSE streams pass through untouched."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        path = scope.get("path", "") if scope["type"] == "http" else ""
        if multi_user() and (path == "/mcp" or path.startswith("/mcp/")):
            if not internal(Headers(scope=scope)):
                await JSONResponse({"detail": "service token required"}, 401)(scope, receive, send)
                return
        if not path.startswith("/api/"):
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        user = resolve(headers)
        if user is None:
            await JSONResponse({"detail": "sign-in required"}, 401)(scope, receive, send)
            return
        site = headers.get("sec-fetch-site")
        if multi_user() and scope["method"] in _WRITES and site not in (None, "same-origin", "none"):
            await JSONResponse({"detail": "cross-site request refused"}, 403)(scope, receive, send)
            return
        scope.setdefault("state", {})["user"] = user
        await self.app(scope, receive, send)
