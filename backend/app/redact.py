"""Keep the server's secrets inside the server.

Two halves. `scrub_environment()` runs once config.py has read the keys: it
removes every secret-looking variable from the process environment, so code
python_exec runs (in this process) finds none through os.environ, os.getenv,
posix.environ or a child process. `redact()` runs on everything a tool hands
back — MCP results, errors, the in-app assistant's tool results — and replaces
any secret value it still finds, drops credential parameters from URLs, and masks
Google credentials by their shape (see _GOOGLE_CREDENTIALS) for a key or token that
python_exec reads from a file instead of the environment.

A library that reads a credential from the environment at use time (GDAL's
AWS_* for /vsis3, say) would lose it here; pass such a value to the library
explicitly instead. The server holds none today: only the CARTO and model keys,
which config.py reads once, and the path of the Vertex service-account key file
(GOOGLE_APPLICATION_CREDENTIALS), which the Gemini adapter loads for its client itself.
"""

from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import quote

SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.I)
# Shorter values are too likely to occur in ordinary text to replace blindly.
_MIN_SECRET_LEN = 8
REDACTED = "[redacted]"

# a credential query parameter: ?key=… / &api_key=… / &access_token=…
_URL_CREDENTIAL = re.compile(
    r"([?&])(?:api[_-]?key|key|access[_-]?token|token|auth|secret|password)=[^&#\s\"'<>]*&?",
    re.I,
)

# Google credentials, found by shape rather than by value because they come out of files
# (gcloud's application_default_credentials.json, a service-account key) the server never
# put in the environment: a refresh token (1//…), an access token (ya29.…), a PEM private-key
# block (up to its END line, or to the end of the text when the output was cut inside the
# key), and the value of a "refresh_token" / "client_secret" / "private_key" field in JSON
# (double quotes) or a printed dict (single quotes).
_GOOGLE_CREDENTIALS = (
    (re.compile(r"1//[0-9A-Za-z_-]{20,}"), REDACTED),
    (re.compile(r"ya29\.[0-9A-Za-z_.-]{20,}"), REDACTED),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)",
                re.S), REDACTED),
    (re.compile(r"""(["'])(refresh_token|client_secret|private_key)\1(\s*:\s*)(["'])"""
                r"""(?:(?!\4)[^\\]|\\.)*\4"""),
     rf"\1\2\1\3\4{REDACTED}\4"),
)

_secrets: set[str] = set()


def remember(value: str | None) -> None:
    """Treat `value` as a secret from now on (with its URL-quoted form)."""
    if value and len(value) >= _MIN_SECRET_LEN:
        _secrets.add(value)
        _secrets.add(quote(value, safe=""))


def scrub_environment() -> list[str]:
    """Remove secret-named variables from the environment, remembering their
    values for redact(). Returns the names removed."""
    removed = [name for name in os.environ if SECRET_NAME.search(name)]
    for name in removed:
        remember(os.environ.pop(name).strip())
    try:  # the interpreter's own start-up copy, which os.environ does not update
        import posix

        for name in [n for n in posix.environ if SECRET_NAME.search(os.fsdecode(n))]:
            remember(os.fsdecode(posix.environ.pop(name)).strip())
    except ImportError:  # not POSIX
        pass
    return removed


def _strip_url_credentials(text: str) -> str:
    def drop(m: re.Match) -> str:
        # keep the separator a following parameter needs: ?key=a&b=1 -> ?b=1
        return m.group(1) if m.group(0).endswith("&") else ""

    return _URL_CREDENTIAL.sub(drop, text)


def redact_text(text: str) -> str:
    for secret in _secrets:
        if secret in text:
            text = text.replace(secret, REDACTED)
    if "=" in text:
        text = _strip_url_credentials(text)
    for pattern, replacement in _GOOGLE_CREDENTIALS:
        text = pattern.sub(replacement, text)
    return text


def redact(value: Any) -> Any:
    """`value` with every string in it redacted (dicts, lists and tuples walked)."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact(v) for v in value)
    return value
