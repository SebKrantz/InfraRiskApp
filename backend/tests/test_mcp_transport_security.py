"""ASSISTANT_MCP_ALLOWED_HOSTS and the MCP transport's Host allow-list.

The MCP SDK enables DNS-rebinding protection whenever the HTTP app is built for a localhost
bind, allowing only 127.0.0.1/localhost/[::1] as Host, and answers anything else with
421 Misdirected Request. That is right for a desktop install but refuses a client that
addresses this service by its container or service name inside a Docker network.
"""

from app import config


def test_unset_keeps_the_sdk_default(monkeypatch):
    """No allow-list configured: return None, so the SDK behaves exactly as before."""
    monkeypatch.setattr(config, "ASSISTANT_MCP_ALLOWED_HOSTS", [])
    assert config.mcp_transport_security() is None


def test_configured_hosts_are_added_with_protection_still_on(monkeypatch):
    monkeypatch.setattr(
        config, "ASSISTANT_MCP_ALLOWED_HOSTS", ["risk:*", "aei-risk:*"]
    )
    settings = config.mcp_transport_security()

    assert settings is not None
    # Protection stays ON; the allow-list is widened, not disabled.
    assert settings.enable_dns_rebinding_protection is True

    # localhost keeps working, so a desktop run is unaffected.
    for host in ("127.0.0.1:*", "localhost:*", "[::1]:*"):
        assert host in settings.allowed_hosts
    for origin in ("http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"):
        assert origin in settings.allowed_origins

    # and the configured names are accepted, with matching origins.
    assert "risk:*" in settings.allowed_hosts
    assert "aei-risk:*" in settings.allowed_hosts
    assert "http://risk:*" in settings.allowed_origins
    assert "http://aei-risk:*" in settings.allowed_origins


def test_entries_are_split_and_stripped(monkeypatch):
    """The env var is comma-separated; blanks and stray spaces are dropped."""
    monkeypatch.setenv("ASSISTANT_MCP_ALLOWED_HOSTS", " risk:* , , aei-risk:* ")
    parsed = [
        h.strip()
        for h in __import__("os").environ["ASSISTANT_MCP_ALLOWED_HOSTS"].split(",")
        if h.strip()
    ]
    assert parsed == ["risk:*", "aei-risk:*"]
