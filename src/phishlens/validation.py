"""Decide whether a string is a URL the model can say something about.

The API and the CLI share these rules. The URL is never fetched, so there is no SSRF
concern here: private addresses are accepted and scored as strings.
"""

from __future__ import annotations

from phishlens.urlparts import parse_url

MAX_URL_LENGTH = 8192
MAX_BATCH_SIZE = 1000
ALLOWED_SCHEMES = ("", "http", "https")


class InvalidUrlError(ValueError):
    """The input cannot be scored. The message is safe to show to the caller."""


def validate_url(value: str) -> str:
    """Return the stripped URL, or raise `InvalidUrlError` with a plain reason."""
    text = value.strip()
    if not text:
        raise InvalidUrlError("the URL is empty")
    if len(text) > MAX_URL_LENGTH:
        raise InvalidUrlError(f"the URL is longer than {MAX_URL_LENGTH} characters")
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in text):
        raise InvalidUrlError("the URL contains whitespace or control characters")
    parts = parse_url(text)
    if parts.scheme not in ALLOWED_SCHEMES:
        raise InvalidUrlError("only http and https URLs can be scored")
    if not parts.host:
        raise InvalidUrlError("the URL has no host")
    if parts.host_kind == "domain" and "." not in parts.host:
        raise InvalidUrlError("the host must be a full domain name or an IP address")
    return text
