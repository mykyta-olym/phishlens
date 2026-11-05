from __future__ import annotations

import pytest

from phishlens.validation import MAX_URL_LENGTH, InvalidUrlError, validate_url


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://example.com/a?b=c",
        "example.com",
        "example.com:8080/x",
        "http://192.0.2.1/x",
        "http://[2001:db8::1]/x",
        "http://10.0.0.1/admin",
        "http://xn--pypal-4ve.com/",
    ],
)
def test_accepts(url: str) -> None:
    assert validate_url(url) == url


def test_strips_surrounding_whitespace() -> None:
    assert validate_url("  https://example.com\n") == "https://example.com"


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("https://exa mple.com", "whitespace"),
        ("https://example.com/\x00", "control"),
        ("https://example.com/\x7f", "control"),
        ("mailto:a@example.com", "http and https"),
        ("javascript:alert(1)", "http and https"),
        ("ftp://example.com/x", "http and https"),
        ("file:///etc/passwd", "http and https"),
        ("https://", "no host"),
        ("https:///path", "no host"),
        ("localhost", "full domain"),
        ("https://intranet/x", "full domain"),
    ],
)
def test_rejects(url: str, reason: str) -> None:
    with pytest.raises(InvalidUrlError, match=reason):
        validate_url(url)


def test_lone_surrogate_is_rejected() -> None:
    with pytest.raises(InvalidUrlError, match="Unicode"):
        validate_url("https://example.com/" + chr(0xD800))


def test_length_limit_is_exact() -> None:
    prefix = "https://example.com/"
    validate_url(prefix + "a" * (MAX_URL_LENGTH - len(prefix)))
    with pytest.raises(InvalidUrlError, match="longer"):
        validate_url(prefix + "a" * (MAX_URL_LENGTH - len(prefix) + 1))
