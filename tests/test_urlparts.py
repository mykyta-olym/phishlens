from __future__ import annotations

import pytest

from phishlens.urlparts import classify_host, parse_url, registered_domain


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.example.co.uk/a", "example.co.uk"),
        ("example.com", "example.com"),
        ("EXAMPLE.COM./x", "example.com"),
        ("secure.paypal.com.evil.example.ru/login", "example.ru"),
        ("a.b.c.example.org", "example.org"),
    ],
)
def test_registered_domain(url: str, expected: str) -> None:
    assert registered_domain(url) == expected


def test_www_on_a_public_suffix_host_stays_in_one_group() -> None:
    assert registered_domain("https://www.gov.uk/x") == registered_domain("https://gov.uk/x")


def test_www_dot_tld_is_not_reduced_to_a_bare_suffix() -> None:
    assert registered_domain("https://www.com/") == "www.com"


def test_platform_tenants_share_one_group() -> None:
    """Tenants of one platform must not end up on both sides of a split."""
    assert registered_domain("https://a.github.io") == "github.io"
    assert registered_domain("https://b.github.io") == "github.io"
    assert registered_domain("https://x.y.web.app") == "web.app"


def test_platform_host_flag() -> None:
    assert parse_url("https://someone.github.io/page").is_platform_host
    assert not parse_url("https://github.com/someone").is_platform_host


def test_ip_addresses_skip_the_suffix_lookup() -> None:
    """The suffix list would otherwise turn 192.168.1.1 into the domain 1.1."""
    parts = parse_url("http://192.168.1.1/admin")
    assert parts.host_kind == "ipv4"
    assert parts.registered_domain == "192.168.1.1"
    assert parts.public_suffix == ""


@pytest.mark.parametrize("host", ["2130706433", "0x7f.0.0.1", "0177.0.0.1", "0x7f000001"])
def test_obfuscated_ip_forms(host: str) -> None:
    assert classify_host(host) == "obfuscated_ip"


@pytest.mark.parametrize("host", ["1.2.3.4", "255.255.255.255"])
def test_plain_ipv4_is_not_obfuscated(host: str) -> None:
    assert classify_host(host) == "ipv4"


def test_ipv6_host_and_port() -> None:
    parts = parse_url("http://[2001:db8::1]:8080/x")
    assert parts.host_kind == "ipv6"
    assert parts.host == "2001:db8::1"
    assert parts.port == 8080


def test_userinfo_does_not_become_the_host() -> None:
    parts = parse_url("http://paypal.com@evil.example.ru/login")
    assert parts.host == "evil.example.ru"
    assert parts.userinfo == "paypal.com"


def test_host_with_port_and_no_scheme() -> None:
    parts = parse_url("example.com:8080/login")
    assert parts.host == "example.com"
    assert parts.port == 8080
    assert parts.path == "/login"


def test_backslashes_are_read_like_a_browser_reads_them() -> None:
    assert parse_url("http:\\\\example.com\\a").host == "example.com"


@pytest.mark.parametrize("url", ["mailto:a@example.com", "javascript:alert(1)", "data:text/html,x"])
def test_other_schemes_have_no_host(url: str) -> None:
    parts = parse_url(url)
    assert parts.host == ""
    assert parts.scheme == url.split(":", 1)[0]


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "http://",
        "http://[bad/x",
        "http://?x=1",
        "://",
        "http://:80",
        "%%%",
        "\x00",
        "[[[",
    ],
)
def test_malformed_input_never_raises(url: str) -> None:
    parse_url(url)


def test_out_of_range_port_is_dropped() -> None:
    assert parse_url("http://example.com:99999/").port is None
