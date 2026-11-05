"""Split a URL string into the parts the model reads.

Nothing here touches the network. Everything is string handling, plus the Public
Suffix List snapshot that ships inside the `publicsuffixlist` wheel.

Two different views of a domain matter:

- `registered_domain` uses ICANN suffixes only, so `a.github.io` and `b.github.io`
  both map to `github.io`. This is the key for leakage-free splitting, because one
  platform must not sit on both sides of a split.
- `is_platform_host` uses the private suffix section, which marks hosts where any
  customer can publish, such as `github.io` or `web.app`.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal
from urllib.parse import SplitResult, urlsplit

from publicsuffixlist import PublicSuffixList

_ICANN = PublicSuffixList(only_icann=True)
_FULL = PublicSuffixList()

_SCHEME = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*):")
_HEX_LABEL = re.compile(r"^0[xX][0-9a-fA-F]+$")
_OCTAL_LABEL = re.compile(r"^0[0-7]+$")
_DECIMAL_LABEL = re.compile(r"^[0-9]+$")

HostKind = Literal["domain", "ipv4", "ipv6", "obfuscated_ip", "none"]


@dataclass(frozen=True, slots=True)
class UrlParts:
    """One parsed URL. Every field is a plain string or number derived from the input."""

    raw: str
    scheme: str
    userinfo: str
    host: str
    port: int | None
    path: str
    query: str
    fragment: str
    host_kind: HostKind
    registered_domain: str
    public_suffix: str
    subdomain: str
    is_platform_host: bool


def _is_numeric_host(host: str) -> bool:
    """True for IPv4 hosts written in a form browsers accept but people cannot read.

    `2130706433`, `0x7f.0.0.1` and `0177.0.0.1` all resolve to 127.0.0.1.
    """
    labels = host.split(".")
    if not 1 <= len(labels) <= 4:
        return False
    plain = all(_DECIMAL_LABEL.match(label) for label in labels)
    if plain and len(labels) == 4 and not any(_OCTAL_LABEL.match(label) for label in labels):
        return False
    return all(
        _HEX_LABEL.match(label) or _OCTAL_LABEL.match(label) or _DECIMAL_LABEL.match(label)
        for label in labels
    )


def classify_host(host: str) -> HostKind:
    if not host:
        return "none"
    try:
        version = ipaddress.ip_address(host).version
    except ValueError:
        return "obfuscated_ip" if _is_numeric_host(host) else "domain"
    return "ipv4" if version == 4 else "ipv6"


@lru_cache(maxsize=65536)
def _split_domain(host: str) -> tuple[str, str, str, bool]:
    """Return registered domain, public suffix, subdomain and the platform flag."""
    suffix = _ICANN.publicsuffix(host) or ""
    registered = _ICANN.privatesuffix(host) or host
    prefix = host[: -len(registered)].rstrip(".") if host.endswith(registered) else ""
    full_suffix = _FULL.publicsuffix(host) or ""
    return registered, suffix, prefix, full_suffix != suffix


def _safe_split(text: str) -> SplitResult:
    try:
        return urlsplit(text)
    except ValueError:
        # Malformed brackets or similar. Keep the string usable as a plain host and path.
        return urlsplit(text.replace("[", "%5B").replace("]", "%5D"))


def _split_netloc(netloc: str) -> tuple[str, str, int | None]:
    """Split `user@host:port` by hand. `SplitResult.port` raises on bad ports."""
    userinfo, _, hostport = netloc.rpartition("@")
    if hostport.startswith("["):
        host, _, rest = hostport[1:].partition("]")
        port_text = rest.removeprefix(":")
    else:
        host, _, port_text = hostport.partition(":")
    port = int(port_text) if port_text.isascii() and port_text.isdigit() else None
    if port is not None and port > 65535:
        port = None
    return userinfo, host, port


def parse_url(raw: str) -> UrlParts:
    """Parse any string without raising. Malformed input gives empty or odd fields."""
    text = raw.strip().replace("\\", "/")
    match = _SCHEME.match(text)
    has_authority = text.startswith("//") or (match is not None and "://" in text[:32])
    if not has_authority and match is not None and match.group(1).lower() not in ("http", "https"):
        # `example.com:8080/x` looks like a scheme to a parser, `mailto:a@b.c` really is one.
        looks_like_host = "." in match.group(1) and text[match.end() : match.end() + 1].isdigit()
        if not looks_like_host:
            parts = _safe_split(text)
            return UrlParts(
                raw=raw,
                scheme=parts.scheme.lower(),
                userinfo="",
                host="",
                port=None,
                path=parts.path,
                query=parts.query,
                fragment=parts.fragment,
                host_kind="none",
                registered_domain="",
                public_suffix="",
                subdomain="",
                is_platform_host=False,
            )
    if not has_authority:
        text = "//" + text
    parts = _safe_split(text)

    userinfo, host, port = _split_netloc(parts.netloc)
    host = host.lower().rstrip(".")
    kind = classify_host(host)
    if kind == "domain":
        registered, suffix, subdomain, platform = _split_domain(host)
    else:
        registered, suffix, subdomain, platform = host, "", "", False

    return UrlParts(
        raw=raw,
        scheme=parts.scheme.lower(),
        userinfo=userinfo,
        host=host,
        port=port,
        path=parts.path,
        query=parts.query,
        fragment=parts.fragment,
        host_kind=kind,
        registered_domain=registered,
        public_suffix=suffix,
        subdomain=subdomain,
        is_platform_host=platform,
    )


def registered_domain(url: str) -> str:
    """The grouping key used for leakage-free splits."""
    return parse_url(url).registered_domain
