"""Turn a URL string into numbers. No network, no page content, no external lookup.

The model reads a canonical view of the URL: host without a leading `www.`, and path
without a trailing slash. It deliberately never sees the scheme, query string,
fragment, port or the `www.` prefix.

That is not an oversight. In the training data those fields separate the classes for
reasons that have nothing to do with phishing. Benign URLs came from anonymised
telemetry, which removed nearly every query string, and the trailing slash rate of
the phishing feed changed over time. A model that sees them scores well on held-out
rows and then flags any real URL with `?utm_source=x`. `artefact_features` keeps
those fields available so the evaluation can measure that harm directly.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from phishlens.lexicon import (
    BRANDS,
    HTML_EXTENSIONS,
    KEYWORDS,
    MIN_SUBSTRING_BRAND_LENGTH,
    SCRIPT_EXTENSIONS,
    SHARED_HOSTS,
    SHORTENERS,
    TLD_VOCABULARY,
)
from phishlens.urlparts import UrlParts, parse_url

_TOKEN = re.compile(r"[a-z0-9]+")
_RANDOM_TOKEN = re.compile(r"^(?=.*\d)(?=.*[a-zA-Z])[a-zA-Z0-9]{12,}$|^[0-9a-fA-F]{16,}$")
_EMBEDDED_URL = re.compile(r"https?[:/%]|www\.|\.(?:com|net|org)\b", re.IGNORECASE)
_VOWELS = frozenset("aeiou")
_TLD_IDS = {tld: index for index, tld in enumerate(TLD_VOCABULARY)}
_LONG_BRANDS = tuple(b for b in BRANDS if len(b) >= MIN_SUBSTRING_BRAND_LENGTH)
_OFFICIAL_DOMAINS = frozenset(domain for domains in BRANDS.values() for domain in domains)
_HOMOGLYPHS = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "7": "t"})
_LOOKALIKE_MAX_DISTANCE = 2
_LOOKALIKE_MIN_LENGTH = 5


@dataclass(frozen=True, slots=True)
class Feature:
    """One model input. `text` is the plain phrase shown in explanations."""

    name: str
    group: str
    text: str


FEATURES: tuple[Feature, ...] = (
    Feature("host_length", "length", "length of the host name"),
    Feature("path_length", "length", "length of the path"),
    Feature("subdomain_depth", "host", "number of subdomain levels"),
    Feature("max_host_label_length", "host", "longest label in the host name"),
    Feature("host_hyphen_count", "host", "hyphens in the host name"),
    Feature("host_digit_ratio", "host", "share of digits in the host name"),
    Feature("is_ip_host", "host", "the host is an IP address"),
    Feature("is_punycode", "host", "the host uses punycode (xn--)"),
    Feature("has_non_ascii_host", "host", "the host has non-ASCII characters"),
    Feature("has_userinfo", "host", "text before an @ hides the real host"),
    Feature("is_platform_host", "host", "the host is on a shared hosting platform"),
    Feature("is_shortener", "host", "the host is a URL shortener"),
    Feature("suffix_label_count", "host", "labels in the public suffix"),
    Feature("tld_id", "host", "the top-level domain"),
    Feature("sld_length", "name", "length of the registered name"),
    Feature("sld_entropy", "name", "randomness of the registered name"),
    Feature("sld_digit_ratio", "name", "share of digits in the registered name"),
    Feature("sld_hyphen_count", "name", "hyphens in the registered name"),
    Feature("sld_vowel_ratio", "name", "share of vowels in the registered name"),
    Feature("sld_consonant_run", "name", "longest consonant run in the registered name"),
    Feature("subdomain_length", "name", "length of the subdomain"),
    Feature("subdomain_entropy", "name", "randomness of the subdomain"),
    Feature("subdomain_digit_ratio", "name", "share of digits in the subdomain"),
    Feature("path_depth", "path", "number of path segments"),
    Feature("max_segment_length", "path", "longest path segment"),
    Feature("path_digit_ratio", "path", "share of digits in the path"),
    Feature("path_entropy", "path", "randomness of the path"),
    Feature("path_hyphen_count", "path", "hyphens in the path"),
    Feature("path_underscore_count", "path", "underscores in the path"),
    Feature("path_dot_count", "path", "dots in the path"),
    Feature("path_uppercase_ratio", "path", "share of capitals in the path"),
    Feature("path_percent_count", "path", "percent-encoded characters in the path"),
    Feature("file_kind", "path", "file type at the end of the path"),
    Feature("has_random_token", "path", "a path segment looks randomly generated"),
    Feature("has_double_slash", "path", "the path contains a double slash"),
    Feature("path_has_embedded_url", "path", "the path contains something like a URL"),
    Feature("host_keyword_count", "lexical", "credential or alert words in the host"),
    Feature("path_keyword_count", "lexical", "credential or alert words in the path"),
    Feature("brand_in_subdomain", "lexical", "a brand name appears in the subdomain"),
    Feature("brand_in_name", "lexical", "a brand name is in a domain it does not own"),
    Feature("brand_in_path", "lexical", "a brand name appears in the path"),
    Feature("brand_official_domain", "lexical", "the domain is one a listed brand owns"),
    Feature("brand_lookalike", "lexical", "the name is a near misspelling of a brand"),
)

FEATURE_NAMES: tuple[str, ...] = tuple(feature.name for feature in FEATURES)
FEATURE_INDEX: dict[str, int] = {name: index for index, name in enumerate(FEATURE_NAMES)}
FEATURE_TEXT: dict[str, str] = {feature.name: feature.text for feature in FEATURES}
CATEGORICAL_FEATURES: tuple[str, ...] = ("tld_id", "file_kind")
FILE_KINDS: tuple[str, ...] = ("none", "html", "script", "other")

ARTEFACT_FEATURE_NAMES: tuple[str, ...] = (
    "artefact_is_https",
    "artefact_has_scheme",
    "artefact_has_www",
    "artefact_trailing_slash",
    "artefact_has_query",
    "artefact_query_length",
    "artefact_query_params",
    "artefact_has_fragment",
    "artefact_has_port",
)


def entropy(text: str) -> float:
    """Shannon entropy in bits per character."""
    if not text:
        return 0.0
    length = len(text)
    return -sum(n / length * math.log2(n / length) for n in Counter(text).values())


def _digit_ratio(text: str) -> float:
    return sum(c.isdigit() for c in text) / len(text) if text else 0.0


def _consonant_run(text: str) -> int:
    best = current = 0
    for char in text:
        current = current + 1 if char.isalpha() and char not in _VOWELS else 0
        best = max(best, current)
    return best


def _edit_distance(left: str, right: str, limit: int) -> int:
    """Levenshtein distance, stopping once it passes `limit`."""
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            cost = left_char != right_char
            current.append(min(previous[column] + 1, current[-1] + 1, previous[column - 1] + cost))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _mentions_brand(text: str, tokens: frozenset[str]) -> str | None:
    """Return the brand named in `text`, matching short names as whole words only."""
    for brand in BRANDS:
        if brand in tokens:
            return brand
    for brand in _LONG_BRANDS:
        if brand in text:
            return brand
    return None


@lru_cache(maxsize=65536)
def _lookalike_distance(name: str) -> int:
    """Distance from `name` to the closest brand, 0 when nothing is close or it is exact."""
    skeleton = name.translate(_HOMOGLYPHS).replace("-", "")
    if len(skeleton) < _LOOKALIKE_MIN_LENGTH:
        return 0
    best = _LOOKALIKE_MAX_DISTANCE + 1
    for brand in BRANDS:
        if len(brand) < _LOOKALIKE_MIN_LENGTH:
            continue
        if name == brand:
            return 0
        best = min(best, _edit_distance(skeleton, brand, _LOOKALIKE_MAX_DISTANCE))
    if best > _LOOKALIKE_MAX_DISTANCE:
        return 0
    # A homoglyph swap such as paypa1 reaches distance 0 and must still count.
    return max(best, 1)


def canonical_host(parts: UrlParts) -> str:
    return parts.host.removeprefix("www.")


def canonical_path(parts: UrlParts) -> str:
    return parts.path.rstrip("/")


def canonical_key(url: str) -> str:
    """What the model sees of a URL. Two URLs with one key always get one score."""
    parts = parse_url(url)
    return canonical_host(parts) + canonical_path(parts)


def _file_kind(last_segment: str) -> int:
    _, dot, extension = last_segment.rpartition(".")
    if not dot or not 1 <= len(extension) <= 5 or not extension.isalnum():
        return FILE_KINDS.index("none")
    extension = extension.lower()
    if extension in HTML_EXTENSIONS:
        return FILE_KINDS.index("html")
    if extension in SCRIPT_EXTENSIONS:
        return FILE_KINDS.index("script")
    return FILE_KINDS.index("other")


def _is_shortener(host: str, registered: str) -> bool:
    return host in SHORTENERS or registered in SHORTENERS


def is_shared_host(parts: UrlParts) -> bool:
    return parts.is_platform_host or canonical_host(parts) in SHARED_HOSTS


def is_shortener(parts: UrlParts) -> bool:
    return _is_shortener(canonical_host(parts), parts.registered_domain)


def extract_parts(parts: UrlParts) -> list[float]:
    """Compute the feature vector, in the order of `FEATURE_NAMES`."""
    host = canonical_host(parts)
    path = canonical_path(parts)
    is_domain = parts.host_kind == "domain"
    registered = parts.registered_domain if is_domain else ""
    suffix = parts.public_suffix if is_domain else ""
    sld = registered[: -len(suffix) - 1] if suffix and registered.endswith("." + suffix) else ""
    subdomain = parts.subdomain.removeprefix("www").removeprefix(".") if is_domain else ""
    labels = host.split(".") if host else []
    segments = [segment for segment in path.split("/") if segment]
    last_segment = segments[-1] if segments else ""
    letters = sum(c.isalpha() for c in path)
    lowered_path = path.lower()
    lowered_subdomain = subdomain.lower()

    host_tokens = frozenset(_TOKEN.findall(host))
    subdomain_tokens = frozenset(_TOKEN.findall(lowered_subdomain))
    sld_tokens = frozenset(_TOKEN.findall(sld))
    path_tokens = frozenset(_TOKEN.findall(lowered_path))
    brand_in_sld = _mentions_brand(sld, sld_tokens)
    owns_domain = brand_in_sld is not None and registered in BRANDS[brand_in_sld]

    values: dict[str, float] = {
        "host_length": len(host),
        "path_length": len(path),
        "subdomain_depth": subdomain.count(".") + 1 if subdomain else 0,
        "max_host_label_length": max(map(len, labels), default=0),
        "host_hyphen_count": host.count("-"),
        "host_digit_ratio": _digit_ratio(host),
        "is_ip_host": parts.host_kind in ("ipv4", "ipv6", "obfuscated_ip"),
        "is_punycode": "xn--" in host,
        "has_non_ascii_host": not host.isascii(),
        "has_userinfo": bool(parts.userinfo),
        "is_platform_host": is_shared_host(parts),
        "is_shortener": is_shortener(parts),
        "suffix_label_count": suffix.count(".") + 1 if suffix else 0,
        "tld_id": _TLD_IDS.get(suffix.rsplit(".", 1)[-1], 0),
        "sld_length": len(sld),
        "sld_entropy": entropy(sld),
        "sld_digit_ratio": _digit_ratio(sld),
        "sld_hyphen_count": sld.count("-"),
        "sld_vowel_ratio": sum(c in _VOWELS for c in sld) / len(sld) if sld else 0.0,
        "sld_consonant_run": _consonant_run(sld),
        "subdomain_length": len(subdomain),
        "subdomain_entropy": entropy(subdomain),
        "subdomain_digit_ratio": _digit_ratio(subdomain),
        "path_depth": len(segments),
        "max_segment_length": max(map(len, segments), default=0),
        "path_digit_ratio": _digit_ratio(path),
        "path_entropy": entropy(path),
        "path_hyphen_count": path.count("-"),
        "path_underscore_count": path.count("_"),
        "path_dot_count": path.count("."),
        "path_uppercase_ratio": sum(c.isupper() for c in path) / letters if letters else 0.0,
        "path_percent_count": path.count("%"),
        "file_kind": _file_kind(last_segment),
        "has_random_token": any(_RANDOM_TOKEN.match(segment) for segment in segments),
        "has_double_slash": "//" in path,
        "path_has_embedded_url": bool(_EMBEDDED_URL.search(path)),
        "host_keyword_count": len(host_tokens & KEYWORDS),
        "path_keyword_count": len(path_tokens & KEYWORDS),
        "brand_in_subdomain": _mentions_brand(lowered_subdomain, subdomain_tokens) is not None,
        "brand_in_name": brand_in_sld is not None and not owns_domain,
        "brand_in_path": _mentions_brand(lowered_path, path_tokens) is not None,
        "brand_official_domain": registered in _OFFICIAL_DOMAINS,
        "brand_lookalike": _lookalike_distance(sld) if sld else 0,
    }
    return [float(values[name]) for name in FEATURE_NAMES]


def extract(url: str) -> list[float]:
    return extract_parts(parse_url(url))


def artefact_features(url: str) -> list[float]:
    """Fields excluded from the model on purpose. Used only to measure their harm."""
    parts = parse_url(url)
    return [
        float(parts.scheme == "https"),
        float(bool(parts.scheme)),
        float(parts.host.startswith("www.")),
        float(url.split("?", 1)[0].split("#", 1)[0].endswith("/")),
        float(bool(parts.query)),
        float(len(parts.query)),
        float(parts.query.count("&") + 1 if parts.query else 0),
        float(bool(parts.fragment)),
        float(parts.port is not None),
    ]
