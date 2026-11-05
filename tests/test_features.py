from __future__ import annotations

import math
import random
import string

import pytest

from phishlens.features import (
    ARTEFACT_FEATURE_NAMES,
    CATEGORICAL_FEATURES,
    FEATURE_INDEX,
    FEATURE_NAMES,
    FEATURE_TEXT,
    FEATURES,
    FILE_KINDS,
    artefact_features,
    canonical_key,
    entropy,
    extract,
)
from phishlens.lexicon import BRANDS, TLD_VOCABULARY


def value(url: str, name: str) -> float:
    return extract(url)[FEATURE_INDEX[name]]


def test_vector_matches_the_feature_list() -> None:
    assert len(extract("https://example.com/a")) == len(FEATURE_NAMES)
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_every_feature_has_explanation_text() -> None:
    assert all(FEATURE_TEXT[feature.name] for feature in FEATURES)


def test_categorical_features_exist() -> None:
    assert set(CATEGORICAL_FEATURES) <= set(FEATURE_NAMES)


@pytest.mark.parametrize(
    "variant",
    [
        "http://example.com/a/b",
        "https://example.com/a/b",
        "example.com/a/b",
        "//example.com/a/b",
        "https://www.example.com/a/b",
        "https://example.com/a/b/",
        "https://EXAMPLE.com/a/b",
        "https://example.com:8443/a/b",
        "https://example.com/a/b?utm_source=newsletter&x=1",
        "https://example.com/a/b#section",
    ],
)
def test_collection_artefacts_cannot_change_the_features(variant: str) -> None:
    """The fields that separate the classes for non-phishing reasons must be invisible."""
    assert extract(variant) == extract("https://example.com/a/b")


@pytest.mark.parametrize(
    "host",
    ["www30.example.com", "www2.a.example.co.uk", "wwwabc.example.org", "gov.uk", "a.web.app"],
)
def test_adding_www_never_changes_the_features(host: str) -> None:
    assert extract(f"https://www.{host}/a") == extract(f"https://{host}/a")


def test_a_label_starting_with_www_keeps_its_name() -> None:
    assert value("https://www30.example.com", "subdomain_length") == len("www30")


def test_canonical_key_ignores_the_same_fields() -> None:
    assert canonical_key("http://www.example.com/a/?x=1#y") == "example.com/a"


def test_path_case_still_matters() -> None:
    assert extract("https://example.com/ABC") != extract("https://example.com/abc")


def test_lengths() -> None:
    assert value("https://abc.example.com/xyz", "host_length") == len("abc.example.com")
    assert value("https://abc.example.com/xyz", "path_length") == len("/xyz")
    assert value("https://example.com/", "path_length") == 0


def test_subdomain_depth_skips_www() -> None:
    assert value("https://www.example.com", "subdomain_depth") == 0
    assert value("https://a.example.com", "subdomain_depth") == 1
    assert value("https://a.b.c.example.co.uk", "subdomain_depth") == 3


def test_ip_host() -> None:
    assert value("http://192.0.2.1/x", "is_ip_host") == 1
    assert value("http://2130706433/x", "is_ip_host") == 1
    assert value("http://example.com/x", "is_ip_host") == 0


def test_userinfo_trick() -> None:
    assert value("http://paypal.com@evil.example.ru/", "has_userinfo") == 1
    assert value("http://evil.example.ru/", "has_userinfo") == 0


def test_punycode_and_non_ascii() -> None:
    assert value("http://xn--pypal-4ve.com/", "is_punycode") == 1
    assert value("http://exämple.com/", "has_non_ascii_host") == 1
    assert value("http://example.com/", "has_non_ascii_host") == 0


def test_platform_and_shared_hosts() -> None:
    assert value("https://a.github.io/", "is_platform_host") == 1
    assert value("https://docs.google.com/forms/d/x", "is_platform_host") == 1
    assert value("https://www.google.com/", "is_platform_host") == 0


def test_shorteners() -> None:
    assert value("https://bit.ly/abc", "is_shortener") == 1
    assert value("https://tinyurl.com/abc", "is_shortener") == 1
    assert value("https://example.com/abc", "is_shortener") == 0


def test_tld_category() -> None:
    assert value("https://example.xyz", "tld_id") == TLD_VOCABULARY.index("xyz")
    assert value("https://example.co.uk", "tld_id") == TLD_VOCABULARY.index("uk")
    assert value("https://example.zzzz", "tld_id") == 0
    assert value("http://192.0.2.1/", "tld_id") == 0


def test_name_shape() -> None:
    assert value("https://abcde.com", "sld_length") == 5
    assert value("https://a1b2.com", "sld_digit_ratio") == 0.5
    assert value("https://a-b-c.com", "sld_hyphen_count") == 2
    assert value("https://aeiou.com", "sld_vowel_ratio") == 1.0
    assert value("https://bcdfgh.com", "sld_consonant_run") == 6


def test_path_shape() -> None:
    assert value("https://example.com/a/b/c", "path_depth") == 3
    assert value("https://example.com/aaaa/b", "max_segment_length") == 4
    assert value("https://example.com/a-b_c.d", "path_hyphen_count") == 1
    assert value("https://example.com/a-b_c.d", "path_underscore_count") == 1
    assert value("https://example.com/a-b_c.d", "path_dot_count") == 1
    assert value("https://example.com/%41%42", "path_percent_count") == 2


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("/", "none"),
        ("/page", "none"),
        ("/index.html", "html"),
        ("/login.php", "script"),
        ("/a.PHP", "script"),
        ("/file.zip", "other"),
    ],
)
def test_file_kind(path: str, kind: str) -> None:
    assert value(f"https://example.com{path}", "file_kind") == FILE_KINDS.index(kind)


def test_random_token() -> None:
    assert value("https://example.com/a1b2c3d4e5f6g7", "has_random_token") == 1
    assert value("https://example.com/0123456789abcdef", "has_random_token") == 1
    assert value("https://example.com/about-our-company", "has_random_token") == 0


def test_embedded_url_in_path() -> None:
    assert value("https://example.com/www.paypal.com/x", "path_has_embedded_url") == 1
    assert value("https://example.com/plain/path", "path_has_embedded_url") == 0


def test_keywords_are_whole_words() -> None:
    assert value("https://login-verify.example.com/", "host_keyword_count") == 2
    assert value("https://example.com/secure/login", "path_keyword_count") == 2
    assert value("https://example.com/passenger/updated", "path_keyword_count") == 0


def test_brand_position() -> None:
    assert value("https://paypal.example.com/", "brand_in_subdomain") == 1
    assert value("https://paypal-login.com/", "brand_in_name") == 1
    assert value("https://example.com/paypal/x", "brand_in_path") == 1


def test_brand_on_its_own_domain_is_not_impersonation() -> None:
    assert value("https://www.paypal.com/", "brand_in_name") == 0
    assert value("https://www.paypal.com/", "brand_official_domain") == 1


def test_short_brands_match_whole_words_only() -> None:
    assert value("https://attorney.example.com/", "brand_in_subdomain") == 0
    assert value("https://att.example.com/", "brand_in_subdomain") == 1
    assert value("https://example.com/pineapple", "brand_in_path") == 0


@pytest.mark.parametrize(
    "url", ["https://paypa1.com", "https://paypall.com", "https://micros0ft.com"]
)
def test_lookalikes_are_detected(url: str) -> None:
    assert value(url, "brand_lookalike") >= 1


@pytest.mark.parametrize("url", ["https://paypal.com", "https://example.com", "https://abc.com"])
def test_exact_or_distant_names_are_not_lookalikes(url: str) -> None:
    assert value(url, "brand_lookalike") == 0


def test_official_domains_use_registered_domain_syntax() -> None:
    for domains in BRANDS.values():
        assert all("." in domain and "/" not in domain for domain in domains)


def test_entropy() -> None:
    assert entropy("") == 0.0
    assert entropy("aaaa") == 0.0
    assert entropy("ab") == 1.0
    assert entropy("abcd") == 2.0


def test_artefact_features_stay_separate_from_the_model() -> None:
    assert not set(ARTEFACT_FEATURE_NAMES) & set(FEATURE_NAMES)
    assert len(artefact_features("https://www.example.com/?a=1&b=2#x")) == len(
        ARTEFACT_FEATURE_NAMES
    )


def test_artefact_values() -> None:
    row = dict(
        zip(
            ARTEFACT_FEATURE_NAMES,
            artefact_features("https://www.example.com:8443/?a=1&b=2#x"),
            strict=True,
        )
    )
    assert row["artefact_is_https"] == 1
    assert row["artefact_has_www"] == 1
    assert row["artefact_trailing_slash"] == 1
    assert row["artefact_query_params"] == 2
    assert row["artefact_has_fragment"] == 1
    assert row["artefact_has_port"] == 1


def test_random_strings_never_raise_and_stay_finite() -> None:
    random_source = random.Random(1234)
    alphabet = string.printable + "".join(map(chr, (0xE9, 0xFC, 0x416, 0x4E2D, 0x202E, 0xD7FF)))
    for _ in range(3000):
        text = "".join(random_source.choices(alphabet, k=random_source.randint(0, 80)))
        vector = extract(text)
        assert len(vector) == len(FEATURE_NAMES)
        assert all(math.isfinite(item) for item in vector)
