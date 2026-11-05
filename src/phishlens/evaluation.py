"""The full evaluation. Writes every reported number to reports/metrics.json.

Run `phishlens train` first. The shipped model is scored through `Detector`, the same
code path the API uses, so the numbers describe what is actually served.

No URL from any dataset is fetched here. Reports hold aggregates only, so no live
phishing link is republished.
"""

from __future__ import annotations

import json
import platform
import re
import time
from collections.abc import Callable
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from phishlens.datasets import (
    DEFAULT_RAW_DIR,
    clean,
    load_hannousse,
    load_legitphish,
    load_phiusiil,
    load_phreshphish,
)
from phishlens.features import (
    ARTEFACT_FEATURE_NAMES,
    CATEGORICAL_FEATURES,
    FEATURE_INDEX,
    FEATURE_NAMES,
    FEATURES,
)
from phishlens.metrics import (
    calibration_table,
    domain_bootstrap,
    full_metrics,
    slice_metrics,
    threshold_for_max_f1,
    threshold_metrics,
)
from phishlens.model import Detector
from phishlens.training import (
    ROUNDS,
    SEED,
    VALIDATION_SHARE,
    artefact_matrix,
    domain_weights,
    feature_matrix,
    fit_booster,
    load_training_frame,
    split_by_domain,
    train_on,
)
from phishlens.urlparts import parse_url

Floats = NDArray[np.float64]
LEAKAGE_REPEATS = 5
LEAKAGE_TEST_SHARE = 0.2
SHUFFLE_ROUNDS = 100
SHUFFLE_REPEATS = 20
IMPORTANCE_SAMPLE = 20000
NGRAM_ALPHA = 1e-6
NGRAM_EPOCHS = 15
NGRAM_BITS = 20
BARE_HOST_RULE = re.compile(r"^https://www\.[^/]+/?$", re.IGNORECASE)
COMPROMISED_PATH = re.compile(r"/wp-(?:content|includes|admin)/", re.IGNORECASE)
BENCHMARK_RATES = {
    "benchmark-500": "5%",
    "benchmark-100": "1%",
    "benchmark-050": "0.5%",
    "benchmark-010": "0.1%",
    "benchmark-005": "0.05%",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def labels_of(frame: pd.DataFrame) -> NDArray[np.int8]:
    return frame["label"].to_numpy(dtype=np.int8)


def domains_of(frame: pd.DataFrame) -> NDArray[np.str_]:
    return frame["domain"].to_numpy(dtype=str)


def predict(booster: lgb.Booster, rows: NDArray[np.float32]) -> Floats:
    return np.asarray(booster.predict(rows), dtype=np.float64)


# Source audit ------------------------------------------------------------------------


def audit_source(frame: pd.DataFrame) -> dict[str, Any]:
    """Class rates of fields that should not separate phishing from benign."""
    parsed = [parse_url(url) for url in frame["url"]]
    table = pd.DataFrame(
        {
            "label": frame["label"].to_numpy(),
            "has_query": [bool(item.query) for item in parsed],
            "has_port": [item.port is not None for item in parsed],
            "no_scheme": [not item.scheme for item in parsed],
            "http": [item.scheme == "http" for item in parsed],
            "www": [item.host.startswith("www.") for item in parsed],
            "no_path": [item.path in ("", "/") for item in parsed],
            "ip_host": [item.host_kind != "domain" for item in parsed],
            "trailing_slash": [
                url.split("?", 1)[0].split("#", 1)[0].endswith("/") for url in frame["url"]
            ],
        }
    )
    rates = table.groupby("label").mean().round(4)
    predicted = ~frame["url"].str.match(BARE_HOST_RULE).to_numpy()
    rule = threshold_metrics(labels_of(frame), predicted.astype(np.float64), 0.5)
    return {
        "rows": len(frame),
        "phishing_rows": int(frame["label"].sum()),
        "benign_rates": rates.loc[0].to_dict(),
        "phishing_rates": rates.loc[1].to_dict(),
        "one_line_rule": {
            "description": "flag everything except https://www.<host> with no path",
            **rule,
        },
    }


def audit_sources(raw_dir: Path) -> dict[str, Any]:
    frames = {
        "phreshphish": load_phreshphish(raw_dir),
        "hannousse": load_hannousse(raw_dir),
        "phiusiil": load_phiusiil(raw_dir),
        "legitphish": load_legitphish(raw_dir),
    }
    audit = {name: audit_source(frame) for name, frame in frames.items()}
    phreshphish = frames["phreshphish"]
    audit["phreshphish"]["trailing_slash_by_split"] = {
        split: round(
            float(
                phreshphish[(phreshphish["split"] == split) & (phreshphish["label"] == 1)]["url"]
                .str.endswith("/")
                .mean()
            ),
            4,
        )
        for split in ("train", "test")
    }
    keys = {name: set(frame["url"]) for name, frame in frames.items()}
    audit["legitphish"]["urls_shared_with_phiusiil"] = len(keys["legitphish"] & keys["phiusiil"])
    return audit


# Baselines ---------------------------------------------------------------------------


def text_vectoriser() -> HashingVectorizer:
    return HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        n_features=2**NGRAM_BITS,
        alternate_sign=False,
        norm="l2",
    )


def linear_on_features() -> Any:
    numeric = [i for i, name in enumerate(FEATURE_NAMES) if name not in CATEGORICAL_FEATURES]
    categorical = [FEATURE_INDEX[name] for name in CATEGORICAL_FEATURES]
    columns = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return make_pipeline(columns, LogisticRegression(max_iter=300, C=1.0))


def run_baselines(
    train: pd.DataFrame,
    rows: NDArray[np.float32],
    fit: NDArray[np.intp],
    validation: NDArray[np.intp],
    test_sets: dict[str, tuple[pd.DataFrame, NDArray[np.float32]]],
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Fit each baseline on the rows the main model used, and pick cuts on validation."""
    labels = labels_of(train)
    results: dict[str, dict[str, Any]] = {}

    def report(name: str, scorer: Callable[[pd.DataFrame, NDArray[np.float32]], Floats]) -> None:
        cut = threshold_for_max_f1(
            labels[validation], scorer(train.iloc[validation], rows[validation])
        )
        results[name] = {
            "threshold": cut,
            **{
                split: full_metrics(labels_of(frame), scorer(frame, matrix), cut)
                for split, (frame, matrix) in test_sets.items()
            },
        }
        log(f"baseline {name}: done")

    base_rate = float(labels[fit].mean())
    report("majority_class", lambda frame, _: np.full(len(frame), base_rate))

    length = FEATURE_INDEX["host_length"], FEATURE_INDEX["path_length"]

    def total_length(matrix: NDArray[np.float32]) -> NDArray[np.float64]:
        return np.log1p(matrix[:, length[0]] + matrix[:, length[1]]).reshape(-1, 1)

    length_model = LogisticRegression(max_iter=200).fit(total_length(rows[fit]), labels[fit])
    report("url_length_only", lambda _, m: length_model.predict_proba(total_length(m))[:, 1])

    linear = linear_on_features().fit(rows[fit], labels[fit])
    report("logistic_regression", lambda _, m: linear.predict_proba(m)[:, 1])

    vectoriser = text_vectoriser()
    ngram = SGDClassifier(
        loss="log_loss", alpha=NGRAM_ALPHA, max_iter=NGRAM_EPOCHS, tol=None, random_state=seed
    )
    ngram.fit(vectoriser.transform(train["key"].iloc[fit]), labels[fit])

    def ngram_scores(frame: pd.DataFrame, _: NDArray[np.float32]) -> Floats:
        raw = ngram.decision_function(sparse.csr_matrix(vectoriser.transform(frame["key"])))
        return np.asarray(1.0 / (1.0 + np.exp(-raw)), dtype=np.float64)

    report("char_ngram_logistic", ngram_scores)
    return results


# Leakage -----------------------------------------------------------------------------


def measure_split(
    rows: NDArray[np.float32],
    labels: NDArray[np.int8],
    domains: NDArray[np.str_],
    fit: NDArray[np.intp],
    held: NDArray[np.intp],
    seed: int,
) -> dict[str, Any]:
    booster = fit_booster(rows[fit], labels[fit], domain_weights(domains[fit]), seed)
    scores = predict(booster, rows[held])
    result = full_metrics(labels[held], scores, 0.5)
    result["held_out_rows_on_a_seen_domain"] = round(
        float(np.isin(domains[held], np.unique(domains[fit])).mean()), 4
    )
    return result


def leakage_experiment(
    train: pd.DataFrame, rows: NDArray[np.float32], seed: int, repeats: int
) -> dict[str, Any]:
    """Same rows, model and held-out size. Only the way rows are split changes."""
    labels = labels_of(train)
    domains = domains_of(train)
    runs: dict[str, list[dict[str, Any]]] = {"random_rows": [], "by_domain": []}
    for repeat in range(repeats):
        run_seed = seed + repeat
        fit, held = split_by_domain(domains, LEAKAGE_TEST_SHARE, run_seed)
        runs["by_domain"].append(measure_split(rows, labels, domains, fit, held, run_seed))
        order = np.random.default_rng(run_seed).permutation(len(labels))
        random_held, random_fit = order[: len(held)], order[len(held) :]
        runs["random_rows"].append(
            measure_split(rows, labels, domains, random_fit, random_held, run_seed)
        )
        log(f"leakage repeat {repeat + 1} of {repeats}")

    def summarise(items: list[dict[str, Any]]) -> dict[str, Any]:
        names = (
            "roc_auc",
            "average_precision",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "held_out_rows_on_a_seen_domain",
        )
        return {
            name: {
                "mean": round(float(np.mean([item[name] for item in items])), 4),
                "std": round(float(np.std([item[name] for item in items])), 4),
                "values": [item[name] for item in items],
            }
            for name in names
        }

    summary: dict[str, Any] = {name: summarise(items) for name, items in runs.items()}
    summary["repeats"] = repeats
    summary["change"] = {
        name: round(summary["by_domain"][name]["mean"] - summary["random_rows"][name]["mean"], 4)
        for name in ("roc_auc", "average_precision", "precision", "recall", "f1")
    }
    return summary


# Artefacts ---------------------------------------------------------------------------


def add_tracking_query(url: str) -> str:
    return url + ("&" if "?" in url else "?") + "utm_source=newsletter"


def add_trailing_slash(url: str) -> str:
    return url if "?" in url or "#" in url else url.rstrip("/") + "/"


def strip_trailing_slash(url: str) -> str:
    return url if "?" in url or "#" in url else url.rstrip("/")


def use_plain_http(url: str) -> str:
    return "http://" + url.split("://", 1)[-1]


def add_www(url: str) -> str:
    scheme, separator, rest = url.rpartition("://")
    return f"{scheme}{separator}{rest if rest.startswith('www.') else 'www.' + rest}"


HARMLESS_EDITS: dict[str, Callable[[str], str]] = {
    "add ?utm_source=newsletter": add_tracking_query,
    "add a trailing slash": add_trailing_slash,
    "remove a trailing slash": strip_trailing_slash,
    "use http instead of https": use_plain_http,
    "add www.": add_www,
}


def artefact_experiment(
    train: pd.DataFrame,
    rows: NDArray[np.float32],
    test: pd.DataFrame,
    test_rows: NDArray[np.float32],
    detector: Detector,
    seed: int,
) -> dict[str, Any]:
    """Train the same model with the excluded fields added, then edit benign URLs."""
    train_extra = artefact_matrix(train["url"].tolist())
    test_extra = artefact_matrix(test["url"].tolist())
    names = (*FEATURE_NAMES, *ARTEFACT_FEATURE_NAMES)
    naive = train_on(train, np.hstack([rows, train_extra]), seed, names=names)
    labels = labels_of(test)
    naive_scores = predict(naive.booster, np.hstack([test_rows, test_extra]))
    shipped_scores = np.array(detector.score(test["url"].tolist()))

    benign = test[labels == 0]["url"].tolist()
    benign_rows = test_rows[labels == 0]
    naive_flagged = float((naive_scores[labels == 0] >= naive.threshold).mean())
    shipped_flagged = float((shipped_scores[labels == 0] >= detector.threshold).mean())

    edits = {}
    for name, edit in HARMLESS_EDITS.items():
        # A few rows end in stray whitespace, which an appended edit would pull inside.
        edited = [edit(url.strip()) for url in benign]
        edited_rows = np.hstack([benign_rows, artefact_matrix(edited)])
        edited_naive = predict(naive.booster, edited_rows)
        edited_shipped = np.array(detector.score(edited))
        edits[name] = {
            "naive_flagged": round(float((edited_naive >= naive.threshold).mean()), 4),
            "shipped_flagged": round(float((edited_shipped >= detector.threshold).mean()), 4),
        }
        log(f"edit: {name}")

    return {
        "naive_model": full_metrics(labels, naive_scores, naive.threshold),
        "shipped_model": full_metrics(labels, shipped_scores, detector.threshold),
        "benign_rows_edited": len(benign),
        "benign_flagged_before_edit": {
            "naive_flagged": round(naive_flagged, 4),
            "shipped_flagged": round(shipped_flagged, 4),
        },
        "benign_flagged_after_edit": edits,
    }


# External datasets -------------------------------------------------------------------


def external_rows(frame: pd.DataFrame, seen: NDArray[np.str_]) -> pd.DataFrame:
    """Clean an outside dataset and drop every domain seen in training."""
    cleaned, _ = clean(frame)
    return cleaned[~cleaned["domain"].isin(seen)].reset_index(drop=True)


def external_test(
    frame: pd.DataFrame, seen: NDArray[np.str_], detector: Detector, seed: int
) -> dict[str, Any]:
    cleaned, report = clean(frame)
    kept = external_rows(frame, seen)
    labels = labels_of(kept)
    scores = np.array(detector.score(kept["url"].tolist()))
    return {
        "cleaning": report.as_dict(),
        "rows_removed_as_seen_in_training": len(cleaned) - len(kept),
        "metrics": full_metrics(labels, scores, detector.threshold),
        "interval_95": domain_bootstrap(labels, scores, domains_of(kept), detector.threshold, seed),
    }


# Base rates --------------------------------------------------------------------------


def base_rate_table(
    raw_test: pd.DataFrame,
    scores: Floats,
    lists: pd.DataFrame,
    threshold: float,
    seed: int,
) -> dict[str, Any]:
    """Precision at realistic prevalence, on the maintainers' published benchmark sets.

    Each set is the shared benign list plus one list of phishing rows. This measures
    precision at those base rates rather than extrapolating it.
    """
    by_sha = pd.Series(scores, index=raw_test["sha256"].to_numpy())
    benign_scores = by_sha.reindex(lists[lists["benchmark"] == "benign"]["sha256"]).dropna()
    false_alarms = int((benign_scores >= threshold).sum())
    table: dict[str, Any] = {
        "benign_rows": len(benign_scores),
        "false_positives": false_alarms,
        "false_positive_rate": round(false_alarms / len(benign_scores), 5),
    }
    random = np.random.default_rng(seed)
    for folder, rate in BENCHMARK_RATES.items():
        members = lists[lists["benchmark"].str.startswith(folder + "/")]
        precisions, recalls = [], []
        for _, group in members.groupby("benchmark"):
            phish = by_sha.reindex(group["sha256"]).dropna().to_numpy()
            hits = int((phish >= threshold).sum())
            precisions.append(hits / (hits + false_alarms) if hits + false_alarms else 0.0)
            recalls.append(hits / len(phish))
        means = [float(np.mean(random.choice(precisions, len(precisions)))) for _ in range(2000)]
        table[rate] = {
            "benchmarks": len(precisions),
            "precision_mean": round(float(np.mean(precisions)), 4),
            "precision_interval_95": [
                round(float(np.percentile(means, 2.5)), 4),
                round(float(np.percentile(means, 97.5)), 4),
            ],
            "recall_mean": round(float(np.mean(recalls)), 4),
        }
    return table


# Failure slices ----------------------------------------------------------------------


def slice_masks(frame: pd.DataFrame, rows: NDArray[np.float32], mixed: set[str]) -> dict[str, Any]:
    urls = frame["url"]
    domains = frame["domain"]
    platform_host = rows[:, FEATURE_INDEX["is_platform_host"]] == 1
    no_path = rows[:, FEATURE_INDEX["path_length"]] == 0
    return {
        "url shorteners": rows[:, FEATURE_INDEX["is_shortener"]] == 1,
        "shared hosting platforms": platform_host,
        "domains a listed brand owns": rows[:, FEATURE_INDEX["brand_official_domain"]] == 1,
        "looks compromised: WordPress path": urls.str.contains(COMPROMISED_PATH).to_numpy(),
        "looks compromised: domain also has benign rows": (
            domains.isin(mixed).to_numpy() & ~platform_host
        ),
        "bare host, no path": no_path,
        "benign pages with a credential word in the path": (
            rows[:, FEATURE_INDEX["path_keyword_count"]] > 0
        ),
        "IP address hosts": rows[:, FEATURE_INDEX["is_ip_host"]] == 1,
        "long URLs (over 100 characters)": (urls.str.len() > 100).to_numpy(),
    }


def failure_slices(
    frame: pd.DataFrame,
    rows: NDArray[np.float32],
    scores: Floats,
    threshold: float,
    mixed: set[str],
) -> dict[str, Any]:
    labels = labels_of(frame)
    return {
        name: slice_metrics(labels[mask], scores[mask], threshold)
        for name, mask in slice_masks(frame, rows, mixed).items()
    }


# Checks and importance ---------------------------------------------------------------


def shuffled_label_check(
    train: pd.DataFrame,
    rows: NDArray[np.float32],
    test: pd.DataFrame,
    test_rows: NDArray[np.float32],
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    """Refit on shuffled labels, so the training labels carry no information.

    Scores should then fall to chance. Several seeds are used because one run moves
    a long way: a model fitted to noise is still a function of the features, and test
    rows cluster strongly by feature.
    """
    weights = np.ones(len(train), dtype=np.float32)
    values = []
    for repeat in range(repeats):
        labels = labels_of(train).copy()
        np.random.default_rng(seed + repeat).shuffle(labels)
        booster = fit_booster(rows, labels, weights, seed + repeat, rounds=SHUFFLE_ROUNDS)
        values.append(full_metrics(labels_of(test), predict(booster, test_rows), 0.5)["roc_auc"])
    return {"roc_auc_mean": round(float(np.mean(values)), 4), "roc_auc_values": values}


def ablation(
    train: pd.DataFrame,
    rows: NDArray[np.float32],
    test: pd.DataFrame,
    test_rows: NDArray[np.float32],
    seed: int,
) -> dict[str, Any]:
    """Refit without each feature group, to show what the score depends on."""
    domains = domains_of(train)
    labels = labels_of(train)
    fit, _ = split_by_domain(domains, VALIDATION_SHARE, seed)
    weights = domain_weights(domains[fit])
    groups = {
        f"without {group}": {f.name for f in FEATURES if f.group == group}
        for group in sorted({feature.group for feature in FEATURES})
    }
    groups["without tld_id"] = {"tld_id"}
    groups["without is_platform_host"] = {"is_platform_host"}
    groups["without tld_id and is_platform_host"] = {"tld_id", "is_platform_host"}
    results = {}
    for name, removed in groups.items():
        kept = [i for i, feature in enumerate(FEATURE_NAMES) if feature not in removed]
        names = [FEATURE_NAMES[i] for i in kept]
        booster = fit_booster(rows[fit][:, kept], labels[fit], weights, seed, names=names)
        scores = predict(booster, test_rows[:, kept])
        metrics = full_metrics(labels_of(test), scores, 0.5)
        results[name] = {key: metrics[key] for key in ("roc_auc", "average_precision", "f1")}
        log(f"ablation: {name}")
    return results


def feature_importance(detector: Detector, urls: list[str], seed: int) -> list[dict[str, Any]]:
    """Mean absolute TreeSHAP contribution on a sample of held-out rows."""
    chosen = np.random.default_rng(seed).permutation(len(urls))[:IMPORTANCE_SAMPLE]
    _, values = detector._contributions([urls[i] for i in chosen])
    means = np.abs(values[:, :-1]).mean(axis=0)
    order = np.argsort(-means)
    return [
        {"feature": FEATURE_NAMES[i], "mean_abs_log_odds": round(float(means[i]), 4)} for i in order
    ]


def measure_throughput(detector: Detector, urls: list[str]) -> dict[str, Any]:
    """Single-thread URLs per second through the same call the API makes."""
    sample = urls[:5000]
    detector.assess(sample[:100])
    started = time.perf_counter()
    detector.assess(sample)
    elapsed = time.perf_counter() - started
    return {
        "urls": len(sample),
        "seconds": round(elapsed, 3),
        "urls_per_second": round(len(sample) / elapsed),
        "machine": f"{platform.system()} {platform.machine()}",
    }


# Orchestration -----------------------------------------------------------------------


def versions() -> dict[str, str]:
    names = ("lightgbm", "numpy", "pandas", "publicsuffixlist", "scikit-learn")
    return {name: package_metadata.version(name) for name in names} | {
        "python": platform.python_version()
    }


def run_evaluation(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output: Path = Path("reports/metrics.json"),
    seed: int = SEED,
    quick: bool = False,
) -> dict[str, Any]:
    detector = Detector.load()
    train, test, cleaning = load_training_frame(raw_dir)
    leaked_keys = set(train["key"]) & set(test["key"])
    if leaked_keys:
        raise RuntimeError(f"{len(leaked_keys)} test URLs also appear in training")
    log(f"train rows {len(train)}, test rows {len(test)}")

    rows = feature_matrix(train["url"].tolist())
    test_rows = feature_matrix(test["url"].tolist())
    labels = labels_of(train)
    domains = domains_of(train)
    fit, validation = split_by_domain(domains, VALIDATION_SHARE, seed)

    seen = np.unique(domains)
    unseen_mask = ~test["domain"].isin(seen).to_numpy()
    unseen = test[unseen_mask].reset_index(drop=True)
    unseen_rows = test_rows[unseen_mask]
    log(f"test rows on unseen domains {len(unseen)}")

    test_scores = np.array(detector.score(test["url"].tolist()))
    unseen_scores = test_scores[unseen_mask]
    threshold = detector.threshold
    strict = float(detector.metadata["strict_threshold"])

    main = {
        "threshold": threshold,
        "strict_threshold": strict,
        "future_unseen_domains": {
            "metrics": full_metrics(labels_of(unseen), unseen_scores, threshold),
            "interval_95": domain_bootstrap(
                labels_of(unseen), unseen_scores, domains_of(unseen), threshold, seed
            ),
            "strict_metrics": threshold_metrics(labels_of(unseen), unseen_scores, strict),
            "calibration": calibration_table(labels_of(unseen), unseen_scores),
        },
        "future_all_domains": {
            "metrics": full_metrics(labels_of(test), test_scores, threshold),
            "interval_95": domain_bootstrap(
                labels_of(test), test_scores, domains_of(test), threshold, seed
            ),
            "strict_metrics": threshold_metrics(labels_of(test), test_scores, strict),
        },
        "validation": full_metrics(
            labels[validation],
            np.array(detector.score(train["url"].iloc[validation].tolist())),
            threshold,
        ),
    }
    log("main model scored")

    report: dict[str, Any] = {
        "generated_with": versions(),
        "seed": seed,
        "quick_run": quick,
        "data": {
            "cleaning": cleaning,
            "train_rows": len(train),
            "train_phishing_share": round(float(labels.mean()), 4),
            "fit_rows": len(fit),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "test_rows_on_unseen_domains": len(unseen),
            "test_keys_also_in_train": len(leaked_keys),
            "train_dates": [str(train["date"].min()), str(train["date"].max())],
            "test_dates": [str(test["date"].min()), str(test["date"].max())],
            "boosting_rounds": ROUNDS,
        },
        "main": main,
    }

    report["source_audit"] = audit_sources(raw_dir)
    log("sources audited")

    outside = {
        name: external_rows(loader(raw_dir), seen)
        for name, loader in (("hannousse", load_hannousse), ("phiusiil", load_phiusiil))
    }
    report["baselines"] = run_baselines(
        train,
        rows,
        fit,
        validation,
        {
            "validation": (train.iloc[validation], rows[validation]),
            "future_unseen_domains": (unseen, unseen_rows),
            "future_all_domains": (test, test_rows),
            **{
                name: (frame, feature_matrix(frame["url"].tolist()))
                for name, frame in outside.items()
            },
        },
        seed,
    )

    repeats = 1 if quick else LEAKAGE_REPEATS
    report["leakage"] = leakage_experiment(train, rows, seed, repeats)
    report["artefacts"] = artefact_experiment(train, rows, unseen, unseen_rows, detector, seed)
    log("artefact experiment done")

    report["external"] = {
        "hannousse": external_test(load_hannousse(raw_dir), seen, detector, seed),
        "phiusiil": external_test(load_phiusiil(raw_dir), seen, detector, seed),
    }
    log("external sets done")

    raw = load_phreshphish(raw_dir)
    raw_test = raw[raw["split"] == "test"].reset_index(drop=True)
    raw_scores = np.array(detector.score(raw_test["url"].tolist()))
    lists = pd.read_csv(raw_dir / "phreshphish" / "benchmarks.csv.gz", dtype=str)
    report["base_rates"] = {
        "threshold": base_rate_table(raw_test, raw_scores, lists, threshold, seed),
        "strict_threshold": base_rate_table(raw_test, raw_scores, lists, strict, seed),
    }
    log("base rates done")

    both = pd.concat([train, test])
    per_domain = both.groupby("domain")["label"].nunique()
    mixed = set(per_domain[per_domain > 1].index)
    report["slices"] = {
        "future_all_domains": failure_slices(test, test_rows, test_scores, threshold, mixed),
        "future_unseen_domains": failure_slices(
            unseen, unseen_rows, unseen_scores, threshold, mixed
        ),
    }

    report["checks"] = {
        "shuffled_labels": shuffled_label_check(
            train, rows, unseen, unseen_rows, seed, 2 if quick else SHUFFLE_REPEATS
        ),
    }
    report["ablation"] = ablation(train, rows, unseen, unseen_rows, seed)
    report["importance"] = feature_importance(detector, unseen["url"].tolist(), seed)
    report["throughput"] = measure_throughput(detector, unseen["url"].tolist())

    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2) + "\n"
    output.write_text(text, encoding="utf-8", newline="\n")
    log(f"wrote {output}")
    return report
