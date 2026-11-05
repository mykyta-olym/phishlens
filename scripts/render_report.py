"""Render reports/EVALUATION.md and the README results block from reports/metrics.json.

    uv run python scripts/render_report.py          # write both
    uv run python scripts/render_report.py --check  # fail if either is out of date

Numbers in those documents come only from metrics.json, so a figure cannot appear
there without having been measured by `phishlens evaluate`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "reports" / "metrics.json"
PREVIOUS = ROOT / "reports" / "before_parser_fix.json"
EVALUATION = ROOT / "reports" / "EVALUATION.md"
README = ROOT / "README.md"
START = "<!-- results:start -->"
END = "<!-- results:end -->"


def num(value: float | None, places: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def pct(value: float | None, places: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{places}f}%"


def table(header: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    head = list(header)
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def span(interval: list[float]) -> str:
    return f"{num(interval[0])} to {num(interval[1])}"


def headline(report: dict[str, Any]) -> str:
    main = report["main"]
    unseen = main["future_unseen_domains"]
    every = main["future_all_domains"]
    leak = report["leakage"]
    rows = []
    for name, block in (("Later dates, unseen domains", unseen), ("Later dates, all", every)):
        metrics = block["metrics"]
        rows.append(
            [
                name,
                f"{metrics['rows']:,}",
                num(metrics["precision"]),
                num(metrics["recall"]),
                num(metrics["f1"]),
                num(metrics["roc_auc"]),
            ]
        )
    for name, key in (("Random row split", "random_rows"), ("Split by domain", "by_domain")):
        block = leak[key]
        rows.append(
            [
                f"{name}, training period",
                "20% of domains" if key == "by_domain" else "same size",
                num(block["precision"]["mean"]),
                num(block["recall"]["mean"]),
                num(block["f1"]["mean"]),
                num(block["roc_auc"]["mean"]),
            ]
        )
    for name, key in (("Hannousse 2020", "hannousse"), ("PhiUSIIL 2023", "phiusiil")):
        metrics = report["external"][key]["metrics"]
        rows.append(
            [
                f"{name}, other dataset",
                f"{metrics['rows']:,}",
                num(metrics["precision"]),
                num(metrics["recall"]),
                num(metrics["f1"]),
                num(metrics["roc_auc"]),
            ]
        )
    return table(["Test", "Rows", "Precision", "Recall", "F1", "ROC AUC"], rows)


def render_readme_block(report: dict[str, Any]) -> str:
    main = report["main"]
    metrics = main["future_unseen_domains"]["metrics"]
    matrix = metrics["confusion"]
    rates = report["base_rates"]["threshold"]
    strict = report["base_rates"]["strict_threshold"]
    unseen_key = "future_unseen_domains"
    baselines = report["baselines"]
    best = max(
        (name for name in baselines if name != "majority_class"),
        key=lambda name: baselines[name][unseen_key]["roc_auc"],
    )
    slices = report["slices"]["future_all_domains"]
    artefact = report["artefacts"]
    tracking = artefact["benign_flagged_after_edit"]["add ?utm_source=newsletter"]
    parts = [
        headline(report),
        "",
        f"Confusion matrix on later dates, unseen domains, at threshold {main['threshold']}: "
        f"{matrix['tp']:,} true positives, {matrix['fp']:,} false positives, "
        f"{matrix['tn']:,} true negatives, {matrix['fn']:,} false negatives.",
        "",
        f"Best baseline on the top row's data: `{best}`, ROC AUC "
        f"{num(baselines[best][unseen_key]['roc_auc'])}, "
        f"F1 {num(baselines[best][unseen_key]['f1'])}. "
        f"On Hannousse it reached ROC AUC {num(baselines[best]['hannousse']['roc_auc'])}, "
        f"and on PhiUSIIL {num(baselines[best]['phiusiil']['roc_auc'])}.",
        "",
        f"Harmless edit test on {artefact['benign_rows_edited']:,} benign URLs: a naive model "
        f"that sees the excluded fields scores F1 {num(artefact['naive_model']['f1'])} on the "
        f"top row's data, and flags {pct(artefact['benign_flagged_before_edit']['naive_flagged'])} "
        f"of them. Add `?utm_source=newsletter` and it flags {pct(tracking['naive_flagged'])}. "
        f"The shipped model flags {pct(tracking['shipped_flagged'])} either way.",
        "",
        "Precision on the dataset's own benchmark sets at realistic prevalence:",
        "",
        table(
            [
                "Phishing share",
                "Precision, default cut",
                "Recall",
                "Precision, strict cut",
                "Recall",
            ],
            [
                [
                    rate,
                    num(rates[rate]["precision_mean"]),
                    num(rates[rate]["recall_mean"]),
                    num(strict[rate]["precision_mean"]),
                    num(strict[rate]["recall_mean"]),
                ]
                for rate in ("5%", "1%", "0.1%")
            ],
        ),
        "",
        "Measured failures on later dates:",
        "",
        table(
            ["Slice", "Phishing rows", "Recall", "Benign rows", "False positive rate"],
            [
                [
                    name,
                    f"{item['phishing_rows']:,}",
                    pct(item["recall"]),
                    f"{item['benign_rows']:,}",
                    pct(item["false_positive_rate"]),
                ]
                for name, item in slices.items()
                if name
                in (
                    "url shorteners",
                    "looks compromised: WordPress path",
                    "looks compromised: domain also has benign rows",
                    "shared hosting platforms",
                    "bare host, no path",
                )
            ],
        ),
    ]
    return "\n".join(parts)


def source_audit(report: dict[str, Any]) -> str:
    rows = []
    for name, item in report["source_audit"].items():
        benign = item["benign_rates"]
        phishing = item["phishing_rates"]
        rule = item["one_line_rule"]
        rows.append(
            [
                name,
                f"{item['rows']:,}",
                pct(benign["no_path"]),
                pct(phishing["no_path"]),
                pct(benign["has_query"], 2),
                pct(phishing["has_query"], 2),
                pct(phishing["ip_host"]),
                num(rule["precision"]),
                num(rule["recall"]),
            ]
        )
    return table(
        [
            "Source",
            "Rows",
            "Benign, no path",
            "Phishing, no path",
            "Benign, query",
            "Phishing, query",
            "Phishing, IP host",
            "Rule precision",
            "Rule recall",
        ],
        rows,
    )


def main_results(report: dict[str, Any]) -> str:
    main = report["main"]
    rows = []
    for name, key in (
        ("later dates, unseen domains", "future_unseen_domains"),
        ("later dates, all rows", "future_all_domains"),
    ):
        block = main[key]
        metrics = block["metrics"]
        interval = block["interval_95"]
        matrix = metrics["confusion"]
        strict = block["strict_metrics"]
        rows.append(
            [
                name,
                f"{metrics['rows']:,}",
                f"{num(metrics['precision'])} ({span(interval['precision'])})",
                f"{num(metrics['recall'])} ({span(interval['recall'])})",
                f"{num(metrics['f1'])} ({span(interval['f1'])})",
                f"{num(metrics['roc_auc'])} ({span(interval['roc_auc'])})",
                num(metrics["average_precision"]),
                pct(metrics["false_positive_rate"]),
                f"TP {matrix['tp']:,}, FP {matrix['fp']:,}, "
                f"TN {matrix['tn']:,}, FN {matrix['fn']:,}",
                f"P {num(strict['precision'])}, R {num(strict['recall'])}, "
                f"FPR {pct(strict['false_positive_rate'])}",
            ]
        )
    return table(
        [
            "Test set",
            "Rows",
            "Precision",
            "Recall",
            "F1",
            "ROC AUC",
            "Avg precision",
            "FPR",
            "Confusion matrix",
            "Strict cut",
        ],
        rows,
    )


def baselines(report: dict[str, Any]) -> str:
    main = report["main"]
    outside = report["external"]
    rows = [
        [
            "LightGBM (shipped)",
            num(main["validation"]["roc_auc"]),
            num(main["future_unseen_domains"]["metrics"]["f1"]),
            num(main["future_unseen_domains"]["metrics"]["roc_auc"]),
            num(main["future_all_domains"]["metrics"]["roc_auc"]),
            num(outside["hannousse"]["metrics"]["roc_auc"]),
            num(outside["phiusiil"]["metrics"]["roc_auc"]),
        ]
    ]
    for name, item in report["baselines"].items():
        rows.append(
            [
                name,
                num(item["validation"]["roc_auc"]),
                num(item["future_unseen_domains"]["f1"]),
                num(item["future_unseen_domains"]["roc_auc"]),
                num(item["future_all_domains"]["roc_auc"]),
                num(item["hannousse"]["roc_auc"]),
                num(item["phiusiil"]["roc_auc"]),
            ]
        )
    return table(
        [
            "Model",
            "AUC, validation",
            "F1, unseen",
            "AUC, unseen",
            "AUC, all later",
            "AUC, Hannousse",
            "AUC, PhiUSIIL",
        ],
        rows,
    )


def leakage(report: dict[str, Any]) -> str:
    leak = report["leakage"]
    rows = []
    for metric in ("roc_auc", "average_precision", "precision", "recall", "f1"):
        random_rows = leak["random_rows"][metric]
        by_domain = leak["by_domain"][metric]
        rows.append(
            [
                metric,
                f"{num(random_rows['mean'])} +/- {num(random_rows['std'])}",
                f"{num(by_domain['mean'])} +/- {num(by_domain['std'])}",
                f"{leak['change'][metric]:+.3f}",
            ]
        )
    seen = leak["random_rows"]["held_out_rows_on_a_seen_domain"]["mean"]
    seen_domain = leak["by_domain"]["held_out_rows_on_a_seen_domain"]["mean"]
    return "\n".join(
        [
            table(["Metric", "Random rows", "By registered domain", "Change"], rows),
            "",
            f"Mean of {leak['repeats']} seeds at a cut of 0.5. Held-out rows on a domain also in "
            f"the fit rows: {pct(seen)} with random rows, {pct(seen_domain)} by domain.",
        ]
    )


def artefacts(report: dict[str, Any]) -> str:
    block = report["artefacts"]
    naive = block["naive_model"]
    shipped = block["shipped_model"]
    scores = table(
        ["Model", "Precision", "Recall", "F1", "ROC AUC"],
        [
            [
                name,
                num(item["precision"]),
                num(item["recall"]),
                num(item["f1"]),
                num(item["roc_auc"]),
            ]
            for name, item in (("naive, sees excluded fields", naive), ("shipped", shipped))
        ],
    )
    before = block["benign_flagged_before_edit"]
    rows = [["no edit", pct(before["naive_flagged"]), pct(before["shipped_flagged"])]]
    for name, item in block["benign_flagged_after_edit"].items():
        rows.append([name, pct(item["naive_flagged"]), pct(item["shipped_flagged"])])
    edits = table(["Edit to benign URLs", "Naive flagged", "Shipped flagged"], rows)
    return f"{scores}\n\n{edits}"


def external(report: dict[str, Any]) -> str:
    rows = []
    for name, item in report["external"].items():
        metrics = item["metrics"]
        interval = item["interval_95"]
        rows.append(
            [
                name,
                f"{metrics['rows']:,}",
                f"{item['rows_removed_as_seen_in_training']:,}",
                num(metrics["precision"]),
                num(metrics["recall"]),
                f"{num(metrics['f1'])} ({span(interval['f1'])})",
                f"{num(metrics['roc_auc'])} ({span(interval['roc_auc'])})",
                pct(metrics["false_positive_rate"]),
            ]
        )
    return table(
        [
            "Dataset",
            "Rows",
            "Removed, domain seen in training",
            "Precision",
            "Recall",
            "F1",
            "ROC AUC",
            "FPR",
        ],
        rows,
    )


def base_rates(report: dict[str, Any]) -> str:
    sections = []
    for title, key in (
        ("Default threshold", "threshold"),
        ("Strict threshold", "strict_threshold"),
    ):
        block = report["base_rates"][key]
        rows = [
            [
                rate,
                block[rate]["benchmarks"],
                f"{num(block[rate]['precision_mean'])} "
                f"({span(block[rate]['precision_interval_95'])})",
                num(block[rate]["recall_mean"]),
            ]
            for rate in ("5%", "1%", "0.5%", "0.1%", "0.05%")
        ]
        sections.append(
            f"{title}: {block['false_positives']:,} false positives on "
            f"{block['benign_rows']:,} benign rows "
            f"(FPR {pct(block['false_positive_rate'], 2)}).\n\n"
            + table(["Phishing share", "Benchmarks", "Precision (95%)", "Recall"], rows)
        )
    return "\n\n".join(sections)


def slices(report: dict[str, Any]) -> str:
    sections = []
    for title, key in (
        ("Later dates, all rows", "future_all_domains"),
        ("Later dates, unseen domains", "future_unseen_domains"),
    ):
        rows = [
            [
                name,
                f"{item['phishing_rows']:,}",
                pct(item["recall"]),
                f"{item['benign_rows']:,}",
                pct(item["false_positive_rate"]),
                num(item["median_score"]),
            ]
            for name, item in report["slices"][key].items()
        ]
        sections.append(
            f"{title}:\n\n"
            + table(
                ["Slice", "Phishing", "Recall", "Benign", "FPR", "Median score"],
                rows,
            )
        )
    return "\n\n".join(sections)


def ablation(report: dict[str, Any]) -> str:
    base = report["main"]["future_unseen_domains"]["metrics"]
    rows = [["all features (shipped)", num(base["roc_auc"]), num(base["average_precision"])]]
    rows += [
        [name, num(item["roc_auc"]), num(item["average_precision"])]
        for name, item in report["ablation"].items()
    ]
    return table(["Feature set", "ROC AUC", "Avg precision"], rows)


def importance(report: dict[str, Any]) -> str:
    rows = [[item["feature"], num(item["mean_abs_log_odds"])] for item in report["importance"][:12]]
    return table(["Feature", "Mean absolute log-odds"], rows)


def calibration(report: dict[str, Any]) -> str:
    rows = [
        [
            item["band"],
            f"{item['rows']:,}",
            num(item["mean_score"]),
            num(item["observed_phishing_share"]),
        ]
        for item in report["main"]["future_unseen_domains"]["calibration"]
    ]
    return table(["Score band", "Rows", "Mean score", "Observed phishing share"], rows)


def shuffled(report: dict[str, Any]) -> str:
    check = report["checks"]["shuffled_labels"]
    values = check["roc_auc_values"]
    above = sum(value > 0.5 for value in values)
    below = sum(value < 0.5 for value in values)
    return (
        f"With shuffled training labels, ROC AUC on the test rows over {len(values)} seeds had "
        f"mean {num(check['roc_auc_mean'])}, from {num(min(values))} to {num(max(values))}, with "
        f"{above} seeds above 0.5 and {below} below. So the real score depends on the real "
        f"labels. The spread is wide for this many rows, probably because a model fitted to "
        f"noise is still a function of strongly clustered features, so one seed proves little. "
        f"The first seed alone gave {num(values[0])}."
    )


def retraining_note(report: dict[str, Any]) -> str:
    """Disclose that the test split was scored before the final retrain."""
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    rows = []
    for name, key in (
        ("later dates, unseen domains", "future_unseen_domains"),
        ("later dates, all rows", "future_all_domains"),
    ):
        before = previous[key]
        after = report["main"][key]["metrics"]
        for metric in ("precision", "recall", "f1", "roc_auc"):
            rows.append(
                [
                    name,
                    metric,
                    num(before[metric]),
                    num(after[metric]),
                    f"{after[metric] - before[metric]:+.3f}",
                ]
            )
    return (
        "The test split was scored on an earlier model before this one. The harmless-edit "
        "check in that run exposed a parsing bug: hosts such as `www30.example.com` lost part "
        "of their subdomain, so adding `www.` changed the features. The bug was fixed and the "
        "model retrained with unchanged settings. The fix came from an equality test, not from "
        "a score, and no feature or setting was chosen because of test results. It also "
        "moved the seeded validation split, so the threshold moved from "
        f"{previous['threshold']} to {report['main']['threshold']}. That means this test split is "
        "no longer strictly unseen by the author, and the table below shows what changed.\n\n"
        + table(["Test set", "Metric", "Before fix", "Final", "Change"], rows)
    )


def render_evaluation(report: dict[str, Any]) -> str:
    data = report["data"]
    cleaning = data["cleaning"]
    throughput = report["throughput"]
    return f"""# Evaluation

Generated by `scripts/render_report.py` from `reports/metrics.json`. Do not edit by hand.
Every figure below came from `uv run phishlens evaluate` with seed {report["seed"]}. Reports
hold aggregates only, so no phishing URL is republished here.

## Protocol

The model was fitted on PhreshPhish `train` rows dated {data["train_dates"][0]} to
{data["train_dates"][1]}. The held-out test uses PhreshPhish `test` rows dated
{data["test_dates"][0]} to {data["test_dates"][1]}, which the model never saw. The headline
test also removes every row whose registered domain occurs anywhere in the training split.

Design decisions used validation rows from the training period only. The audit below
looked at class rates over all rows before features were written, so the choice of
excluded fields is not blind to the test split, but no test score informed it. Thresholds
were chosen on validation domains.

{retraining_note(report)}

Rows: {cleaning["rows_in"]:,} in, {cleaning["duplicates"]:,} duplicates removed after
canonicalising, {cleaning["label_conflicts"]:,} rows dropped because one canonical URL carried
both labels, {cleaning["rows_out"]:,} left. That gives {data["train_rows"]:,} training rows
({data["fit_rows"]:,} fit, {data["validation_rows"]:,} validation) and {data["test_rows"]:,} test
rows, of which {data["test_rows_on_unseen_domains"]:,} are on unseen domains. Canonical URLs
shared by train and test: {data["test_keys_also_in_train"]}.

Intervals are 95 percent, from resampling whole registered domains, because rows on one
domain are not independent.

## Audit of the sources

The rule flags everything except `https://www.<host>` with no path. It uses no learning.

{source_audit(report)}

## Held-out results

{main_results(report)}

## Baselines

Each baseline used the same fit rows, and a threshold picked the same way on validation.
The model design was fixed before the test split was first scored. During development the n-gram
baseline and LightGBM looked close on a time-based cut of the training data, a comparison
not recorded in metrics.json, and LightGBM was kept because it explains a score in named
URL properties. The table shows the n-gram baseline ahead on every column. Swapping it in
now would turn the test split into a selection set, so the result is reported instead.

{baselines(report)}

## Leakage: random rows against registered domains

Same rows, same model and same held-out size. Only the split changes.

{leakage(report)}

## Why some fields are excluded

The naive model adds the scheme, `www.`, trailing slash, query, fragment and port. Both
models are scored on later dates and unseen domains. The second table edits benign URLs
in ways that do not change where they lead, then counts how many get flagged.

{artefacts(report)}

## Other datasets

The thresholds were not retuned. Rows on any domain seen in training were removed.

{external(report)}

## Realistic base rates

These are the benchmark sets published by the PhreshPhish maintainers: one shared benign
list plus lists of phishing rows at each prevalence. They contain some domains seen in
training, as a deployment would. Precision is measured on those sets, not extrapolated.

{base_rates(report)}

## Where it fails

"Looks compromised" is a proxy, since no dataset labels compromise. It means a phishing
row with a WordPress system path, or a row on a non-platform domain that also has benign
rows in the data.

{slices(report)}

## Checks

{shuffled(report)}

Removing feature groups, on later dates and unseen domains:

{ablation(report)}

Mean contribution on unseen-domain test rows:

{importance(report)}

Calibration on later dates and unseen domains. The score is not a real-world probability:
the test sets have far more phishing than real traffic.

{calibration(report)}

Throughput: {throughput["urls_per_second"]:,} URLs per second on one thread, with explanations,
over {throughput["urls"]:,} URLs on {throughput["machine"]}.

Versions: {", ".join(f"{name} {value}" for name, value in report["generated_with"].items())}.
"""


def replace_block(text: str, block: str) -> str:
    before, _, rest = text.partition(START)
    _, separator, after = rest.partition(END)
    if not separator:
        raise SystemExit(f"README.md must contain {START} and {END}")
    return f"{before}{START}\n{block}\n{END}{after}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render reports from metrics.json.")
    parser.add_argument("--check", action="store_true", help="fail if a document is out of date")
    args = parser.parse_args()

    report = json.loads(METRICS.read_text(encoding="utf-8"))
    readme = README.read_text(encoding="utf-8")
    outputs = {
        EVALUATION: render_evaluation(report),
        README: replace_block(readme, render_readme_block(report)),
    }
    if args.check:
        stale = [
            path.name
            for path, text in outputs.items()
            if not path.exists() or path.read_text(encoding="utf-8") != text
        ]
        if stale:
            print(f"out of date: {', '.join(stale)}", file=sys.stderr)
            return 1
        print("reports match metrics.json")
        return 0
    for path, text in outputs.items():
        path.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
