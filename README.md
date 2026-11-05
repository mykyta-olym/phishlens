# phishlens

Scores how likely a URL is to be phishing, using only the URL string, and says which
parts of the string drove the score. It is a personal project. The URL is never fetched
or resolved, so it is safe to run on anything, but it also means the tool cannot see the
page, the certificate or the domain's reputation.

**It must not be the only control.** A low score means this string gave no strong signal.
It does not mean the link is safe, so the API says `no_strong_signal` and never `safe`.

The most useful part of this repository may be the evaluation, because the obvious way to
build this gives a score that looks excellent and means very little.

## The trap in public phishing datasets

The first dataset I looked at, PhiUSIIL, is widely used. Every legitimate URL in it is a
bare `https://www.` homepage. A one-line rule with no learning nearly solves it, and the
same rule falls apart on a dataset with ordinary URLs. A model trained there learns what
the collection script did, not what phishing looks like.

So training uses PhreshPhish, where benign URLs come from real browsing. That dataset has
traps of its own. Anonymised telemetry removed almost every query string from the benign
class, so "has a query string" looks like a superb phishing signal. It is not one. A
model that reads it flags ordinary pages for carrying `?utm_source=newsletter`. The
trailing slash rate of the phishing feed also changed between the train and test periods.

The feature extractor therefore never sees the scheme, `www.`, the query string, the
fragment, the port or a trailing slash. The evaluation trains a naive model with those
fields added, so the cost and the benefit are both measured instead of argued about.

## Results

All figures below are generated from `reports/metrics.json` by `scripts/render_report.py`,
and a test fails if they drift. The full report, with intervals, baselines, ablations and
calibration, is in [reports/EVALUATION.md](reports/EVALUATION.md).

<!-- results:start -->
| Test | Rows | Precision | Recall | F1 | ROC AUC |
| --- | --- | --- | --- | --- | --- |
| Later dates, unseen domains | 67,709 | 0.915 | 0.828 | 0.869 | 0.942 |
| Later dates, all | 162,734 | 0.929 | 0.868 | 0.897 | 0.973 |
| Random row split, training period | same size | 0.954 | 0.906 | 0.929 | 0.989 |
| Split by domain, training period | 20% of domains | 0.933 | 0.872 | 0.901 | 0.979 |
| Hannousse 2020, other dataset | 7,133 | 0.683 | 0.613 | 0.646 | 0.700 |
| PhiUSIIL 2023, other dataset | 179,143 | 0.493 | 0.678 | 0.571 | 0.734 |

Confusion matrix on later dates, unseen domains, at threshold 0.435: 32,140 true positives, 2,973 false positives, 25,897 true negatives, 6,699 false negatives.

Best baseline on the top row's data: `char_ngram_logistic`, ROC AUC 0.959, F1 0.901. On Hannousse it reached ROC AUC 0.744, and on PhiUSIIL 0.768.

Harmless edit test on 28,870 benign URLs: a naive model that sees the excluded fields scores F1 0.922 on the top row's data, and flags 5.9% of them. Add `?utm_source=newsletter` and it flags 99.3%. The shipped model flags 10.3% either way.

Precision on the dataset's own benchmark sets at realistic prevalence:

| Phishing share | Precision, default cut | Recall | Precision, strict cut | Recall |
| --- | --- | --- | --- | --- |
| 5% | 0.458 | 0.827 | 0.807 | 0.692 |
| 1% | 0.139 | 0.827 | 0.445 | 0.693 |
| 0.1% | 0.016 | 0.827 | 0.073 | 0.693 |

Measured failures on later dates:

| Slice | Phishing rows | Recall | Benign rows | False positive rate |
| --- | --- | --- | --- | --- |
| url shorteners | 189 | 99.5% | 1 | 100.0% |
| shared hosting platforms | 13,906 | 99.8% | 255 | 65.1% |
| looks compromised: WordPress path | 704 | 57.2% | 140 | 6.4% |
| looks compromised: domain also has benign rows | 1,671 | 48.5% | 6,233 | 4.3% |
| bare host, no path | 45,899 | 92.8% | 8,064 | 33.6% |
<!-- results:end -->

How to read this:

- The top row is the honest one. It uses data from after training ended, on registered
  domains the model never saw.
- The two middle rows are the leakage check. They use the same rows and model, and change
  only how the rows are split, so the difference is what a shared domain is worth.
- The last two rows are other collections of URLs, with every domain seen in training
  removed. Scores fall a long way. They are the better guide to traffic that does not look
  like the training data.
- The precision column of the first table is not what you would see in production. Those
  test sets are roughly half phishing and real traffic is not. The second table measures
  that directly: at realistic prevalence, most flags are false alarms, even at the strict
  threshold. Treat a flag as a reason to look, not as a finding.

## A simple baseline beat the shipped model

Logistic regression on hashed character n-grams scored higher than LightGBM on every set
in the report, including both outside datasets. So the model that ships is not the best
scorer here.

During development the two looked close on a time-based cut of the training data, and
LightGBM explains a score in named URL properties, such as "a brand name appears in the
subdomain", where n-grams would give fragments of text. I froze it on that basis. The full
comparison only came from the evaluation, and by then the test split had been scored.
Picking a model because it won on that split would have turned it into a selection set,
so the result is reported instead. The next version should adopt the n-gram model or
combine the two, and evaluate on fresh data.

One retrain did happen after the test split was first scored. The harmless-edit check
exposed a parsing bug, which I fixed. No setting changed, and the report lists exactly how
far each headline number moved.

## Run it

You need [uv](https://docs.astral.sh/uv/). The trained model is in the repository, so
scoring needs no dataset.

```sh
uv sync
uv run phishlens score --top 3 "http://paypal.secure-login.example.com/verify/account.php"
```

```text
0.756  likely_phishing  http://paypal.secure-login.example.com/verify/account.php
  +1.92 towards phishing: a brand name appears in the subdomain (yes)
  +0.93 towards phishing: length of the host name (31)
  +0.48 towards phishing: credential or alert words in the path (2)
```

Each line is an exact TreeSHAP contribution in log-odds. Use `--json` for scripts,
`--file urls.txt` for a list, and `--fail-on-flag` to exit 1 when anything is flagged.

Run the service:

```sh
uv run phishlens serve --port 8000
curl -s localhost:8000/v1/score -H 'content-type: application/json' \
  -d '{"url": "https://example.org/about/team", "top": 3}'
curl -s localhost:8000/v1/score/batch -H 'content-type: application/json' \
  -d '{"urls": ["https://example.org/", "http://192.0.2.1/login"]}'
```

| Route | Purpose |
| --- | --- |
| `POST /v1/score` | One URL, with explanation |
| `POST /v1/score/batch` | Up to 1000 URLs, order kept |
| `GET /v1/model` | Model version, threshold and stated limits |
| `GET /health` | Liveness and model version |
| `GET /docs` | Generated API documentation |

Input is validated before scoring. Only `http`, `https` or a bare host are accepted, a URL
may be 8192 characters, and bodies have a size limit. One bad URL in a batch rejects the
batch with a 422 that gives its position. Error bodies do not echo the input, and URLs are
not logged, because real URLs carry tokens.

Or with Docker:

```sh
docker build -t phishlens .
docker run --rm -p 8000:8000 phishlens
```

The image has no training stack and runs as a non-root user with a read-only root
filesystem if you ask for one. `sh scripts/docker_smoke.sh` builds it and tests all of this.

## Reproduce the evaluation

```sh
uv run python scripts/download_data.py    # about 130 MB, checksums verified
uv run phishlens train                    # fixed seed, a few minutes
uv run phishlens evaluate                 # writes reports/metrics.json
uv run python scripts/render_report.py    # rewrites the tables from it
```

PhreshPhish is about 36 GB because it stores page HTML. The download reads only its string
columns with HTTP range requests against a pinned commit, so no page body reaches your
machine. Two training runs here produced a byte-identical model. Sources, licenses and
changes made to the data are in [data/README.md](data/README.md).

## How it is built

- `urlparts.py` parses any string without raising. It handles IP forms such as
  `0x7f.0.0.1`, `user@host` tricks and missing schemes. It uses the bundled Public Suffix
  List, so there is no network access at run time.
- `features.py` turns the canonical URL into numbers: host structure, randomness of the
  name, path shape, brand and credential words, and near-misspellings of brands.
- LightGBM does the scoring. The model is stored as text, not as a pickle, so loading a
  model file cannot run code.
- `model.py` checks the model's hash and feature list at startup and re-scores stored URLs.
  A mismatch stops the service rather than letting it serve wrong scores.
- `evaluation.py` scores test rows through the same `Detector` class the API uses, so the
  numbers describe what is served.

Splits group rows by registered domain using ICANN suffixes only. That puts `a.github.io`
and `b.github.io` in one group, so one platform cannot sit on both sides of a split. URLs
that differ only in fields the model ignores are also treated as duplicates.

## How it was tested

- `uv run pytest` covers the parser, features, validation, metrics, model loading, API,
  CLI, cleaning, checksums and the agreement between these documents and the measurements.
- Tests assert that scoring opens no socket, and that harmless edits to a URL do not move
  its score.
- `uv run ruff format --check`, `uv run ruff check` and `uv run mypy` (strict) all pass, and
  CI runs the same checks plus the Docker smoke test.
- Mutation checks: I broke 11 behaviours one at a time in a scratch copy, such as letting
  the query string into the features. Ten were caught by a failing assertion. The eleventh,
  skipping the model checksum, was caught because LightGBM's native parser then aborted the
  test process on the damaged file, which is why the hash is checked before parsing.
- Refitting on shuffled labels over 20 seeds gives scores centred on chance, so the result
  depends on the real labels. Single seeds vary widely, so one run would prove little.

## What it does not do

- **Compromised legitimate sites.** A phishing page on a real, old, ordinary domain looks
  like an ordinary URL. The string has nothing to give away, and the table above shows it.
- **URL shorteners.** The destination is hidden, so the string has nothing to say about it.
  The test data holds a single benign shortener row, so the model learned to flag nearly
  all of them and the false alarm rate cannot be measured here. Expect a high score on any
  short link, benign or not. The output adds a note saying so.
- **Ordinary homepages.** Bare hosts are common in phishing and rare in the benign telemetry,
  so a normal homepage is flagged far too often. The table above and the PhiUSIIL row, whose
  legitimate URLs are all homepages, agree on this. This is the same kind of artefact as the
  query string, only weaker, and it is still in the model.
- **Shared platforms.** Benign pages on hosts such as `github.io` are rare in the training
  data and phishing on them is common, so most legitimate pages there get flagged. The
  output adds a note for these too.
- **Traffic unlike the training data.** The outside datasets show how much of the score is
  specific to one collection. Even URL length points the opposite way on them, as the
  baseline table in the report shows.
- **Query strings.** They are ignored for the reason above, so an open redirect or a
  phishing signal that lives only in the query is invisible.
- **Time.** Phishing changes quickly, and the ROC AUC gap between the by-domain row and the
  top row suggests what later dates cost. A saved model decays and needs retraining.
- **Probabilities.** The score ranks URLs. It is not a real-world probability, because the
  training data has far more phishing than real traffic does.
- **Label noise.** The phishing class comes from community feeds and holds pages that do
  not look like phishing, so no figure here is exact.
- **IP addresses and ports.** The benign class has no IP hosts, so an IP host will score
  high whatever it is, including a router admin page.

Use it to rank URLs for review or as one signal among several, next to page analysis,
reputation feeds, mail authentication and user reporting.

## License

The code is MIT. The datasets keep their own licenses, listed in
[data/README.md](data/README.md).
