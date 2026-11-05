# Data

Nothing in `data/raw/` is committed. `uv run python scripts/download_data.py` rebuilds it
and refuses any file whose checksum differs from `sources.json`. No URL from inside a
dataset is ever requested. The rows are read as plain strings.

| Source | Use here | License | Pin |
| --- | --- | --- | --- |
| [PhreshPhish](https://huggingface.co/datasets/phreshphish/phreshphish) | training, validation, main test | CC BY 4.0, and the card says it should only be used for anti-phishing research | commit plus a SHA-256 of the extract |
| [Hannousse and Yahiouche](https://data.mendeley.com/datasets/c2gw7fy2j4/3) | external test | CC BY 4.0 | publisher's SHA-256 |
| [PhiUSIIL](https://data.mendeley.com/datasets/shwpxscxy2/2) | audit, external test | CC BY 4.0 | publisher's SHA-256 |
| [LegitPhish](https://data.mendeley.com/datasets/hx4m73v2sf/1) | audit only | CC BY 4.0 | publisher's SHA-256 |

Full citations are in `sources.json`.

## Changes made to the data

CC BY asks that changes be stated. This project:

- reads only the URL and label columns, plus split, date, target and row ID for PhreshPhish
- skips the PhreshPhish `html`, `lang` and `lang_score` columns, which are never downloaded
- drops one LegitPhish row with no label
- removes rows that become identical once scheme, `www.`, query, fragment, port and trailing
  slash are ignored, and removes rows where that form carries both labels

The saved model is derived from PhreshPhish. It contains no URLs from the dataset, but its
use should respect the dataset's terms. The MIT license covers this code, not the datasets.

## Why PhreshPhish is the training source

`reports/EVALUATION.md` has the measurements. In short, a one-line rule with no learning
nearly solves PhiUSIIL, because its legitimate URLs are all bare `https://www.` homepages.
Most of LegitPhish's phishing class is raw-IP malware delivery, and it shares its benign
URLs with PhiUSIIL. Hannousse is realistic but small. PhreshPhish has benign URLs from
real browsing, and a test split that lies strictly after its train split, though it has
artefacts of its own, described in the README.
