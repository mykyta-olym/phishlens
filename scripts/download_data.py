"""Download the datasets listed in data/sources.json and verify their checksums.

    uv run python scripts/download_data.py            # everything
    uv run python scripts/download_data.py hannousse  # one source

Plain files are checked against the SHA-256 their publisher lists. PhreshPhish is
about 36 GB because it stores page HTML, so this script reads only the string
columns of each Parquet file with HTTP range requests against a pinned commit. The
HTML column is never fetched. The result is checked against a pinned hash of its
uncompressed content, so gzip versions cannot change the outcome.

The only hosts contacted are the dataset publishers named in the manifest. No URL
from inside a dataset is ever requested. The rows are treated as plain strings.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
import time
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import fsspec
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "sources.json"
RAW_DIR = ROOT / "data" / "raw"
USER_AGENT = "phishlens-data-download/0.1"
CHUNK = 1 << 20
ATTEMPTS = 5
HTTP_TIMEOUT_SECONDS = 120


def with_retries[T](action: Callable[[], T], label: str) -> T:
    """Retry a network call with backoff. Transient 429 and 5xx responses are common."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return action()
        except OSError as error:
            if attempt == ATTEMPTS:
                raise
            wait = 2**attempt
            print(f"  retry {attempt} for {label} in {wait}s: {error}", file=sys.stderr)
            time.sleep(wait)
    raise AssertionError("unreachable")


def fetch(url: str) -> Iterator[bytes]:
    """Stream a manifest URL. The scheme is fixed to https by the check below."""
    if not url.startswith("https://"):
        raise ValueError(f"refusing a non-https source: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:  # noqa: S310
        while block := response.read(CHUNK):
            yield block


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def download_file(name: str, source: dict[str, Any], target_dir: Path) -> None:
    target = target_dir / source["output"]
    if target.exists() and file_sha256(target) == source["sha256"]:
        print(f"{name}: already present, checksum ok")
        return

    partial = target.with_suffix(target.suffix + ".part")

    def attempt() -> None:
        with partial.open("wb") as handle:
            for block in fetch(source["url"]):
                handle.write(block)

    with_retries(attempt, name)
    actual = file_sha256(partial)
    if actual != source["sha256"]:
        partial.unlink()
        raise SystemExit(f"{name}: checksum mismatch, got {actual}, want {source['sha256']}")
    partial.replace(target)
    print(f"{name}: downloaded {target.stat().st_size} bytes, checksum ok")


class HashingGzipCsv:
    """Write rows to a reproducible gzip file and hash the uncompressed bytes."""

    def __init__(self, path: Path) -> None:
        self.digest = hashlib.sha256()
        self.rows = 0
        self._raw = path.open("wb")
        self._gzip = gzip.GzipFile(filename="", fileobj=self._raw, mode="wb", mtime=0)
        self._buffer = io.StringIO(newline="")
        self._writer = csv.writer(self._buffer, lineterminator="\n")

    def write(self, row: list[str]) -> None:
        self._writer.writerow(row)
        encoded = self._buffer.getvalue().encode("utf-8", "surrogatepass")
        self._buffer.seek(0)
        self._buffer.truncate()
        self.digest.update(encoded)
        self._gzip.write(encoded)
        self.rows += 1

    def close(self) -> None:
        self._gzip.close()
        self._raw.close()


def read_columns(url: str, columns: list[str]) -> list[dict[str, Any]]:
    filesystem = fsspec.filesystem("http", block_size=CHUNK)
    with filesystem.open(url, "rb", cache_type="bytes") as handle:
        table = pq.ParquetFile(handle).read(columns=columns)
    rows: list[dict[str, Any]] = table.to_pylist()
    return rows


def extract_rows(source: dict[str, Any], target: Path) -> tuple[str, int]:
    base = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}"
    columns: list[str] = source["columns"]
    partial = target.with_suffix(target.suffix + ".part")
    writer = HashingGzipCsv(partial)
    writer.write(["split", *columns])
    for split, count in source["parquet_files"].items():
        for index in range(count):
            url = f"{base}/data/{split}-{index:03d}.parquet"
            rows = with_retries(lambda url=url: read_columns(url, columns), url)  # type: ignore[misc]
            for row in rows:
                values = [split]
                values += ["" if row[name] is None else str(row[name]) for name in columns]
                writer.write(values)
            print(f"  {split}-{index:03d}: {len(rows)} rows, {writer.rows - 1} total", flush=True)
    writer.close()
    partial.replace(target)
    return writer.digest.hexdigest(), writer.rows - 1


def fetch_text(url: str) -> str:
    return b"".join(fetch(url)).decode("ascii")


def extract_benchmarks(source: dict[str, Any], target: Path) -> tuple[str, int]:
    base = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}"
    partial = target.with_suffix(target.suffix + ".part")
    writer = HashingGzipCsv(partial)
    writer.write(["benchmark", "sha256"])
    lists = [("benign", "benign.txt"), ("full", "phish.txt")]
    for folder, count in source["benchmarks"].items():
        lists += [(f"{folder}/{index}", f"{folder}/{index}.txt") for index in range(count)]
    for position, (label, path) in enumerate(lists):
        url = f"{base}/benchmark/{path}"
        text = with_retries(lambda url=url: fetch_text(url), url)  # type: ignore[misc]
        for sha in text.split():
            writer.write([label, sha])
        if position % 100 == 0:
            print(f"  benchmark list {position + 1} of {len(lists)}", flush=True)
    writer.close()
    partial.replace(target)
    return writer.digest.hexdigest(), writer.rows - 1


def gz_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def check_pin(name: str, field: str, actual: str, source: dict[str, Any], update: bool) -> None:
    expected = source.get(field)
    if expected == actual:
        print(f"{name}: {field} ok")
    elif update:
        source[field] = actual
        print(f"{name}: recorded {field} {actual}")
    else:
        raise SystemExit(f"{name}: {field} mismatch, got {actual}, want {expected}")


def download_columns(
    name: str, source: dict[str, Any], target_dir: Path, update: bool, force: bool
) -> None:
    rows_path = target_dir / source["output"]
    if force or not rows_path.exists():
        digest, rows = extract_rows(source, rows_path)
        if rows != source["rows"]:
            raise SystemExit(f"{name}: got {rows} rows, want {source['rows']}")
    else:
        digest = gz_content_sha256(rows_path)
    check_pin(name, "content_sha256", digest, source, update)

    bench_path = target_dir / source["benchmark_output"]
    if force or not bench_path.exists():
        digest, rows = extract_benchmarks(source, bench_path)
        if update:
            source["benchmark_rows"] = rows
        elif rows != source["benchmark_rows"]:
            raise SystemExit(f"{name}: got {rows} benchmark rows")
    else:
        digest = gz_content_sha256(bench_path)
    check_pin(name, "benchmark_content_sha256", digest, source, update)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("names", nargs="*", help="sources to fetch, default all")
    parser.add_argument("--force", action="store_true", help="rebuild extracts that exist")
    parser.add_argument(
        "--update-pins", action="store_true", help="record new extract hashes in the manifest"
    )
    args = parser.parse_args()

    manifest: dict[str, dict[str, Any]] = json.loads(MANIFEST.read_text(encoding="utf-8"))
    names = args.names or list(manifest)
    unknown = sorted(set(names) - set(manifest))
    if unknown:
        parser.error(f"unknown source: {', '.join(unknown)}")

    for name in names:
        source = manifest[name]
        target_dir = RAW_DIR / name
        target_dir.mkdir(parents=True, exist_ok=True)
        if source["kind"] == "file":
            download_file(name, source, target_dir)
        else:
            download_columns(name, source, target_dir, args.update_pins, args.force)

    if args.update_pins:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
