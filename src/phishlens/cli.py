"""Command line interface. `phishlens --help` lists the commands."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from phishlens import __version__
from phishlens.model import DEFAULT_TOP_FEATURES, Assessment, Detector, ModelError
from phishlens.validation import InvalidUrlError, validate_url

EXIT_OK = 0
EXIT_FLAGGED = 1
EXIT_USAGE = 2
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def _read_lines(path: str) -> list[str]:
    if path == "-":
        return sys.stdin.read().splitlines()
    return Path(path).read_text(encoding="utf-8").splitlines()


def _collect_urls(args: argparse.Namespace) -> list[str]:
    urls: list[str] = list(args.urls)
    if args.file:
        urls += [line.strip() for line in _read_lines(args.file) if line.strip()]
    return urls


def _render(result: Assessment) -> Iterable[str]:
    yield f"{result.score:.3f}  {result.verdict}  {result.url}"
    for item in result.contributions:
        direction = "towards phishing" if item.contribution > 0 else "away from phishing"
        yield f"  {item.contribution:+.2f} {direction}: {item.description} ({item.value})"
    for note in result.notes:
        yield f"  note: {note}"


def _score(args: argparse.Namespace) -> int:
    urls = _collect_urls(args)
    if not urls:
        print("error: give at least one URL, or --file", file=sys.stderr)
        return EXIT_USAGE
    try:
        urls = [validate_url(url) for url in urls]
    except InvalidUrlError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE

    results = _load(args).assess(urls, top=args.top)
    if args.json:
        print(json.dumps([item.as_dict() for item in results], indent=2))
    else:
        for result in results:
            print("\n".join(_render(result)))
    flagged = any(item.verdict == "likely_phishing" for item in results)
    return EXIT_FLAGGED if args.fail_on_flag and flagged else EXIT_OK


def _load(args: argparse.Namespace) -> Detector:
    return Detector.load(args.model_dir) if args.model_dir else Detector.load()


def _serve(args: argparse.Namespace) -> int:
    import uvicorn  # noqa: PLC0415

    from phishlens.api import create_app  # noqa: PLC0415

    uvicorn.run(create_app(args.model_dir), host=args.host, port=args.port)
    return EXIT_OK


def _train(args: argparse.Namespace) -> int:
    from phishlens.training import run_training  # noqa: PLC0415

    run_training(raw_dir=args.raw_dir, output_dir=args.output_dir, seed=args.seed)
    return EXIT_OK


def _evaluate(args: argparse.Namespace) -> int:
    from phishlens.evaluation import run_evaluation  # noqa: PLC0415

    run_evaluation(raw_dir=args.raw_dir, output=args.output, seed=args.seed, quick=args.quick)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phishlens",
        description="Score URLs for phishing from the string alone. URLs are never fetched.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    score = commands.add_parser("score", help="score URLs and explain each score")
    score.add_argument("urls", nargs="*", help="URLs to score")
    score.add_argument("--file", help="read URLs from a file, one per line, or - for stdin")
    score.add_argument("--json", action="store_true", help="print JSON")
    score.add_argument("--top", type=int, default=DEFAULT_TOP_FEATURES, help="features to show")
    score.add_argument("--model-dir", type=Path, help="use a model from this directory")
    score.add_argument(
        "--fail-on-flag", action="store_true", help="exit 1 when any URL is likely phishing"
    )
    score.set_defaults(handler=_score)

    serve = commands.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--host", default=os.environ.get("PHISHLENS_HOST", DEFAULT_HOST))
    serve.add_argument(
        "--port", type=int, default=int(os.environ.get("PHISHLENS_PORT", DEFAULT_PORT))
    )
    serve.add_argument("--model-dir", type=Path, help="use a model from this directory")
    serve.set_defaults(handler=_serve)

    train = commands.add_parser("train", help="train the model (needs the train dependencies)")
    train.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    train.add_argument("--output-dir", type=Path, default=None)
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(handler=_train)

    evaluate = commands.add_parser("evaluate", help="run the full evaluation")
    evaluate.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    evaluate.add_argument("--output", type=Path, default=Path("reports/metrics.json"))
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--quick", action="store_true", help="fewer repeats, for a smoke run")
    evaluate.set_defaults(handler=_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code: int = args.handler(args)
    except (ModelError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    return code


if __name__ == "__main__":
    sys.exit(main())
