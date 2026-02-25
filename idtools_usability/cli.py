from __future__ import annotations

import argparse
from collections.abc import Callable
from importlib import import_module

RunnerPath = tuple[str, str]


RUNNERS: dict[str, RunnerPath] = {
    "reliability": ("scripts.analyze_reliability", "main"),
    "phase3-themes": ("scripts.analyze_phase3_themes", "main"),
    "phase3": ("scripts.analyze_phase3_themes", "main"),
    "themes": ("scripts.analyze_phase3_themes", "main"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="idtools_usability",
        description="Run usability analysis workflows.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run an analysis workflow.")
    run_parser.add_argument(
        "target",
        choices=sorted(RUNNERS),
        help="Workflow to run.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "run":
        parser.print_help()
        return 1

    module_name, fn_name = RUNNERS[args.target]
    runner = _load_runner(module_name, fn_name)
    runner()
    return 0


def _load_runner(module_name: str, fn_name: str) -> Callable[[], None]:
    module = import_module(module_name)
    fn = getattr(module, fn_name)
    return fn


if __name__ == "__main__":
    raise SystemExit(main())
