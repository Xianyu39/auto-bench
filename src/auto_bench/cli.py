from __future__ import annotations

import argparse
import sys
from pathlib import Path

from auto_bench import __version__
from auto_bench.errors import AutobenchError
from auto_bench.renderer import render_file
from auto_bench.resolver import dump_yaml, resolve_file
from auto_bench.templates import TEMPLATES, get_template


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-bench",
        description="Automate TensorRT-LLM benchmark experiment orchestration.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    resolve_parser = subparsers.add_parser(
        "resolve",
        help="Resolve an autobench YAML file into command-plus-config cases.",
    )
    resolve_parser.add_argument("input", type=Path, help="Experiment YAML file.")
    resolve_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write resolved YAML to this path instead of stdout.",
    )

    render_parser = subparsers.add_parser(
        "render",
        help="Render an experiment YAML into cmd.sh and config.yaml artifacts.",
    )
    render_parser.add_argument("input", type=Path, help="Experiment YAML file.")
    render_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("artifacts/rendered"),
        help="Directory for rendered artifacts.",
    )

    template_parser = subparsers.add_parser(
        "template",
        help="Generate a starter experiment YAML template.",
    )
    template_parser.add_argument(
        "kind",
        choices=sorted(TEMPLATES),
        nargs="?",
        default="minimal",
        help="Template kind to generate.",
    )
    template_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write template YAML to this path instead of stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "resolve":
            rendered = dump_yaml(resolve_file(args.input))
            if args.output is None:
                sys.stdout.write(rendered)
            else:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered, encoding="utf-8")
            return 0
        if args.command == "render":
            case_dirs = render_file(args.input, args.output_dir)
            for case_dir in case_dirs:
                sys.stdout.write(f"{case_dir}\n")
            return 0
        if args.command == "template":
            rendered = dump_yaml(get_template(args.kind))
            if args.output is None:
                sys.stdout.write(rendered)
            else:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered, encoding="utf-8")
            return 0
    except AutobenchError as exc:
        parser.exit(2, f"auto-bench: error: {exc}\n")
    return 0
