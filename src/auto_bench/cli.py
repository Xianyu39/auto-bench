from __future__ import annotations

import argparse
import sys
from pathlib import Path

from auto_bench import __version__
from auto_bench.collector import collect_results, render_results
from auto_bench.errors import AutobenchError
from auto_bench.renderer import render_resolved
from auto_bench.resolver import dump_yaml, resolve_file
from auto_bench.runner import run_cases
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
        aliases=["plan"],
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
    render_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Render run_all.sh so failed cases are logged and later cases still run."
        ),
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Render an experiment YAML and run the generated scripts.",
    )
    run_parser.add_argument("input", type=Path, help="Experiment YAML file.")
    run_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("artifacts/rendered"),
        help="Directory for rendered artifacts.",
    )
    run_parser.add_argument(
        "--profile",
        action="store_true",
        help="Run generated profile.sh/profile_all.sh instead of cmd.sh/run_all.sh.",
    )
    run_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Render controller scripts so failed cases are logged and later "
            "cases still run."
        ),
    )
    run_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Print case events instead of rendering the interactive progress view.",
    )

    collect_parser = subparsers.add_parser(
        "collect_results",
        aliases=["collect-results"],
        help="Collect benchmark metrics from rendered artifacts.",
    )
    collect_parser.add_argument(
        "artifact_dir",
        type=Path,
        help="Rendered artifact directory containing resolved.yaml and run.log files.",
    )
    collect_parser.add_argument(
        "--framework",
        required=True,
        choices=["trtllm-bench"],
        help="Benchmark framework whose output should be parsed.",
    )
    collect_parser.add_argument(
        "--format",
        choices=["csv", "yaml"],
        default="csv",
        help="Output format for collected results.",
    )
    collect_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write collected results to this path instead of stdout.",
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
            resolved = resolve_file(args.input)
            _emit_warnings(resolved)
            rendered = dump_yaml(resolved)
            if args.output is None:
                sys.stdout.write(rendered)
            else:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered, encoding="utf-8")
            return 0
        if args.command in {"render", "plan"}:
            resolved = resolve_file(args.input)
            _emit_warnings(resolved)
            case_dirs = render_resolved(
                resolved,
                args.output_dir,
                continue_on_error=args.continue_on_error,
            )
            for case_dir in case_dirs:
                sys.stdout.write(f"{case_dir}\n")
            return 0
        if args.command == "run":
            resolved = resolve_file(args.input)
            _emit_warnings(resolved)
            case_dirs = render_resolved(
                resolved,
                args.output_dir,
                continue_on_error=args.continue_on_error,
            )
            return run_cases(
                resolved,
                args.output_dir,
                case_dirs,
                profile=args.profile,
                continue_on_error=args.continue_on_error,
                progress=not args.no_progress,
            )
        if args.command in {"collect_results", "collect-results"}:
            rows = collect_results(args.artifact_dir, args.framework)
            rendered = render_results(rows, args.format)
            if args.output is None:
                sys.stdout.write(rendered)
            else:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered, encoding="utf-8")
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


def _emit_warnings(resolved: dict[str, object]) -> None:
    warnings = resolved.get("warnings")
    if not isinstance(warnings, list):
        return
    for warning in warnings:
        sys.stderr.write(f"{warning}\n")
