from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

from auto_bench.resolver import dump_yaml, resolve_file


def render_file(input_path: str | Path, output_dir: str | Path) -> list[Path]:
    resolved = resolve_file(input_path)
    return render_resolved(resolved, output_dir)


def render_resolved(resolved: dict[str, Any], output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "resolved.yaml").write_text(dump_yaml(resolved), encoding="utf-8")

    cases = resolved.get("cases")
    if not isinstance(cases, list):
        raise TypeError("resolved payload must contain a cases list")

    case_dirs: list[Path] = []
    multi_case = len(cases) != 1
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError("case must be a mapping")
        case_id = str(case["case_id"])
        case_dir = root / case_id if multi_case else root
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_case_artifacts(case, case_dir)
        case_dirs.append(case_dir)

    if multi_case:
        _write_run_all(root, case_dirs)
    return case_dirs


def _write_case_artifacts(case: dict[str, Any], case_dir: Path) -> None:
    write_config = case["commands"].get("write_config")
    config_path: str | None = None
    if isinstance(write_config, dict):
        config_path = "config.yaml"
        (case_dir / config_path).write_text(
            dump_yaml(write_config["content"]), encoding="utf-8"
        )

    cmd = _cmd_script(case, config_path)
    cmd_path = case_dir / "cmd.sh"
    cmd_path.write_text(cmd, encoding="utf-8")
    cmd_path.chmod(cmd_path.stat().st_mode | 0o111)


def _write_run_all(root: Path, case_dirs: list[Path]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        "",
    ]
    for case_dir in case_dirs:
        relative = case_dir.relative_to(root)
        lines.append(f"bash \"$SCRIPT_DIR/{relative}/cmd.sh\"")
    lines.append("")

    run_all = root / "run_all.sh"
    run_all.write_text("\n".join(lines), encoding="utf-8")
    run_all.chmod(run_all.stat().st_mode | 0o111)


def _cmd_script(case: dict[str, Any], local_config_path: str | None) -> str:
    commands = case["commands"]
    benchmark_argv = list(commands["benchmark"]["argv"])
    if local_config_path is not None:
        benchmark_argv = _replace_config_path(benchmark_argv, "$SCRIPT_DIR/config.yaml")

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        "",
    ]

    prepare = commands.get("prepare_dataset")
    if isinstance(prepare, dict):
        output = str(prepare["output"])
        lines.extend(
            [
                f"if [ ! -f {_sh(output)} ]; then",
                f"  mkdir -p {_sh(str(Path(output).parent))}",
                _format_command(prepare["argv"], indent="  "),
                "fi",
                "",
            ]
        )

    lines.append(_format_command(benchmark_argv))
    lines.append("")
    return "\n".join(lines)


def _replace_config_path(argv: list[Any], config_path: str) -> list[str]:
    rendered = [str(item) for item in argv]
    if "--config" not in rendered:
        return rendered
    index = rendered.index("--config")
    if index + 1 >= len(rendered):
        return rendered
    rendered[index + 1] = config_path
    return rendered


def _format_command(argv: list[Any], indent: str = "") -> str:
    parts = _quote_args(argv)
    if not parts:
        return ""
    if len(parts) == 1:
        return f"{indent}{parts[0]}"
    groups = _group_args(parts[1:])
    lines = [f"{indent}{parts[0]} \\"]
    for index, group in enumerate(groups):
        suffix = " \\" if index < len(groups) - 1 else ""
        lines.append(f"{indent}  {group}{suffix}")
    return "\n".join(lines)


def _group_args(args: list[str]) -> list[str]:
    groups: list[str] = []
    index = 0
    while index < len(args):
        current = args[index]
        next_index = index + 1
        if current.startswith("--") and next_index < len(args):
            next_arg = args[next_index]
            if not next_arg.startswith("--"):
                groups.append(f"{current} {next_arg}")
                index += 2
                continue
        groups.append(current)
        index += 1
    return groups


def _quote_args(argv: list[Any]) -> list[str]:
    parts: list[str] = []
    for arg in argv:
        text = str(arg)
        if text.startswith("$SCRIPT_DIR/"):
            parts.append(f'"{text}"')
        else:
            parts.append(_sh(text))
    return parts


def _sh(value: str) -> str:
    return shlex.quote(os.fspath(value))
