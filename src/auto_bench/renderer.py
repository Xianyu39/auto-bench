from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

from auto_bench.resolver import dump_yaml, resolve_file


def render_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    continue_on_error: bool = False,
) -> list[Path]:
    resolved = resolve_file(input_path)
    return render_resolved(resolved, output_dir, continue_on_error=continue_on_error)


def render_resolved(
    resolved: dict[str, Any],
    output_dir: str | Path,
    *,
    continue_on_error: bool = False,
) -> list[Path]:
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
        _write_run_all(root, cases, case_dirs, continue_on_error=continue_on_error)
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


def _write_run_all(
    root: Path,
    cases: list[Any],
    case_dirs: list[Path],
    *,
    continue_on_error: bool,
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail" if continue_on_error else "set -euo pipefail",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'LOG_FILE="$SCRIPT_DIR/run.log"',
        ': > "$LOG_FILE"',
        'exec > >(tee -a "$LOG_FILE") 2>&1',
        "",
    ]
    if continue_on_error:
        lines.extend(
            [
                "FAILED=0",
                "run_case() {",
                "  local case_name=\"$1\"",
                "  local case_script=\"$2\"",
                "  if bash \"$case_script\"; then",
                "    return 0",
                "  fi",
                "  local status=$?",
                '  echo "auto-bench: case failed: ${case_name} (exit ${status})"',
                "  FAILED=1",
                "  return 0",
                "}",
                "",
            ]
        )
    for index, case_dir in enumerate(case_dirs):
        relative = case_dir.relative_to(root)
        if continue_on_error:
            lines.append(
                f"run_case {_sh(str(relative))} "
                f"\"$SCRIPT_DIR/{relative}/cmd.sh\""
            )
        else:
            lines.append(f"bash \"$SCRIPT_DIR/{relative}/cmd.sh\"")
        if index < len(case_dirs) - 1:
            gap = _metadata_gap(cases[index])
            if gap > 0:
                lines.append(f"sleep {_sh(str(gap))}")
    if continue_on_error:
        lines.append('exit "$FAILED"')
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
        'LOG_FILE="$SCRIPT_DIR/run.log"',
        ': > "$LOG_FILE"',
        'exec > >(tee -a "$LOG_FILE") 2>&1',
        "",
    ]
    lines.extend(_environment_lines(case.get("metadata", {})))
    lines.extend(_gpu_frequency_lines(case.get("metadata", {})))

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


def _metadata_gap(case: Any) -> int | float:
    if not isinstance(case, dict):
        return 0
    metadata = case.get("metadata", {})
    if not isinstance(metadata, dict):
        return 0
    gap = metadata.get("gap", 0)
    if isinstance(gap, int | float) and gap > 0:
        return gap
    return 0


def _environment_lines(metadata: Any) -> list[str]:
    if not isinstance(metadata, dict):
        return []
    env = metadata.get("env")
    if not isinstance(env, dict) or not env:
        return []
    lines = [f"export {key}={_sh(str(value))}" for key, value in env.items()]
    lines.append("")
    return lines


def _gpu_frequency_lines(metadata: Any) -> list[str]:
    if not isinstance(metadata, dict):
        return []
    config = metadata.get("gpu_frequency")
    if config in (None, False):
        return []

    min_mhz: int | float | None
    max_mhz: int | float | None
    gpu_ids: list[Any] | None

    if isinstance(config, int | float):
        min_mhz = config
        max_mhz = config
        gpu_ids = None
    elif isinstance(config, dict):
        if config.get("enabled", True) is False:
            return []
        min_mhz = config.get("min_mhz", config.get("mhz"))
        max_mhz = config.get("max_mhz", min_mhz)
        ids = config.get("gpu_ids")
        gpu_ids = ids if isinstance(ids, list) else None
    else:
        return []

    if min_mhz is None or max_mhz is None:
        return []

    commands: list[str] = []
    clocks = f"{min_mhz},{max_mhz}"
    if gpu_ids:
        for gpu_id in gpu_ids:
            commands.append(
                _format_command(["nvidia-smi", "-i", gpu_id, "-lgc", clocks])
            )
    else:
        commands.append(_format_command(["nvidia-smi", "-lgc", clocks]))
    commands.append("")
    return commands


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
        if current.startswith("-") and next_index < len(args):
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
        if _is_script_dir_path(text):
            parts.append(f'"{text}"')
        else:
            parts.append(_sh(text))
    return parts


def _sh(value: str) -> str:
    if _is_script_dir_path(value):
        return f'"{value}"'
    return shlex.quote(os.fspath(value))


def _is_script_dir_path(value: str) -> bool:
    return value == "$SCRIPT_DIR" or value.startswith("$SCRIPT_DIR/")
