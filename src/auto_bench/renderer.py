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
        _write_profile_all(root, cases, case_dirs, continue_on_error=continue_on_error)
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

    nsys = _nsys_config(case.get("nsys"))
    if nsys is not None:
        profile = _profile_script(nsys["prefix"])
        profile_path = case_dir / "profile.sh"
        profile_path.write_text(profile, encoding="utf-8")
        profile_path.chmod(profile_path.stat().st_mode | 0o111)


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


def _write_profile_all(
    root: Path,
    cases: list[Any],
    case_dirs: list[Path],
    *,
    continue_on_error: bool,
) -> None:
    profile_cases = [
        (case, case_dir)
        for case, case_dir in zip(cases, case_dirs, strict=True)
        if isinstance(case, dict) and _nsys_config(case.get("nsys")) is not None
    ]
    if not profile_cases:
        return

    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail" if continue_on_error else "set -euo pipefail",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'LOG_FILE="$SCRIPT_DIR/profile_all.log"',
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
                (
                    '  echo "auto-bench: profile case failed: '
                    '${case_name} (exit ${status})"'
                ),
                "  FAILED=1",
                "  return 0",
                "}",
                "",
            ]
        )
    for index, (case, case_dir) in enumerate(profile_cases):
        relative = case_dir.relative_to(root)
        if continue_on_error:
            lines.append(
                f"run_case {_sh(str(relative))} "
                f"\"$SCRIPT_DIR/{relative}/profile.sh\""
            )
        else:
            lines.append(f"bash \"$SCRIPT_DIR/{relative}/profile.sh\"")
        if index < len(profile_cases) - 1:
            gap = _metadata_gap(case)
            if gap > 0:
                lines.append(f"sleep {_sh(str(gap))}")
    if continue_on_error:
        lines.append('exit "$FAILED"')
    lines.append("")

    profile_all = root / "profile_all.sh"
    profile_all.write_text("\n".join(lines), encoding="utf-8")
    profile_all.chmod(profile_all.stat().st_mode | 0o111)


def _cmd_script(case: dict[str, Any], local_config_path: str | None) -> str:
    commands = case["commands"]
    benchmark_argv = list(commands["benchmark"]["argv"])
    if local_config_path is not None:
        benchmark_argv = _replace_config_path(benchmark_argv, "$SCRIPT_DIR/config.yaml")
    metadata = case.get("metadata", {})
    benchmark_argv = _run_dir_argv(benchmark_argv)

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'RUN_DIR="${AUTO_BENCH_RUN_DIR:-$SCRIPT_DIR}"',
        'mkdir -p "$RUN_DIR"',
        'LOG_FILE="$RUN_DIR/run.log"',
        ': > "$LOG_FILE"',
        'if [ "${AUTO_BENCH_QUIET:-}" = "1" ]; then',
        '  exec > "$LOG_FILE" 2>&1',
        "else",
        '  exec > >(tee -a "$LOG_FILE") 2>&1',
        "fi",
        "",
    ]
    lines.extend(_environment_lines(metadata))
    lines.extend(_gpu_frequency_lines(metadata))

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


def _profile_script(nsys_prefix: list[Any]) -> str:
    nsys_argv = [
        *_variant_argv(nsys_prefix, "$PROFILE_DIR"),
        "env",
        "AUTO_BENCH_RUN_DIR=$PROFILE_DIR",
        "bash",
        "$SCRIPT_DIR/cmd.sh",
    ]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'PROFILE_DIR="$SCRIPT_DIR/profile"',
        'mkdir -p "$PROFILE_DIR"',
        'PROFILE_LOG_FILE="$PROFILE_DIR/profile.log"',
        ': > "$PROFILE_LOG_FILE"',
        'if [ "${AUTO_BENCH_QUIET:-}" = "1" ]; then',
        '  exec > "$PROFILE_LOG_FILE" 2>&1',
        "else",
        '  exec > >(tee -a "$PROFILE_LOG_FILE") 2>&1',
        "fi",
        "",
        _format_command(nsys_argv),
        "",
    ]
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


def _nsys_config(config: Any) -> dict[str, Any] | None:
    if config in (None, False):
        return None
    if config is True:
        config = {}
    if not isinstance(config, dict):
        return None
    if config.get("enabled", True) is False:
        return None
    prefix = [*_nsys_tool_env_prefix(config), *_nsys_prefix(config)]
    if not prefix:
        return None
    return {"prefix": prefix}


def _nsys_tool_env_prefix(config: dict[str, Any]) -> list[str]:
    env = config.get("tool_env")
    if not isinstance(env, dict) or not env:
        return []
    assignments = [f"{key}={value}" for key, value in env.items()]
    return ["env", *assignments]


def _nsys_prefix(config: dict[str, Any]) -> list[Any]:
    command_prefix = config.get("command_prefix")
    if isinstance(command_prefix, str):
        return shlex.split(command_prefix)
    if isinstance(command_prefix, list):
        return command_prefix

    executable = config.get("executable", "nsys")
    prefix: list[Any] = [executable, "profile"]
    prefix.extend(_nsys_env_options(config))
    prefix.extend(_nsys_options(config))
    extra_args = config.get("args")
    if isinstance(extra_args, list):
        prefix.extend(extra_args)
    return prefix


def _nsys_env_options(config: dict[str, Any]) -> list[str]:
    env = config.get("env")
    if not isinstance(env, dict) or not env:
        return []
    assignments = [f"{key}={value}" for key, value in env.items()]
    return ["-e", ",".join(assignments)]


def _nsys_options(config: dict[str, Any]) -> list[str]:
    reserved = {
        "args",
        "command_prefix",
        "enabled",
        "env",
        "executable",
        "options",
        "tool_env",
    }
    options: dict[str, Any] = {}
    if "force_overwrite" not in config:
        options["force_overwrite"] = True
    if "trace" not in config:
        options["trace"] = "cuda,nvtx"
    if "output" not in config:
        options["output"] = "$SCRIPT_DIR/nsys_trace"
    nested = config.get("options")
    if isinstance(nested, dict):
        options.update(nested)
    for name, value in config.items():
        if name not in reserved:
            options[name] = value

    rendered: list[str] = []
    for name, value in options.items():
        if value is None:
            continue
        rendered.extend([_nsys_option_name(name), _nsys_value(value)])
    return rendered


def _nsys_option_name(name: str) -> str:
    return NSYS_PROFILE_SHORT_OPTIONS.get(name, f"--{name.replace('_', '-')}")


NSYS_PROFILE_SHORT_OPTIONS = {
    "backtrace": "-b",
    "capture_range": "-c",
    "delay": "-y",
    "duration": "-d",
    "force_overwrite": "-f",
    "inherit_environment": "-n",
    "nvtx_capture": "-p",
    "output": "-o",
    "sample": "-s",
    "show_output": "-w",
    "start_later": "-Y",
    "stop_on_exit": "-x",
    "trace": "-t",
}


def _nsys_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _run_dir_argv(argv: list[Any]) -> list[str]:
    return _variant_argv(argv, "$RUN_DIR")


def _variant_argv(argv: list[Any], variant_dir: str) -> list[str]:
    return [_variant_path(str(arg), variant_dir) for arg in argv]


def _variant_path(value: str, variant_dir: str) -> str:
    if "," in value and any("=$SCRIPT_DIR/" in item for item in value.split(",")):
        return ",".join(_variant_path(item, variant_dir) for item in value.split(","))
    if "=$SCRIPT_DIR/" in value:
        name, path = value.split("=", 1)
        return f"{name}={_variant_path(path, variant_dir)}"
    if not value.startswith("$SCRIPT_DIR/"):
        return value
    if _is_shared_case_path(value):
        return value
    return f"{variant_dir}/{value.removeprefix('$SCRIPT_DIR/')}"


def _is_shared_case_path(value: str) -> bool:
    return value == "$SCRIPT_DIR/config.yaml" or value.startswith(
        "$SCRIPT_DIR/datasets/"
    )


def _logged_command_block(label: str, argv: list[Any], log_var: str) -> list[str]:
    return [
        label,
        "{",
        _format_command(argv, indent="  "),
        f'}} > >(tee -a "{log_var}") 2>&1',
    ]


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
        env_assignment_list = _env_assignment_list(text)
        env_assignment = _env_assignment_path(text)
        if env_assignment_list is not None:
            parts.append(
                ",".join(
                    _quote_env_assignment(name, value)
                    for name, value in env_assignment_list
                )
            )
        elif env_assignment is not None:
            name, value = env_assignment
            parts.append(_quote_env_assignment(name, value))
        elif _is_script_dir_path(text):
            parts.append(f'"{text}"')
        else:
            parts.append(_sh(text))
    return parts


def _quote_env_assignment(name: str, value: str) -> str:
    return f"{name}={_sh(value)}"


def _sh(value: str) -> str:
    if _is_script_dir_path(value):
        return f'"{value}"'
    return shlex.quote(os.fspath(value))


def _is_script_dir_path(value: str) -> bool:
    shell_dirs = ("$SCRIPT_DIR", "$RUN_DIR", "$PROFILE_DIR")
    return any(
        value == shell_dir or value.startswith(f"{shell_dir}/")
        for shell_dir in shell_dirs
    )


def _env_assignment_path(value: str) -> tuple[str, str] | None:
    if "=" not in value:
        return None
    name, path = value.split("=", 1)
    if not name or not _is_script_dir_path(path):
        return None
    return name, path


def _env_assignment_list(value: str) -> list[tuple[str, str]] | None:
    if "," not in value:
        return None
    assignments: list[tuple[str, str]] = []
    for item in value.split(","):
        if "=" not in item:
            return None
        name, assignment_value = item.split("=", 1)
        if not name:
            return None
        assignments.append((name, assignment_value))
    return assignments
