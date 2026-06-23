from __future__ import annotations

import copy
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from auto_bench.errors import ProtocolError
from auto_bench.expressions import render_value, slug
from auto_bench.manifest import (
    COMMANDS,
    DATASET_GENERATORS,
    TRTLLM_MANIFEST,
    ParamSpec,
)

yaml = YAML()
yaml.default_flow_style = False
yaml.width = 4096
yaml.representer.ignore_aliases = lambda *_args: True

RUNTIME_CONTEXT = {
    "run_dir": "$SCRIPT_DIR",
    "log_path": "$SCRIPT_DIR/run.log",
    "config_path": "$SCRIPT_DIR/config.yaml",
    "dataset_dir": "$SCRIPT_DIR/datasets",
}
BENCHMARK_SECTION = "trtllm-bench"
LEGACY_BENCHMARK_SECTION = "trtllm"
BENCHMARK_EXPR_NAME = "trtllm_bench"


@dataclass(frozen=True)
class SweepField:
    path: tuple[str, ...]
    values: Sequence[Any]


@dataclass(frozen=True)
class PathOptionResult:
    option_name: str
    path: str
    operation_name: str
    operation: dict[str, Any]


def load_experiment(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.load(handle)
    if not isinstance(data, dict):
        raise ProtocolError("root: expected a YAML mapping")
    return dict(data)


def resolve_file(path: str | Path) -> dict[str, Any]:
    return resolve(load_experiment(path))


def resolve(data: Mapping[str, Any]) -> dict[str, Any]:
    _validate_top_level(data)
    raw = copy.deepcopy(_normalize_top_level(data))
    sweeps = _collect_sweeps(raw)
    cases = [_resolve_case(raw, assignment) for assignment in _assignments(sweeps)]
    return {"version": "autobench.resolved/v0.1", "cases": cases}


def dump_yaml(data: Mapping[str, Any]) -> str:
    from io import StringIO

    stream = StringIO()
    yaml.dump(data, stream)
    return stream.getvalue()


def _validate_top_level(data: Mapping[str, Any]) -> None:
    allowed = {"metadata", "vars", BENCHMARK_SECTION, LEGACY_BENCHMARK_SECTION}
    actual = set(data)
    unknown = actual - allowed
    if unknown:
        raise ProtocolError(f"root: unknown top-level sections: {sorted(unknown)}")
    if BENCHMARK_SECTION in data and LEGACY_BENCHMARK_SECTION in data:
        raise ProtocolError(
            f"root: use only one of {BENCHMARK_SECTION!r} or "
            f"{LEGACY_BENCHMARK_SECTION!r}"
        )
    missing = {"metadata"} - actual
    if BENCHMARK_SECTION not in data and LEGACY_BENCHMARK_SECTION not in data:
        missing.add(BENCHMARK_SECTION)
    if missing:
        raise ProtocolError(f"root: missing top-level sections: {sorted(missing)}")
    if not isinstance(data["metadata"], dict):
        raise ProtocolError("metadata: expected a mapping")
    benchmark_section = _benchmark_section(data)
    if not isinstance(data[benchmark_section], dict):
        raise ProtocolError(f"{benchmark_section}: expected a mapping")
    if "vars" in data and not isinstance(data["vars"], dict):
        raise ProtocolError("vars: expected a mapping")


def _normalize_top_level(data: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    if LEGACY_BENCHMARK_SECTION in normalized:
        normalized[BENCHMARK_SECTION] = normalized.pop(LEGACY_BENCHMARK_SECTION)
    return normalized


def _benchmark_section(data: Mapping[str, Any]) -> str:
    if BENCHMARK_SECTION in data:
        return BENCHMARK_SECTION
    return LEGACY_BENCHMARK_SECTION


def _collect_sweeps(data: Mapping[str, Any]) -> list[SweepField]:
    fields: list[SweepField] = []

    def walk(value: Any, path: tuple[str, ...]) -> None:
        if _is_sweep(value):
            values = value["sweep"]
            if not isinstance(values, list) or not values:
                raise ProtocolError(
                    f"{_format_path(path)}: sweep must be non-empty list"
                )
            fields.append(SweepField(path, values))
            return
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, (*path, str(key)))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, (*path, str(index)))

    walk(data["metadata"], ("metadata",))
    if "vars" in data:
        walk(data["vars"], ("vars",))
    walk(data[BENCHMARK_SECTION], (BENCHMARK_SECTION,))
    return fields


def _assignments(sweeps: list[SweepField]) -> list[dict[tuple[str, ...], Any]]:
    if not sweeps:
        return [{}]
    products = itertools.product(*(sweep.values for sweep in sweeps))
    return [
        {sweep.path: value for sweep, value in zip(sweeps, values, strict=True)}
        for values in products
    ]


def _resolve_case(
    raw: Mapping[str, Any], assignment: Mapping[tuple[str, ...], Any]
) -> dict[str, Any]:
    case_data = copy.deepcopy(dict(raw))
    for path, value in assignment.items():
        _set_path(case_data, path, value)

    context = {
        "metadata": case_data["metadata"],
        "vars": case_data.get("vars", {}),
        BENCHMARK_EXPR_NAME: case_data[BENCHMARK_SECTION],
        LEGACY_BENCHMARK_SECTION: case_data[BENCHMARK_SECTION],
        "runtime": RUNTIME_CONTEXT,
    }
    rendered = render_value(case_data, context, "")
    metadata = rendered["metadata"]
    variables = rendered.get("vars", {})
    trtllm = rendered[BENCHMARK_SECTION]
    if (
        not isinstance(metadata, dict)
        or not isinstance(variables, dict)
        or not isinstance(trtllm, dict)
    ):
        raise ProtocolError(
            f"resolved case: metadata, vars, and {BENCHMARK_SECTION} must be mappings"
        )

    _apply_defaults(trtllm)
    _command_entry(trtllm)
    case_id = _case_id(metadata, assignment)
    runtime = {**RUNTIME_CONTEXT, "case_id": case_id}
    operations = _resolve_path_options(trtllm)
    benchmark = _benchmark_command(trtllm)
    _assert_no_unresolved(rendered)

    return {
        "case_id": case_id,
        "metadata": metadata,
        "vars": variables,
        "runtime": runtime,
        BENCHMARK_SECTION: trtllm,
        "commands": {
            "prepare_dataset": operations.get("prepare_dataset"),
            "write_config": operations.get("write_config"),
            "benchmark": benchmark,
        },
    }


def _apply_defaults(trtllm: dict[str, Any]) -> None:
    command_options = _command_options(trtllm)
    for name, spec in TRTLLM_MANIFEST.items():
        if (
            spec.location == "global"
            and name not in trtllm
            and spec.default is not None
        ):
            trtllm[name] = copy.deepcopy(spec.default)
        if (
            spec.location != "global"
            and name not in command_options
            and spec.default is not None
        ):
            command_options[name] = copy.deepcopy(spec.default)


def _command_entry(trtllm: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    command_names = [name for name in trtllm if name in COMMANDS]
    if not command_names:
        raise ProtocolError(
            f"{BENCHMARK_SECTION}: missing benchmark command section; "
            f"expected one of {sorted(COMMANDS)}"
        )
    if len(command_names) > 1:
        raise ProtocolError(
            f"{BENCHMARK_SECTION}: exactly one benchmark command section is allowed, "
            f"got {command_names}"
        )
    command_name = command_names[0]
    command_options = trtllm[command_name]
    if not isinstance(command_options, dict):
        raise ProtocolError(f"{BENCHMARK_SECTION}.{command_name}: expected a mapping")
    return command_name, command_options


def _command_options(trtllm: Mapping[str, Any]) -> dict[str, Any]:
    command_name, command_options = _command_entry(trtllm)
    if not isinstance(command_options, dict):
        raise ProtocolError(f"{BENCHMARK_SECTION}.{command_name}: expected a mapping")
    return command_options


def _resolve_path_options(trtllm: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
    command_name, command_options = _command_entry(trtllm)
    operations: dict[str, dict[str, Any] | None] = {
        "prepare_dataset": None,
        "write_config": None,
    }
    for resolver in (_dataset_path_option, _config_path_option):
        result = resolver(trtllm, command_name, command_options)
        if result is None:
            continue
        command_options[result.option_name] = result.path
        operations[result.operation_name] = result.operation
    return operations


def _dataset_path_option(
    trtllm: dict[str, Any],
    command_name: str,
    command_options: dict[str, Any],
) -> PathOptionResult | None:
    dataset = command_options.get("dataset")
    if dataset is None:
        return None
    if isinstance(dataset, str):
        return None
    if not isinstance(dataset, dict):
        raise ProtocolError(
            f"{BENCHMARK_SECTION}.{command_name}.dataset: "
            "expected path string or managed object"
        )
    root = dataset.get("root")
    generator = dataset.get("generator")
    if not isinstance(root, str) or not isinstance(generator, str):
        raise ProtocolError(
            f"{BENCHMARK_SECTION}.{command_name}.dataset: "
            "managed dataset requires root and generator"
        )
    generator_args = DATASET_GENERATORS.get(generator)
    if generator_args is None:
        raise ProtocolError(
            f"{BENCHMARK_SECTION}.{command_name}.dataset.generator: "
            f"unsupported {generator!r}"
        )
    unknown = set(dataset) - {"root", "generator"} - set(generator_args)
    if unknown:
        raise ProtocolError(
            f"{BENCHMARK_SECTION}.{command_name}.dataset: "
            f"unknown generator args: {sorted(unknown)}"
        )

    model = slug(trtllm.get("model", "model"))
    filename = (
        f"{generator}__model={model}"
        f"__in={dataset.get('input_mean')}_{dataset.get('input_stdev')}"
        f"__out={dataset.get('output_mean')}_{dataset.get('output_stdev')}"
        f"__n={dataset.get('num_requests')}.txt"
    )
    output = str(Path(root) / filename)

    argv = ["trtllm-bench"]
    if "model" in trtllm:
        argv.extend(["--model", str(trtllm["model"])])
    argv.append("prepare-dataset")
    argv.extend(["--output", output, generator])
    for field, cli_name in generator_args.items():
        if field in dataset:
            argv.extend([f"--{cli_name}", str(dataset[field])])
    return PathOptionResult(
        option_name="dataset",
        path=output,
        operation_name="prepare_dataset",
        operation={"if_missing": True, "output": output, "argv": argv},
    )


def _config_path_option(
    _trtllm: dict[str, Any],
    command_name: str,
    command_options: dict[str, Any],
) -> PathOptionResult | None:
    config = command_options.get("config")
    if config is None:
        return None
    if isinstance(config, str):
        return None
    if not isinstance(config, dict):
        raise ProtocolError(
            f"{BENCHMARK_SECTION}.{command_name}.config: "
            "expected path string, null, or managed object"
        )
    content = config.get("content")
    if not isinstance(content, dict):
        raise ProtocolError(
            f"{BENCHMARK_SECTION}.{command_name}.config: "
            "managed config requires mapping content"
        )
    path = "config.yaml"
    return PathOptionResult(
        option_name="config",
        path=path,
        operation_name="write_config",
        operation={"path": path, "content": content},
    )


def _benchmark_command(trtllm: Mapping[str, Any]) -> dict[str, list[str]]:
    command_name, command_options = _command_entry(trtllm)
    argv = ["trtllm-bench"]

    for name, value in trtllm.items():
        if name in COMMANDS:
            continue
        spec = TRTLLM_MANIFEST.get(name)
        argv.extend(
            _render_option(_cli_name(name, spec), value, _value_taking_bool(spec))
        )

    argv.append(command_name)

    for name, value in command_options.items():
        if name == "dataset":
            if isinstance(value, str):
                argv.extend(["--dataset", value])
            continue
        if name == "config":
            if isinstance(value, str):
                argv.extend(["--config", value])
            elif isinstance(value, dict):
                argv.extend(["--config", str(value["path"])])
            continue
        spec = TRTLLM_MANIFEST.get(name)
        argv.extend(
            _render_option(_cli_name(name, spec), value, _value_taking_bool(spec))
        )
    return {"argv": argv}


def _cli_name(name: str, spec: ParamSpec | None) -> str:
    if spec is not None and spec.cli_name is not None:
        return spec.cli_name
    return name


def _value_taking_bool(spec: ParamSpec | None) -> bool:
    return bool(spec is not None and spec.value_taking_bool)


def _render_option(name: str, value: Any, value_taking_bool: bool) -> list[str]:
    option = f"--{name}"
    if value is None:
        return [option]
    if isinstance(value, bool) and not value_taking_bool:
        return [option] if value else []
    if isinstance(value, list):
        rendered = [option]
        rendered.extend(str(item) for item in value)
        return rendered
    return [option, str(value)]


def _case_id(
    metadata: Mapping[str, Any], assignment: Mapping[tuple[str, ...], Any]
) -> str:
    base = slug(metadata.get("name", "experiment"))
    if not assignment:
        return base
    parts = [
        f"{'.'.join(path)}={slug(value)}"
        for path, value in assignment.items()
    ]
    return "__".join([base, *parts])


def _assert_no_unresolved(value: Any, path: str = "") -> None:
    if _is_sweep(value):
        raise ProtocolError(f"{path}: unresolved sweep object")
    if isinstance(value, str) and "${" in value:
        raise ProtocolError(f"{path}: unresolved expression")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_no_unresolved(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_unresolved(item, f"{path}[{index}]")


def _is_sweep(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"sweep"}


def _set_path(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current: Any = data
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value


def _format_path(path: tuple[str, ...]) -> str:
    return ".".join(path)
