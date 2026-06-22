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
)

yaml = YAML()
yaml.default_flow_style = False
yaml.width = 4096
yaml.representer.ignore_aliases = lambda *_args: True


@dataclass(frozen=True)
class SweepField:
    path: tuple[str, ...]
    values: Sequence[Any]


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
    raw = copy.deepcopy(dict(data))
    sweeps = _collect_sweeps(raw)
    cases = [_resolve_case(raw, assignment) for assignment in _assignments(sweeps)]
    return {"version": "autobench.resolved/v0.1", "cases": cases}


def dump_yaml(data: Mapping[str, Any]) -> str:
    from io import StringIO

    stream = StringIO()
    yaml.dump(data, stream)
    return stream.getvalue()


def _validate_top_level(data: Mapping[str, Any]) -> None:
    allowed = {"metadata", "vars", "trtllm"}
    actual = set(data)
    unknown = actual - allowed
    if unknown:
        raise ProtocolError(f"root: unknown top-level sections: {sorted(unknown)}")
    missing = {"metadata", "trtllm"} - actual
    if missing:
        raise ProtocolError(f"root: missing top-level sections: {sorted(missing)}")
    if not isinstance(data["metadata"], dict):
        raise ProtocolError("metadata: expected a mapping")
    if not isinstance(data["trtllm"], dict):
        raise ProtocolError("trtllm: expected a mapping")
    if "vars" in data and not isinstance(data["vars"], dict):
        raise ProtocolError("vars: expected a mapping")


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
    walk(data["trtllm"], ("trtllm",))
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
        "trtllm": case_data["trtllm"],
    }
    rendered = render_value(case_data, context, "")
    metadata = rendered["metadata"]
    variables = rendered.get("vars", {})
    trtllm = rendered["trtllm"]
    if (
        not isinstance(metadata, dict)
        or not isinstance(variables, dict)
        or not isinstance(trtllm, dict)
    ):
        raise ProtocolError(
            "resolved case: metadata, vars, and trtllm must be mappings"
        )

    _apply_defaults(trtllm)
    _validate_trtllm(trtllm)
    case_id = _case_id(metadata, assignment)
    prepare_dataset = _resolve_dataset(trtllm)
    write_config = _resolve_config(trtllm)
    benchmark = _benchmark_command(trtllm)
    _assert_no_unresolved(rendered)

    return {
        "case_id": case_id,
        "metadata": metadata,
        "vars": variables,
        "trtllm": trtllm,
        "commands": {
            "prepare_dataset": prepare_dataset,
            "write_config": write_config,
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


def _validate_trtllm(trtllm: Mapping[str, Any]) -> None:
    command_name, command_options = _command_entry(trtllm)
    global_names = {
        name for name, spec in TRTLLM_MANIFEST.items() if spec.location == "global"
    }
    command_names = {
        name for name, spec in TRTLLM_MANIFEST.items() if spec.location != "global"
    }
    unknown_globals = set(trtllm) - global_names - COMMANDS
    if unknown_globals:
        raise ProtocolError(
            f"trtllm: unknown global parameters: {sorted(unknown_globals)}"
        )
    unknown_options = set(command_options) - command_names
    if unknown_options:
        raise ProtocolError(
            "trtllm."
            f"{command_name}: unknown command parameters: {sorted(unknown_options)}"
        )
    for name, spec in TRTLLM_MANIFEST.items():
        if spec.required and spec.location == "global" and name not in trtllm:
            raise ProtocolError(f"trtllm.{name}: missing required parameter")
        if (
            spec.required
            and spec.location != "global"
            and name not in command_options
        ):
            raise ProtocolError(
                f"trtllm.{command_name}.{name}: missing required parameter"
            )


def _command_entry(trtllm: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    command_names = [name for name in trtllm if name in COMMANDS]
    if not command_names:
        raise ProtocolError(
            "trtllm: missing benchmark command section; "
            f"expected one of {sorted(COMMANDS)}"
        )
    if len(command_names) > 1:
        raise ProtocolError(
            "trtllm: exactly one benchmark command section is allowed, "
            f"got {command_names}"
        )
    command_name = command_names[0]
    command_options = trtllm[command_name]
    if not isinstance(command_options, dict):
        raise ProtocolError(f"trtllm.{command_name}: expected a mapping")
    return command_name, command_options


def _command_options(trtllm: Mapping[str, Any]) -> dict[str, Any]:
    command_name, command_options = _command_entry(trtllm)
    if not isinstance(command_options, dict):
        raise ProtocolError(f"trtllm.{command_name}: expected a mapping")
    return command_options


def _resolve_dataset(trtllm: dict[str, Any]) -> dict[str, Any] | None:
    command_name, command_options = _command_entry(trtllm)
    dataset = command_options.get("dataset")
    if isinstance(dataset, str):
        return None
    if not isinstance(dataset, dict):
        raise ProtocolError(
            f"trtllm.{command_name}.dataset: expected path string or managed object"
        )
    root = dataset.get("root")
    generator = dataset.get("generator")
    if not isinstance(root, str) or not isinstance(generator, str):
        raise ProtocolError(
            f"trtllm.{command_name}.dataset: "
            "managed dataset requires root and generator"
        )
    generator_args = DATASET_GENERATORS.get(generator)
    if generator_args is None:
        raise ProtocolError(
            f"trtllm.{command_name}.dataset.generator: unsupported {generator!r}"
        )
    unknown = set(dataset) - {"root", "generator"} - set(generator_args)
    if unknown:
        raise ProtocolError(
            f"trtllm.{command_name}.dataset: unknown generator args: {sorted(unknown)}"
        )

    model = slug(trtllm.get("model", "model"))
    filename = (
        f"{generator}__model={model}"
        f"__in={dataset.get('input_mean')}_{dataset.get('input_stdev')}"
        f"__out={dataset.get('output_mean')}_{dataset.get('output_stdev')}"
        f"__n={dataset.get('num_requests')}.txt"
    )
    output = str(Path(root) / filename)
    command_options["dataset"] = output

    argv = ["trtllm-bench", "--model", str(trtllm["model"]), "prepare-dataset"]
    argv.extend(["--output", output, generator])
    for field, cli_name in generator_args.items():
        if field in dataset:
            argv.extend([f"--{cli_name}", str(dataset[field])])
    return {"if_missing": True, "output": output, "argv": argv}


def _resolve_config(trtllm: dict[str, Any]) -> dict[str, Any] | None:
    command_name, command_options = _command_entry(trtllm)
    config = command_options.get("config")
    if config is None:
        command_options["config"] = None
        return None
    if isinstance(config, str):
        return None
    if not isinstance(config, dict):
        raise ProtocolError(
            f"trtllm.{command_name}.config: "
            "expected path string, null, or managed object"
        )
    content = config.get("content")
    if not isinstance(content, dict):
        raise ProtocolError(
            f"trtllm.{command_name}.config: managed config requires mapping content"
        )
    path = "config.yaml"
    artifact = {"path": path, "content": content}
    command_options["config"] = artifact
    return artifact


def _benchmark_command(trtllm: Mapping[str, Any]) -> dict[str, list[str]]:
    command_name, command_options = _command_entry(trtllm)
    argv = ["trtllm-bench"]

    for name, value in trtllm.items():
        if name in COMMANDS:
            continue
        spec = TRTLLM_MANIFEST[name]
        argv.extend(
            _render_option(spec.cli_name or name, value, spec.value_taking_bool)
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
        spec = TRTLLM_MANIFEST[name]
        argv.extend(
            _render_option(spec.cli_name or name, value, spec.value_taking_bool)
        )
    return {"argv": argv}


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
