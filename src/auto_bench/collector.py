from __future__ import annotations

import csv
import re
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import Any, Literal

from ruamel.yaml import YAML

from auto_bench.errors import AutobenchError
from auto_bench.resolver import dump_yaml

Framework = Literal["trtllm-bench"]
OutputFormat = Literal["csv", "yaml"]

yaml = YAML()

_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*|\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?")
_KV_RE = re.compile(
    r"^\s*(?:\[[^\]]+\]\s*)*"
    r"(?P<key>[A-Za-z][A-Za-z0-9 _./%()\-]{2,}?)"
    r"\s*:\s*"
    r"(?P<value>[-+]?(?:\d+(?:,\d{3})*|\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"\s*(?P<unit>[A-Za-z][A-Za-z0-9_./%\-]*)?\s*$"
)
_KEY_UNIT_RE = re.compile(r"^(?P<key>.*?)\s*\((?P<unit>[^)]+)\)\s*$")
_SECTION_RE = re.compile(r"^\s*=\s*(?P<section>[A-Z][A-Z0-9 +./%\-]+?)\s*$")


def collect_results(
    artifact_dir: str | Path,
    framework: Framework,
) -> list[dict[str, Any]]:
    if framework != "trtllm-bench":
        raise AutobenchError(f"unsupported framework {framework!r}")

    root = Path(artifact_dir)
    resolved_path = root / "resolved.yaml"
    if not resolved_path.exists():
        raise AutobenchError(f"{resolved_path}: missing resolved.yaml")

    with resolved_path.open("r", encoding="utf-8") as handle:
        resolved = yaml.load(handle)
    if not isinstance(resolved, dict) or not isinstance(resolved.get("cases"), list):
        raise AutobenchError(f"{resolved_path}: expected resolved autobench payload")

    cases = resolved["cases"]
    multi_case = len(cases) != 1
    rows: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise AutobenchError(f"{resolved_path}: case entries must be mappings")
        case_id = str(case.get("case_id", "case"))
        case_dir = root / case_id if multi_case else root
        log_path = case_dir / "run.log"

        row: dict[str, Any] = {
            "case_id": case_id,
            "log_path": str(log_path),
        }
        _add_flattened(row, "metadata", case.get("metadata", {}))
        _add_flattened(row, "vars", case.get("vars", {}))

        if not log_path.exists():
            row["status"] = "missing_log"
            rows.append(row)
            continue

        metrics = extract_trtllm_bench_metrics(log_path.read_text(encoding="utf-8"))
        row["status"] = "ok" if metrics else "no_metrics"
        for name, metric in sorted(metrics.items()):
            row[f"metrics.{name}"] = metric["value"]
            unit = metric.get("unit")
            if unit:
                row[f"metrics.{name}_unit"] = unit
        rows.append(row)
    return rows


def render_results(rows: list[dict[str, Any]], output_format: OutputFormat) -> str:
    if output_format == "yaml":
        return dump_yaml({"version": "autobench.results/v0.1", "results": rows})
    if output_format == "csv":
        return _rows_to_csv(rows)
    raise AutobenchError(f"unsupported output format {output_format!r}")


def extract_trtllm_bench_metrics(text: str) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    section: str | None = None
    for line in text.splitlines():
        section_match = _SECTION_RE.match(line)
        if section_match is not None:
            section, _unit = _normalize_metric_key(section_match.group("section"))
            continue

        parsed = _parse_key_value_metric(line)
        if parsed is None:
            parsed = _parse_pipe_metric(line)
        if parsed is None:
            continue
        key, value, unit = parsed
        name, key_unit = _normalize_metric_key(key)
        if not name:
            continue
        metric_names: list[str] = []
        if section:
            metric_names.append(f"{section}.{name}")
            if name not in {"min", "max", "avg", "p90", "p95", "p99"}:
                metric_names.append(name)
        else:
            metric_names.append(name)
        for metric_name in metric_names:
            metrics[metric_name] = {"value": value}
            if unit or key_unit:
                metrics[metric_name]["unit"] = unit or key_unit
    return metrics


def _parse_key_value_metric(line: str) -> tuple[str, int | float, str | None] | None:
    match = _KV_RE.match(line)
    if match is None:
        return None
    value = _parse_number(match.group("value"))
    if value is None:
        return None
    return match.group("key"), value, match.group("unit")


def _parse_pipe_metric(line: str) -> tuple[str, int | float, str | None] | None:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if len(cells) < 2 or not cells[0] or set(cells[0]) <= {"-", ":"}:
        return None
    value_match = _NUMBER_RE.search(cells[1])
    if value_match is None:
        return None
    value = _parse_number(value_match.group(0))
    if value is None:
        return None
    unit = cells[2] if len(cells) > 2 and cells[2] else None
    return cells[0], value, unit


def _normalize_metric_key(key: str) -> tuple[str, str | None]:
    key = key.strip()
    unit: str | None = None
    unit_match = _KEY_UNIT_RE.match(key)
    if unit_match is not None:
        key = unit_match.group("key")
        unit = unit_match.group("unit")
    key = key.lower().replace("%", " percent ")
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return key, unit


def _parse_number(value: str) -> int | float | None:
    cleaned = value.replace(",", "")
    if cleaned in {"", "+", "-", ".", "+.", "-."}:
        return None
    try:
        parsed = float(cleaned)
    except ValueError:
        return None
    if parsed.is_integer() and "." not in cleaned and "e" not in cleaned.lower():
        return int(parsed)
    return parsed


def _add_flattened(row: dict[str, Any], prefix: str, value: Any) -> None:
    if not isinstance(value, Mapping):
        return
    for key, item in _flatten(value).items():
        row[f"{prefix}.{key}"] = item


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(_flatten(item, path))
        elif isinstance(item, list):
            flattened[path] = ",".join(str(part) for part in item)
        else:
            flattened[path] = item
    return flattened


def _rows_to_csv(rows: list[dict[str, Any]]) -> str:
    base_columns = ["case_id", "status", "log_path"]
    other_columns = sorted(
        key for row in rows for key in row if key not in set(base_columns)
    )
    fieldnames = [*base_columns, *dict.fromkeys(other_columns)]

    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()
