from __future__ import annotations

from pathlib import Path

from auto_bench.collector import (
    collect_results,
    extract_trtllm_bench_metrics,
    render_results,
)
from auto_bench.renderer import render_resolved


def test_extract_trtllm_bench_key_value_and_table_metrics() -> None:
    metrics = extract_trtllm_bench_metrics(
        """
===========================================================
= PERFORMANCE OVERVIEW
===========================================================
Number of requests:              1000
Average Input Length (tokens):   2048.0000
Average Output Length (tokens):  2048.0000
Token Throughput (tokens/sec):   1585.7480
Request Throughput (req/sec):    0.7743
Total Latency (ms):              1291504.1051
| Average latency (ms) | 12.25 |
"""
    )

    assert metrics["performance_overview.request_throughput"] == {
        "value": 0.7743,
        "unit": "req/sec",
    }
    assert metrics["token_throughput"] == {
        "value": 1585.748,
        "unit": "tokens/sec",
    }
    assert metrics["average_latency"] == {"value": 12.25, "unit": "ms"}


def test_extract_trtllm_bench_latency_repeated_breakdown_metrics() -> None:
    metrics = extract_trtllm_bench_metrics(
        """
===========================================================
= LATENCY OVERVIEW
===========================================================
Average time-to-first-token (ms): 147.7502
Average inter-token latency (ms): 30.9274
Acceptance Rate (Speculative):    1.00
===========================================================
= GENERATION LATENCY BREAKDOWN
===========================================================
MIN (ms): 63266.8804
MAX (ms): 63374.7770
AVG (ms): 63308.3201
P90 (ms): 63307.1885
===========================================================
= ACCEPTANCE BREAKDOWN
===========================================================
MIN: 1.00
MAX: 1.00
AVG: 1.00
P90: 1.00
"""
    )

    assert metrics["latency_overview.average_time_to_first_token"] == {
        "value": 147.7502,
        "unit": "ms",
    }
    assert metrics["generation_latency_breakdown.min"] == {
        "value": 63266.8804,
        "unit": "ms",
    }
    assert metrics["acceptance_breakdown.min"] == {"value": 1.0}
    assert "min" not in metrics


def test_collect_results_from_rendered_multi_case_artifacts(tmp_path: Path) -> None:
    payload = _resolved_payload()
    render_resolved(payload, tmp_path)
    (tmp_path / "case_one" / "run.log").write_text(
        "Request throughput (req/sec): 10.5\n", encoding="utf-8"
    )
    (tmp_path / "case_two" / "run.log").write_text(
        "Request throughput (req/sec): 22\n", encoding="utf-8"
    )

    rows = collect_results(tmp_path, "trtllm-bench")

    assert rows[0]["case_id"] == "case_one"
    assert rows[0]["variant"] == "default"
    assert rows[0]["status"] == "ok"
    assert rows[0]["vars.batch_size"] == 1
    assert rows[0]["metrics.request_throughput"] == 10.5
    assert rows[0]["metrics.request_throughput_unit"] == "req/sec"
    assert rows[1]["case_id"] == "case_two"
    assert rows[1]["metrics.request_throughput"] == 22


def test_collect_results_marks_missing_logs(tmp_path: Path) -> None:
    render_resolved(
        {"version": "autobench.resolved/v0.1", "cases": [_case("one", 1)]},
        tmp_path,
    )

    rows = collect_results(tmp_path, "trtllm-bench")

    assert rows[0]["status"] == "missing_log"
    assert "metrics.request_throughput" not in rows[0]


def test_collect_results_reads_nsys_compare_logs(tmp_path: Path) -> None:
    payload = {"version": "autobench.resolved/v0.1", "cases": [_case("one", 1)]}
    payload["cases"][0]["nsys"] = {"compare": True}
    render_resolved(payload, tmp_path)
    (tmp_path / "baseline").mkdir()
    (tmp_path / "nsys").mkdir()
    (tmp_path / "baseline" / "run.log").write_text(
        "Request throughput (req/sec): 10\n", encoding="utf-8"
    )
    (tmp_path / "nsys" / "run.log").write_text(
        "Request throughput (req/sec): 8\n", encoding="utf-8"
    )

    rows = collect_results(tmp_path, "trtllm-bench")

    assert [row["variant"] for row in rows] == ["baseline", "nsys"]
    assert rows[0]["nsys.compare"] is True
    assert rows[0]["metrics.request_throughput"] == 10
    assert rows[1]["metrics.request_throughput"] == 8


def test_render_results_csv() -> None:
    output = render_results(
        [
            {
                "case_id": "case_one",
                "status": "ok",
                "log_path": "/tmp/run.log",
                "metrics.request_throughput": 10,
            }
        ],
        "csv",
    )

    assert "case_id,variant,status,log_path,metrics.request_throughput" in output
    assert "case_one,,ok,/tmp/run.log,10" in output


def _resolved_payload() -> dict:
    return {
        "version": "autobench.resolved/v0.1",
        "cases": [_case("case_one", 1), _case("case_two", 2)],
    }


def _case(case_id: str, batch_size: int) -> dict:
    return {
        "case_id": case_id,
        "metadata": {"name": "collect", "tags": ["test"]},
        "vars": {"batch_size": batch_size},
        "trtllm-bench": {"model": "llama"},
        "commands": {
            "prepare_dataset": None,
            "write_config": None,
            "benchmark": {
                "argv": ["trtllm-bench", "--model", "llama", "throughput"]
            },
        },
    }
