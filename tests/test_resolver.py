from __future__ import annotations

import pytest

from auto_bench.errors import ProtocolError
from auto_bench.resolver import resolve


def test_resolve_sweeps_dataset_config_and_commands() -> None:
    result = resolve(
        {
            "metadata": {"name": "decode_sweep"},
            "vars": {
                "batch_size": {"sweep": [1, 4]},
            },
            "trtllm-bench": {
                "model": "meta-llama/Llama-2-7b-hf",
                "model_path": "/engines/llama",
                "throughput": {
                    "isl": {"sweep": [128, 256]},
                    "osl": 64,
                    "streaming": None,
                    "iteration_log": "/logs/iter.json",
                    "dataset": {
                        "root": "/datasets",
                        "generator": "token_norm_dist",
                        "num_requests": 100,
                        "input_mean": "${trtllm_bench.throughput.isl}",
                        "output_mean": "${trtllm_bench.throughput.osl}",
                        "input_stdev": 0,
                        "output_stdev": 0,
                    },
                    "config": {
                        "content": {
                            "cuda_graph_config": {
                                "enable_padding": True,
                                "batch_sizes": [1, "${vars.batch_size}"],
                            }
                        },
                    },
                    "max_batch_size": "${vars.batch_size}",
                    "max_num_tokens": (
                        "${vars.batch_size * trtllm_bench.throughput.osl}"
                    ),
                },
            },
        }
    )

    cases = result["cases"]
    assert len(cases) == 4
    case = cases[1]
    assert (
        case["case_id"]
        == "decode_sweep__vars.batch_size=1__trtllm-bench.throughput.isl=256"
    )
    assert case["trtllm-bench"]["throughput"]["dataset"].endswith(
        "__in=256_0__out=64_0__n=100.txt"
    )
    assert case["vars"]["batch_size"] == 1
    assert case["trtllm-bench"]["throughput"]["max_num_tokens"] == 64
    assert case["commands"]["prepare_dataset"]["if_missing"] is True
    assert case["commands"]["write_config"]["content"]["cuda_graph_config"][
        "batch_sizes"
    ] == [1, 1]
    assert case["commands"]["write_config"]["path"] == "config.yaml"
    assert case["trtllm-bench"]["throughput"]["config"] == "config.yaml"
    assert "--config" in case["commands"]["benchmark"]["argv"]
    assert "--dataset" in case["commands"]["benchmark"]["argv"]
    assert "--batch_size" not in case["commands"]["benchmark"]["argv"]
    argv = case["commands"]["benchmark"]["argv"]
    assert argv.index("--model_path") < argv.index("throughput")
    assert argv.index("--isl") > argv.index("throughput")
    assert argv[argv.index("--iteration_log") + 1] == "/logs/iter.json"
    assert argv[argv.index("--streaming")] == "--streaming"


def test_user_managed_dataset_and_config_path() -> None:
    result = resolve(
        {
            "metadata": {"name": "static"},
            "trtllm-bench": {
                "model": "llama",
                "throughput": {
                    "dataset": "/datasets/static.txt",
                    "config": "/configs/static.yaml",
                    "max_batch_size": 1,
                },
            },
        }
    )

    case = result["cases"][0]
    argv = case["commands"]["benchmark"]["argv"]
    assert case["commands"]["prepare_dataset"] is None
    assert case["commands"]["write_config"] is None
    assert argv[argv.index("--config") + 1] == "/configs/static.yaml"
    assert argv[argv.index("--dataset") + 1] == "/datasets/static.txt"


def test_legacy_trtllm_section_is_normalized() -> None:
    result = resolve(
        {
            "metadata": {"name": "legacy"},
            "trtllm": {
                "model": "llama",
                "throughput": {
                    "dataset": "/datasets/static.txt",
                    "max_batch_size": "${trtllm.throughput.batch_size}",
                    "batch_size": 2,
                },
            },
        }
    )

    case = result["cases"][0]
    assert "trtllm" not in case
    assert case["trtllm-bench"]["throughput"]["max_batch_size"] == 2


def test_unknown_trtllm_parameters_are_preserved_and_rendered() -> None:
    result = resolve(
        {
            "metadata": {"name": "custom"},
            "trtllm-bench": {
                "model": "llama",
                "custom_global": "root-value",
                "throughput": {
                    "dataset": "/datasets/static.txt",
                    "custom_command": 42,
                },
            },
        }
    )

    case = result["cases"][0]
    argv = case["commands"]["benchmark"]["argv"]
    assert case["trtllm-bench"]["custom_global"] == "root-value"
    assert case["trtllm-bench"]["throughput"]["custom_command"] == 42
    assert argv[argv.index("--custom_global") + 1] == "root-value"
    assert argv[argv.index("--custom_command") + 1] == "42"
    assert argv.index("--custom_global") < argv.index("throughput")
    assert argv.index("--custom_command") > argv.index("throughput")


def test_runtime_variables_are_available_to_expressions() -> None:
    result = resolve(
        {
            "metadata": {"name": "runtime_vars"},
            "trtllm-bench": {
                "model": "llama",
                "run_dir_marker": "${runtime.run_dir}",
                "throughput": {
                    "dataset": {
                        "root": "${runtime.dataset_dir}",
                        "generator": "token_norm_dist",
                        "num_requests": 8,
                        "input_mean": 16,
                        "output_mean": 4,
                        "input_stdev": 0,
                        "output_stdev": 0,
                    },
                    "log_path": "${runtime.log_path}",
                    "config_path": "${runtime.config_path}",
                },
            },
        }
    )

    case = result["cases"][0]
    argv = case["commands"]["benchmark"]["argv"]
    assert case["runtime"] == {
        "case_id": "runtime_vars",
        "run_dir": "$SCRIPT_DIR",
        "log_path": "$SCRIPT_DIR/run.log",
        "config_path": "$SCRIPT_DIR/config.yaml",
        "dataset_dir": "$SCRIPT_DIR/datasets",
    }
    assert case["trtllm-bench"]["run_dir_marker"] == "$SCRIPT_DIR"
    assert case["trtllm-bench"]["throughput"]["dataset"].startswith(
        "$SCRIPT_DIR/datasets/token_norm_dist__"
    )
    assert argv[argv.index("--run_dir_marker") + 1] == "$SCRIPT_DIR"
    assert argv[argv.index("--log_path") + 1] == "$SCRIPT_DIR/run.log"
    assert argv[argv.index("--config_path") + 1] == "$SCRIPT_DIR/config.yaml"


def test_missing_reference_errors() -> None:
    with pytest.raises(ProtocolError, match="does not exist"):
        resolve(
            {
                "metadata": {"name": "bad"},
                "trtllm-bench": {
                    "model": "llama",
                    "throughput": {
                        "dataset": "/datasets/${trtllm_bench.missing}.txt",
                    },
                },
            }
        )
