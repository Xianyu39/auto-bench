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
            "trtllm": {
                "model": "meta-llama/Llama-2-7b-hf",
                "model_path": "/engines/llama",
                "throughput": {
                    "isl": {"sweep": [128, 256]},
                    "osl": 64,
                    "iteration_log": None,
                    "dataset": {
                        "root": "/datasets",
                        "generator": "token-norm-dist",
                        "num_requests": 100,
                        "input_mean": "${trtllm.throughput.isl}",
                        "output_mean": "${trtllm.throughput.osl}",
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
                    "max_num_tokens": "${vars.batch_size * trtllm.throughput.osl}",
                },
            },
        }
    )

    cases = result["cases"]
    assert len(cases) == 4
    case = cases[1]
    assert (
        case["case_id"]
        == "decode_sweep__vars.batch_size=1__trtllm.throughput.isl=256"
    )
    assert case["trtllm"]["throughput"]["dataset"].endswith(
        "__in=256_0__out=64_0__n=100.txt"
    )
    assert case["vars"]["batch_size"] == 1
    assert case["trtllm"]["throughput"]["max_num_tokens"] == 64
    assert case["commands"]["prepare_dataset"]["if_missing"] is True
    assert case["commands"]["write_config"]["content"]["cuda_graph_config"][
        "batch_sizes"
    ] == [1, 1]
    assert case["commands"]["write_config"]["path"] == "config.yaml"
    assert "--config" in case["commands"]["benchmark"]["argv"]
    assert "--dataset" in case["commands"]["benchmark"]["argv"]
    assert "--batch_size" not in case["commands"]["benchmark"]["argv"]
    argv = case["commands"]["benchmark"]["argv"]
    assert argv.index("--model_path") < argv.index("throughput")
    assert argv.index("--isl") > argv.index("throughput")
    assert argv[argv.index("--iteration-log")] == "--iteration-log"


def test_user_managed_dataset_and_config_path() -> None:
    result = resolve(
        {
            "metadata": {"name": "static"},
            "trtllm": {
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


def test_unknown_trtllm_parameters_are_preserved_and_rendered() -> None:
    result = resolve(
        {
            "metadata": {"name": "custom"},
            "trtllm": {
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
    assert case["trtllm"]["custom_global"] == "root-value"
    assert case["trtllm"]["throughput"]["custom_command"] == 42
    assert argv[argv.index("--custom_global") + 1] == "root-value"
    assert argv[argv.index("--custom_command") + 1] == "42"
    assert argv.index("--custom_global") < argv.index("throughput")
    assert argv.index("--custom_command") > argv.index("throughput")


def test_missing_reference_errors() -> None:
    with pytest.raises(ProtocolError, match="does not exist"):
        resolve(
            {
                "metadata": {"name": "bad"},
                "trtllm": {
                    "model": "llama",
                    "throughput": {
                        "dataset": "/datasets/${trtllm.missing}.txt",
                    },
                },
            }
        )
