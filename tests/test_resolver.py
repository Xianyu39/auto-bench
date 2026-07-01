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


def test_resolve_cases_fixed_combinations() -> None:
    result = resolve(
        {
            "metadata": {"name": "profile_cases"},
            "vars": {
                "profile": {
                    "cases": [
                        {"batch_size": 1, "isl": 128},
                        {"batch_size": 4, "isl": 256},
                    ],
                },
            },
            "trtllm-bench": {
                "model": "llama",
                "throughput": {
                    "dataset": "/datasets/static.txt",
                    "max_batch_size": "${vars.profile.batch_size}",
                    "isl": "${vars.profile.isl}",
                },
            },
        }
    )

    cases = result["cases"]
    assert len(cases) == 2
    assert (
        cases[0]["case_id"]
        == "profile_cases__vars.profile.batch_size=1__vars.profile.isl=128"
    )
    assert (
        cases[1]["case_id"]
        == "profile_cases__vars.profile.batch_size=4__vars.profile.isl=256"
    )
    assert cases[0]["vars"]["profile"] == {"batch_size": 1, "isl": 128}
    assert cases[0]["trtllm-bench"]["throughput"]["max_batch_size"] == 1
    assert cases[0]["trtllm-bench"]["throughput"]["isl"] == 128
    assert cases[1]["trtllm-bench"]["throughput"]["max_batch_size"] == 4
    assert cases[1]["trtllm-bench"]["throughput"]["isl"] == 256


def test_cases_combine_with_sweeps_at_group_level() -> None:
    result = resolve(
        {
            "metadata": {"name": "profile_backend"},
            "vars": {
                "profile": {
                    "cases": [
                        {"batch_size": 1, "isl": 128},
                        {"batch_size": 4, "isl": 256},
                    ],
                },
            },
            "trtllm-bench": {
                "model": "llama",
                "throughput": {
                    "backend": {"sweep": ["pytorch", "tensorrt"]},
                    "dataset": "/datasets/static.txt",
                    "max_batch_size": "${vars.profile.batch_size}",
                    "isl": "${vars.profile.isl}",
                },
            },
        }
    )

    assert [case["case_id"] for case in result["cases"]] == [
        "profile_backend__vars.profile.batch_size=1__vars.profile.isl=128"
        "__trtllm-bench.throughput.backend=pytorch",
        "profile_backend__vars.profile.batch_size=1__vars.profile.isl=128"
        "__trtllm-bench.throughput.backend=tensorrt",
        "profile_backend__vars.profile.batch_size=4__vars.profile.isl=256"
        "__trtllm-bench.throughput.backend=pytorch",
        "profile_backend__vars.profile.batch_size=4__vars.profile.isl=256"
        "__trtllm-bench.throughput.backend=tensorrt",
    ]
    assert [
        (
            case["vars"]["profile"]["batch_size"],
            case["trtllm-bench"]["throughput"]["backend"],
        )
        for case in result["cases"]
    ] == [
        (1, "pytorch"),
        (1, "tensorrt"),
        (4, "pytorch"),
        (4, "tensorrt"),
    ]


@pytest.mark.parametrize(
    ("cases", "match"),
    [
        ([], "vars.profile: cases must be non-empty list"),
        ("bad", "vars.profile: cases must be non-empty list"),
        ([1], "vars.profile.0: cases item must be non-empty mapping"),
        ([{}], "vars.profile.0: cases item must be non-empty mapping"),
    ],
)
def test_invalid_cases_errors(cases: object, match: str) -> None:
    with pytest.raises(ProtocolError, match=match):
        resolve(
            {
                "metadata": {"name": "bad_cases"},
                "vars": {
                    "profile": {
                        "cases": cases,
                    },
                },
                "trtllm-bench": {
                    "model": "llama",
                    "throughput": {
                        "dataset": "/datasets/static.txt",
                    },
                },
            }
        )


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
    assert result["warnings"] == [
        "Warning: option 'trtllm-bench.custom_global': This option is not "
        "documented for TensorRT-LLM 1.3.0rc13 or is not supported by "
        "auto-bench yet.",
        "Warning: option 'trtllm-bench.throughput.custom_command': This option "
        "is not documented for TensorRT-LLM 1.3.0rc13 or is not supported by "
        "auto-bench yet.",
    ]
    assert argv[argv.index("--custom_global") + 1] == "root-value"
    assert argv[argv.index("--custom_command") + 1] == "42"
    assert argv.index("--custom_global") < argv.index("throughput")
    assert argv.index("--custom_command") > argv.index("throughput")


def test_unknown_command_is_preserved_with_warning() -> None:
    result = resolve(
        {
            "metadata": {"name": "custom_command"},
            "trtllm-bench": {
                "model": "llama",
                "profile": {
                    "dataset": "/datasets/static.txt",
                    "max_batch_size": 1,
                },
            },
        }
    )

    case = result["cases"][0]
    argv = case["commands"]["benchmark"]["argv"]
    assert result["warnings"] == [
        "Warning: command 'profile': This command is not documented for "
        "TensorRT-LLM 1.3.0rc13 or is not supported by auto-bench yet."
    ]
    assert "profile" in argv
    assert "--model" in argv
    assert "--dataset" in argv


def test_unknown_dataset_generator_args_warn_and_render() -> None:
    result = resolve(
        {
            "metadata": {"name": "custom_dataset"},
            "trtllm-bench": {
                "model": "llama",
                "throughput": {
                    "dataset": {
                        "root": "/datasets",
                        "generator": "my_generator",
                        "custom_arg": 7,
                    },
                },
            },
        }
    )

    case = result["cases"][0]
    prepare = case["commands"]["prepare_dataset"]["argv"]
    assert result["warnings"] == [
        "Warning: dataset generator 'my_generator': This generator is not "
        "documented for TensorRT-LLM 1.3.0rc13 or is not supported by "
        "auto-bench yet."
    ]
    assert "my_generator" in prepare
    assert prepare[prepare.index("--custom-arg") + 1] == "7"


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


def test_top_level_nsys_is_resolved() -> None:
    result = resolve(
        {
            "metadata": {"name": "profile"},
            "nsys": {
                "output": "${runtime.run_dir}/nsys_trace",
            },
            "trtllm-bench": {
                "model": "llama",
                "throughput": {
                    "dataset": "/datasets/static.txt",
                },
            },
        }
    )

    case = result["cases"][0]
    assert case["nsys"] == {
        "output": "$SCRIPT_DIR/nsys_trace",
    }


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
