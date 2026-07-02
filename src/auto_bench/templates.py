from __future__ import annotations

from collections.abc import Callable
from typing import Any


def minimal_template() -> dict[str, Any]:
    return {
        "metadata": {
            "name": "minimal_sweep",
            "description": "Minimal autobench experiment.",
            "tags": ["minimal"],
            "gap": 30,
            "env": {
                "CUDA_VISIBLE_DEVICES": 0,
            },
        },
        "vars": {
            "batch_size": {
                "sweep": [1, 4],
            },
        },
        "trtllm-bench": {
            "model": "meta-llama/Llama-2-7b-hf",
            "model_path": "/mnt/engines/llama2-7b",
            "throughput": {
                "isl": 128,
                "osl": 64,
                "max_batch_size": "${vars.batch_size}",
                "max_num_tokens": "${vars.batch_size * trtllm_bench.throughput.osl}",
                "dataset": {
                    "root": "/mnt/datasets/autobench",
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
                        },
                    },
                },
            },
        },
    }


def decode_template() -> dict[str, Any]:
    template = minimal_template()
    template["metadata"].update(
        {
            "name": "decode_sweep",
            "description": "Decode throughput sweep example.",
            "tags": ["decode"],
            "gpu_frequency": {
                "min_mhz": 1410,
                "max_mhz": 1410,
                "gpu_ids": [0],
            },
        }
    )
    template["trtllm-bench"]["throughput"]["isl"] = {"sweep": [128, 256]}
    template["trtllm-bench"]["throughput"]["osl"] = 64
    return template


def prefill_template() -> dict[str, Any]:
    return {
        "metadata": {
            "name": "prefill_sweep",
            "description": "Minimal prefill sweep example.",
            "tags": ["prefill"],
            "gap": 30,
            "gpu_frequency": {
                "min_mhz": 1400,
                "max_mhz": 1400,
            },
            "env": {
                "TRTLLM_LOG_LEVEL": "INFO",
            },
        },
        "vars": {
            "batch_size": {
                "sweep": [1, 2, 4, 8, 16, 32],
            },
        },
        "nsys": {
            "tool_env": {
                "NSYS_STATS_PATH": "${runtime.run_dir}/stats",
            },
            "env": {
                "CUDA_VISIBLE_DEVICES": 0,
                "TLLM_PROFILE_START_STOP": "10-20",
            },
            "output": "${runtime.run_dir}/nsys_trace",
            "force_overwrite": True,
            "trace": ["cuda", "nvtx"],
            "capture_range": "cudaProfilerApi",
            "trace_fork_before_exec": True,
            "cuda-graph-trace": "node",
        },
        "trtllm-bench": {
            "model": "meta-llama/Llama-2-7b-hf",
            "model_path": "/mnt/engines/llama2-7b",
            "throughput": {
                "ep": 4,
                "tp": 4,
                "warmup": 0,
                "backend": "pytorch",
                "max_batch_size": "${vars.batch_size}",
                "max_num_tokens": (
                    "${vars.batch_size * "
                    "trtllm_bench.throughput.dataset.input_mean + 1}"
                ),
                "num_requests": 256,
                "iteration_log": "${runtime.run_dir}/iter.log",
                "dataset": {
                    "root": "/mnt/datasets/autobench",
                    "generator": "token_norm_dist",
                    "num_requests": 256,
                    "input_mean": 1024,
                    "output_mean": 1024,
                    "input_stdev": 0,
                    "output_stdev": 0,
                },
                "config": {
                    "content": {
                        "cuda_graph_config": {
                            "enable_padding": True,
                            "batch_sizes": [1, "${vars.batch_size}"],
                        },
                        "enable_attention_dp": True,
                    },
                },
            },
        },
    }


TEMPLATES: dict[str, Callable[[], dict[str, Any]]] = {
    "minimal": minimal_template,
    "decode": decode_template,
    "prefill": prefill_template,
}


def get_template(name: str) -> dict[str, Any]:
    return TEMPLATES[name]()
