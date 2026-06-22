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
        "trtllm": {
            "model": "meta-llama/Llama-2-7b-hf",
            "model_path": "/mnt/engines/llama2-7b",
            "throughput": {
                "isl": 128,
                "osl": 64,
                "max_batch_size": "${vars.batch_size}",
                "max_num_tokens": "${vars.batch_size * trtllm.throughput.osl}",
                "dataset": {
                    "root": "/mnt/datasets/autobench",
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
    template["trtllm"]["throughput"]["isl"] = {"sweep": [128, 256]}
    template["trtllm"]["throughput"]["osl"] = 64
    return template


def prefill_template() -> dict[str, Any]:
    template = minimal_template()
    template["metadata"].update(
        {
            "name": "prefill_sweep",
            "description": "Prefill throughput sweep example.",
            "tags": ["prefill"],
            "gpu_frequency": {
                "min_mhz": 1410,
                "max_mhz": 1410,
                "gpu_ids": [0],
            },
            "env": {
                "CUDA_VISIBLE_DEVICES": 0,
                "TRTLLM_LOG_LEVEL": "INFO",
            },
        }
    )
    template["vars"]["batch_size"] = {"sweep": [1, 2, 4, 8, 16, 32]}
    template["trtllm"]["throughput"].update(
        {
            "isl": 1024,
            "osl": 1,
            "ep": 4,
            "dp": 4,
            "max_num_tokens": "${vars.batch_size * trtllm.throughput.osl + 1}",
            "iteration_log": None,
        }
    )
    template["trtllm"]["throughput"]["dataset"]["num_requests"] = 256
    template["trtllm"]["throughput"]["config"]["content"]["enable_attention_dp"] = True
    return template


TEMPLATES: dict[str, Callable[[], dict[str, Any]]] = {
    "minimal": minimal_template,
    "decode": decode_template,
    "prefill": prefill_template,
}


def get_template(name: str) -> dict[str, Any]:
    return TEMPLATES[name]()
