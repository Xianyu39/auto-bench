from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ParamLocation = Literal["global", "command", "protocol", "config"]


@dataclass(frozen=True)
class ParamSpec:
    location: ParamLocation
    cli_name: str | None = None
    required: bool = False
    default: object | None = None
    value_taking_bool: bool = False


PROTOCOL_FIELDS = {"command", "dataset", "config"}


TRTLLM_MANIFEST: dict[str, ParamSpec] = {
    "model": ParamSpec("global", "model", required=True),
    "command": ParamSpec("protocol", default="throughput"),
    "dataset": ParamSpec("protocol", required=True),
    "config": ParamSpec("protocol", default=None),
    # Common benchmark fields used by the protocol examples. The full fixed
    # TensorRT-LLM manifest can replace or extend this table.
    "model_path": ParamSpec("command", "model_path"),
    "isl": ParamSpec("command", "isl"),
    "osl": ParamSpec("command", "osl"),
    "batch_size": ParamSpec("command", "batch_size"),
    "kv_cache_dtype": ParamSpec("command", "kv_cache_dtype"),
    "max_batch_size": ParamSpec("command", "max_batch_size"),
    "max_num_tokens": ParamSpec("command", "max_num_tokens"),
    "warmup": ParamSpec("command", "warmup"),
    "iterations": ParamSpec("command", "iterations"),
    "streaming": ParamSpec("command", "streaming"),
}


COMMANDS = {"throughput", "latency", "build"}
DATASET_GENERATORS = {
    "token-norm-dist": {
        "num_requests": "num-requests",
        "input_mean": "input-mean",
        "output_mean": "output-mean",
        "input_stdev": "input-stdev",
        "output_stdev": "output-stdev",
    }
}
