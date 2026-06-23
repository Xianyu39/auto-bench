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


PROTOCOL_FIELDS = {"dataset", "config"}

TRTLLM_BENCH_GLOBAL_PARAMS = {
    "model": ParamSpec("global", "model", required=True),
    "model_path": ParamSpec("global", "model_path"),
    "workspace": ParamSpec("global", "workspace"),
    "log_level": ParamSpec("global", "log_level"),
    "revision": ParamSpec("global", "revision"),
    "telemetry": ParamSpec("global", "telemetry"),
    "no_telemetry": ParamSpec("global", "no-telemetry"),
}

TRTLLM_BENCH_COMMAND_PARAMS = {
    "engine_dir": ParamSpec("command", "engine_dir"),
    "backend": ParamSpec("command", "backend"),
    "custom_module_dirs": ParamSpec("command", "custom_module_dirs"),
    "extra_llm_api_options": ParamSpec("command", "extra_llm_api_options"),
    "sampler_options": ParamSpec("command", "sampler_options"),
    "max_batch_size": ParamSpec("command", "max_batch_size"),
    "max_num_tokens": ParamSpec("command", "max_num_tokens"),
    "max_seq_len": ParamSpec("command", "max_seq_len"),
    "beam_width": ParamSpec("command", "beam_width"),
    "kv_cache_free_gpu_mem_fraction": ParamSpec(
        "command", "kv_cache_free_gpu_mem_fraction"
    ),
    "no_skip_tokenizer_init": ParamSpec("command", "no_skip_tokenizer_init"),
    "custom_tokenizer": ParamSpec("command", "custom_tokenizer"),
    "eos_id": ParamSpec("command", "eos_id"),
    "modality": ParamSpec("command", "modality"),
    "image_data_format": ParamSpec("command", "image_data_format"),
    "data_device": ParamSpec("command", "data_device"),
    "max_input_len": ParamSpec("command", "max_input_len"),
    "num_requests": ParamSpec("command", "num_requests"),
    "warmup": ParamSpec("command", "warmup"),
    "target_input_len": ParamSpec("command", "target_input_len"),
    "target_output_len": ParamSpec("command", "target_output_len"),
    "tp": ParamSpec("command", "tp"),
    "pp": ParamSpec("command", "pp"),
    "ep": ParamSpec("command", "ep"),
    "cluster_size": ParamSpec("command", "cluster_size"),
    "concurrency": ParamSpec("command", "concurrency"),
    "streaming": ParamSpec("command", "streaming"),
    "report_json": ParamSpec("command", "report_json"),
    "iteration_log": ParamSpec("command", "iteration_log"),
    "output_json": ParamSpec("command", "output_json"),
    "request_json": ParamSpec("command", "request_json"),
    "enable_chunked_context": ParamSpec("command", "enable_chunked_context"),
    "disable_chunked_context": ParamSpec("command", "disable_chunked_context"),
    "scheduler_policy": ParamSpec("command", "scheduler_policy"),
    "medusa_choices": ParamSpec("command", "medusa_choices"),
    "tp_size": ParamSpec("command", "tp_size"),
    "pp_size": ParamSpec("command", "pp_size"),
    "quantization": ParamSpec("command", "quantization"),
    "no_weights_loading": ParamSpec("command", "no_weights_loading"),
    "trust_remote_code": ParamSpec("command", "trust_remote_code"),
    # Convenience aliases still used in our examples. These are not used as a
    # whitelist; unknown TensorRT-LLM options are preserved and rendered.
    "dp": ParamSpec("command", "dp"),
    "isl": ParamSpec("command", "isl"),
    "osl": ParamSpec("command", "osl"),
    "kv_cache_dtype": ParamSpec("command", "kv_cache_dtype"),
}

TRTLLM_MANIFEST: dict[str, ParamSpec] = {
    **TRTLLM_BENCH_GLOBAL_PARAMS,
    "dataset": ParamSpec("protocol", required=True),
    "config": ParamSpec("protocol", default=None),
    **TRTLLM_BENCH_COMMAND_PARAMS,
}


COMMANDS = {"throughput", "latency", "build"}
DATASET_GENERATORS = {
    "dataset": {
        "input": "input",
        "max_input_length": "max-input-length",
        "max_output_length": "max-output-length",
        "num_samples": "num-samples",
        "format": "format",
    },
    "token_norm_dist": {
        "num_requests": "num-requests",
        "input_mean": "input-mean",
        "input_stdev": "input-stdev",
        "output_mean": "output-mean",
        "output_stdev": "output-stdev",
    },
    "token_unif_dist": {
        "num_requests": "num-requests",
        "input_min": "input-min",
        "input_max": "input-max",
        "output_min": "output-min",
        "output_max": "output-max",
    },
}
