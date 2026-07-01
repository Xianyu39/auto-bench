# Autobench YAML Protocol v0.1

This document defines the YAML protocol for describing TensorRT-LLM benchmark
experiments in autobench. It specifies the configuration format, resolution
semantics, dataset preparation semantics, and the command-plus-config plan
emitted by the resolver. It does not define a runner, result collector, report
generator, or TensorRT-LLM invocation implementation.

## Design Goals

The protocol has two goals:

1. Let users describe benchmark experiments with concise YAML.
2. Resolve that YAML into one or more executable TensorRT-LLM benchmark
   command-plus-config plans.

The input YAML may contain sweep values and expressions. The resolved output
must not. Every resolved case must contain:

- A fully materialized `trtllm-bench` config with no missing TensorRT-LLM benchmark
  parameters from the input after sweep and expression expansion.
- A dataset path that exists before the benchmark command runs, when the
  selected benchmark command is configured with a dataset.
- A TensorRT-LLM `config.yaml` artifact when deep runtime or LLM API options
  are configured.
- A benchmark command expressed as an argv list.
- Runtime helper paths such as `runtime.run_dir` and `runtime.log_path`.

If a managed dataset is configured and the dataset file is missing, autobench
must generate it before running the benchmark command.

## Top-Level Structure

An autobench YAML file has two required top-level sections and two optional
top-level sections:

```yaml
metadata:
  name: llama2_7b_decode
  description: Decode benchmark on H100.
  tags: [decode, h100]
  gap: 30
  gpu_frequency:
    min_mhz: 1410
    max_mhz: 1410
    gpu_ids: [0]
  env:
    CUDA_VISIBLE_DEVICES: 0
    TRTLLM_LOG_LEVEL: INFO

vars:
  batch_size:
    sweep: [1, 2, 4, 8]

nsys:
  env:
    NSYS_STATS_PATH: "${runtime.run_dir}/stats"
    CUDA_VISIBLE_DEVICES: 0
    TLLM_PROFILE_START_STOP: 10-20
  output: "${runtime.run_dir}/nsys_trace"
  force_overwrite: true
  trace: [cuda, nvtx]
  capture_range: cudaProfilerApi
  trace_fork_before_exec: true
  cuda-graph-trace: node

trtllm-bench:
  model: meta-llama/Llama-2-7b-hf
  model_path: /mnt/engines/llama2-7b
  throughput:
    isl: 1024
    osl: 128
    max_batch_size: "${vars.batch_size}"
    max_num_tokens: "${vars.batch_size * trtllm_bench.throughput.osl}"
    dataset:
      root: /data/autobench/datasets
      generator: token_norm_dist
      num_requests: 1000
      input_mean: "${trtllm_bench.throughput.isl}"
      output_mean: "${trtllm_bench.throughput.osl}"
      input_stdev: 0
      output_stdev: 0
    config:
      content:
        cuda_graph_config:
          enable_padding: true
          batch_sizes: [1, 2, 4, 8]
        kv_cache_config:
          free_gpu_memory_fraction: 0.9
```

### `metadata`

`metadata` describes the experiment. It is not mapped to TensorRT-LLM benchmark
command-line arguments.

Recommended fields:

- `name`: stable experiment name.
- `description`: human-readable description.
- `tags`: list of labels for searching and grouping experiments.
- `gap`: seconds to wait between rendered case scripts in `run_all.sh`.
- `gpu_frequency`: optional GPU graphics clock lock rendered into each
  `cmd.sh` before the benchmark command.
- `env`: environment variables exported by each rendered `cmd.sh`.

Additional metadata fields are allowed. They may be referenced by expressions
with `metadata.<path>`.

`metadata.gap` is a number. When multiple cases are rendered, `run_all.sh`
sleeps for this duration after each case except the last one. This gives
drivers, processes, and GPU state time to settle between experiments.

`metadata.gpu_frequency` may be a number or a mapping. A number locks all GPUs
to that fixed graphics clock in MHz. A mapping supports:

- `enabled`: optional boolean, defaults to `true`.
- `mhz`: shorthand fixed clock. Used when `min_mhz` is not set.
- `min_mhz`: minimum graphics clock in MHz.
- `max_mhz`: maximum graphics clock in MHz. Defaults to `min_mhz`.
- `gpu_ids`: optional list of GPU ids. When omitted, `nvidia-smi -lgc` applies
  to the default GPU selection.

Example rendered lock command:

```bash
nvidia-smi -i 0 -lgc 1410,1410
```

These metadata fields affect rendered shell scripts only. They are not
TensorRT-LLM benchmark parameters.

By default, multi-case `run_all.sh` stops on the first failing case. The
`auto-bench render --continue-on-error` CLI option changes only the rendered
controller script: it logs failed cases, continues to later cases, and exits
non-zero if any case failed.

`metadata.env` is a mapping from environment variable names to scalar values.
The render step writes them before GPU frequency locking and before benchmark
commands:

```bash
export CUDA_VISIBLE_DEVICES=0
export TRTLLM_LOG_LEVEL=INFO
```

### `nsys`

`nsys` is an optional top-level section for Nsight Systems profiling. It may be
`true`, `false`, `null`, or a mapping. When enabled, the render step prefixes
the benchmark command with an Nsight Systems command. The default prefix is:

```bash
nsys profile -f true -t cuda,nvtx -o "$SCRIPT_DIR/nsys_trace"
```

The mapping supports:

- `enabled`: optional boolean, defaults to `true`.
- `env`: optional mapping of environment variables injected into the nsys
  command as `-e KEY=VALUE` options.
- `executable`: optional command name, defaults to `nsys`.
- `output`: optional trace output path, defaults to `$SCRIPT_DIR/nsys_trace`.
- `trace`: optional value for `--trace`, defaults to `cuda,nvtx`. A list is
  rendered as a comma-separated value.
- `force_overwrite`: optional value for `--force-overwrite`, defaults to
  `true`. Set it to `null` to omit that option.
- `options`: optional mapping of additional nsys options. Keys are rendered as
  CLI option names with underscores converted to hyphens.
- Any non-reserved field under `nsys` is also rendered as an nsys option. For
  example, `capture_range: cudaProfilerApi` renders as
  `-c cudaProfilerApi`.
- Known nsys profile short flags are rendered according to the official CLI:
  `backtrace -> -b`, `capture_range -> -c`, `delay -> -y`, `duration -> -d`,
  `env -> -e`, `force_overwrite -> -f`, `inherit_environment -> -n`,
  `nvtx_capture -> -p`, `output -> -o`, `sample -> -s`,
  `show_output -> -w`, `start_later -> -Y`, `stop_on_exit -> -x`, and
  `trace -> -t`. Other option names render as long flags with underscores
  converted to hyphens, such as `trace_fork_before_exec ->
  --trace-fork-before-exec`.
- `args`: optional list of raw extra nsys arguments inserted before `-o`.
- `command_prefix`: optional full command prefix as a string or list. When set,
  it replaces the generated prefix.

Example:

```yaml
nsys:
  env:
    NSYS_STATS_PATH: "${runtime.run_dir}/stats"
    CUDA_VISIBLE_DEVICES: 0
    TLLM_PROFILE_START_STOP: 10-20
  output: "${runtime.run_dir}/nsys_trace"
  force_overwrite: true
  capture_range: cudaProfilerApi
  trace: [cuda, nvtx]
  trace_fork_before_exec: true
  cuda-graph-trace: node
```

When `nsys` is enabled, rendering creates both `cmd.sh` and `profile.sh`.
`cmd.sh` always runs the ordinary benchmark. `profile.sh` wraps `cmd.sh` with
the nsys prefix and sets `AUTO_BENCH_RUN_DIR=$PROFILE_DIR`, so benchmark output
paths derived from `runtime.run_dir` are written under `profile/`. Shared
inputs such as `$SCRIPT_DIR/config.yaml` and `$SCRIPT_DIR/datasets/...` remain
case-level paths. For multi-case renders, `profile_all.sh` runs only
`profile.sh` scripts. Result collection emits separate rows with
`variant: default` and `variant: profile`; rows whose logs were not produced are
marked `missing_log`.

### `vars`

`vars` describes experiment variables that can be swept and referenced by
expressions, but are not TensorRT-LLM benchmark parameters and are never mapped
directly to command-line arguments.

Use `vars` for values such as conceptual batch size, derived dimensions, naming
tokens, or any other experiment control value that TensorRT-LLM does not accept
as a direct parameter.

Fields under `vars` may be referenced with `vars.<path>`.

### `trtllm-bench`

`trtllm-bench` describes TensorRT-LLM benchmark parameters. It is the only section
that maps to benchmark parameters and benchmark commands.

Every non-command key under the `trtllm-bench` root is treated as a root-level
`trtllm-bench` option. Autobench does not require this parameter to appear in
its internal manifest.

The `trtllm-bench` root contains options that belong to `trtllm-bench` itself, such
as `model` and `model_path`. Exactly one supported benchmark subcommand must
also appear as a mapping key, for example `throughput`, `latency`, or `build`.
Fields under that subcommand are rendered as subcommand options.

`dataset` and `config` are special protocol fields under the selected
subcommand. In input YAML, `<command>.dataset` may be either a dataset path or a
managed dataset specification. In resolved output, it must always be a dataset
path. `<command>.config` describes the TensorRT-LLM YAML config artifact passed
with `--config`; in resolved output, it must be either `null` or a config path.
The file-write operation, when needed, is recorded under
`commands.write_config`.

### `runtime`

`runtime` is a built-in read-only expression namespace. It is not a YAML
top-level section and is not rendered as command-line arguments.

Available runtime values:

- `runtime.run_dir`: case run directory at shell execution time, rendered as
  `$SCRIPT_DIR`.
- `runtime.log_path`: case log file path, rendered as `$SCRIPT_DIR/run.log`.
- `runtime.config_path`: case-local config path, rendered as
  `$SCRIPT_DIR/config.yaml`.
- `runtime.dataset_dir`: case-local dataset directory, rendered as
  `$SCRIPT_DIR/datasets`.
- `runtime.case_id`: resolved case identifier. This value is recorded in
  resolved output; it is not available while evaluating input expressions.

Runtime path values intentionally use shell variables so rendered `cmd.sh`
files remain relocatable.

## Parameter Value Types

Fields in `metadata`, `vars`, and `trtllm-bench` may use the following value types.

### Scalar

A scalar is copied into each resolved case as-is.

```yaml
trtllm-bench:
  isl: 1024
  osl: 128
  streaming: false
```

An explicit `null` value under `trtllm-bench` represents a no-value CLI flag. For
example:

```yaml
trtllm-bench:
  streaming: null
```

renders as:

```bash
--streaming
```

Omit the field entirely when the flag should not be rendered.

### Sweep

A sweep field expands the experiment into multiple cases.

```yaml
vars:
  batch_size:
    sweep: [1, 2, 4, 8]
```

After expansion, `vars.batch_size` is a scalar in each resolved case.

A mapping is treated as a sweep object only when it has exactly one key,
`sweep`. The value of `sweep` must be a non-empty list.

### Expression

An expression is a string whose entire value is a single `${...}` block.

```yaml
trtllm-bench:
  throughput:
    max_num_tokens: "${vars.batch_size * trtllm_bench.throughput.osl}"
```

When the whole YAML value is an expression, the resolved value keeps the
expression result type. For example, the result of the expression above is an
integer, not a string.

### String Interpolation

A string interpolation is a string that contains one or more `${...}` blocks but
is not itself a single expression.

```yaml
trtllm-bench:
  throughput:
    dataset: "/data/llama2/i${trtllm_bench.throughput.isl}_o${trtllm_bench.throughput.osl}.txt"
```

Interpolated values are always resolved to strings.

### Managed Dataset

`<command>.dataset` may describe a dataset that autobench manages. The
resolver turns this object into a deterministic dataset path, and the runner
generates the file on demand.

```yaml
trtllm-bench:
  model: meta-llama/Llama-2-7b-hf
  throughput:
    isl:
      sweep: [128, 512]
    osl: 128
    dataset:
      root: /data/autobench/datasets
      generator: token_norm_dist
      num_requests: 1000
      input_mean: "${trtllm_bench.throughput.isl}"
      output_mean: "${trtllm_bench.throughput.osl}"
      input_stdev: 0
      output_stdev: 0
```

Required managed dataset fields:

- `root`: directory where autobench stores generated datasets.
- `generator`: TensorRT-LLM dataset generator name. v0.1 supports
  `dataset`, `token_norm_dist`, and `token_unif_dist`.

Generator arguments are written as fields in the dataset object. For
`token_norm_dist`, common arguments are:

- `num_requests`
- `input_mean`
- `output_mean`
- `input_stdev`
- `output_stdev`

The resolver must generate a readable, deterministic filename from the resolved
dataset fields. The recommended filename format is:

```text
<generator>__model=<slug(trtllm_bench.model)>__in=<input_mean>_<input_stdev>__out=<output_mean>_<output_stdev>__n=<num_requests>.txt
```

Example:

```text
/data/autobench/datasets/token_norm_dist__model=meta-llama_Llama-2-7b-hf__in=128_0__out=128_0__n=1000.txt
```

The filename must include every generator argument that can change dataset
content. This prevents two different dataset specs from resolving to the same
path.

When the runner executes a case:

1. If the resolved dataset path exists, use it.
2. If the path does not exist, run the resolved prepare-dataset command.
3. After preparation, verify that the dataset path exists.
4. Run the benchmark command using that dataset path.

Managed dataset preparation must not be treated as part of the measured
benchmark interval.

### Managed Config

`<command>.config` describes a per-case TensorRT-LLM `config.yaml` file. This
is used for options that TensorRT-LLM expects through `--config` rather than as
direct benchmark CLI flags.

```yaml
trtllm-bench:
  throughput:
    config:
      content:
        cuda_graph_config:
          enable_padding: true
          batch_sizes:
            - 1
            - "${vars.batch_size}"
        print_iter_log: true
```

Required managed config fields:

- `content`: YAML mapping to write into the generated config file.

The resolver must evaluate expressions and sweep values inside `content`. The
resolved content must be valid YAML data and must not contain `sweep` objects or
unresolved `${...}` expressions.

Managed config files are case-local artifacts. The resolver records their path
as `config.yaml`, and the render step writes that file next to the case's
`cmd.sh`. This keeps configs small, local, and easy to inspect.

If `<command>.config` is omitted or `null`, no config artifact is generated and
the benchmark command does not include `--config`.

If `<command>.config` is a string path, autobench treats it as a user-managed
config file. The benchmark command includes `--config <path>`, but autobench
does not generate the file.

If `<command>.config` is a managed config object, the resolved
`<command>.config` value is the generated config path `config.yaml`. The
generated config content and file-write plan are recorded under
`commands.write_config`.

## Path References And Expressions

Expressions may reference values with dotted paths:

- `metadata.<path>`
- `vars.<path>`
- `trtllm_bench.<path>`
- `runtime.<path>`

Examples:

```yaml
metadata:
  model_family: llama2

vars:
  batch_size:
    sweep: [1, 2, 4]

trtllm-bench:
  artifact_dir: "${runtime.run_dir}/artifacts"
  throughput:
    isl: 1024
    osl: 128
    dataset:
      root: "${runtime.dataset_dir}"
      generator: token_norm_dist
      num_requests: 1000
      input_mean: "${trtllm_bench.throughput.isl}"
      output_mean: "${trtllm_bench.throughput.osl}"
      input_stdev: 0
      output_stdev: 0
    log_path: "${runtime.log_path}"
    max_batch_size: "${vars.batch_size}"
    max_num_tokens: "${vars.batch_size * trtllm_bench.throughput.osl}"
```

During expression evaluation, all sweep fields in the current case have already
been replaced by their current scalar values.

### Supported Expression Subset

v0.1 supports a safe expression subset:

- Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- Comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Boolean logic: `and`, `or`, `not`
- Parentheses
- String and numeric literals
- Path references under `metadata`, `vars`, `trtllm-bench`, and `runtime`
- Functions: `min`, `max`, `int`, `str`, `ceil`, `floor`, `slug`

The expression evaluator must not execute arbitrary Python or shell code. File
access, imports, attribute access outside `metadata`, `vars`, `trtllm-bench`, and
`runtime`, and calls to unlisted functions are forbidden.

`slug(value)` converts a value into a path-safe string. It is intended for
dataset filenames and case identifiers.

## Sweep Expansion Rules

All `sweep` fields are expanded by Cartesian product.

```yaml
vars:
  batch_size:
    sweep: [1, 2, 4]

trtllm-bench:
  isl:
    sweep: [128, 512]
```

This produces `2 * 3 = 6` cases.

Expansion order must be deterministic:

1. Traverse `metadata`, `vars`, and `trtllm-bench` in YAML order.
2. Collect sweep fields in that order.
3. Expand Cartesian products in collected order.

The resolved `case_id` should be stable for the same input. The recommended
format is:

```text
<metadata.name>__<path>=<value>__<path>=<value>
```

Example:

```text
llama2_7b_decode__vars.batch_size=4__trtllm.throughput.isl=128
```

If `metadata.name` is missing, the resolver must use a deterministic fallback
such as `experiment`.

## Resolution Order

A resolver must process the input in this order:

1. Parse YAML.
2. Validate that the only top-level keys are `metadata`, `vars`, and `trtllm-bench`.
3. Load the fixed TensorRT-LLM benchmark parameter manifest.
4. Collect all sweep fields from `metadata`, `vars`, and `trtllm-bench`.
5. Expand the Cartesian product of sweep values.
6. For each expanded case, replace sweep objects with the current scalar values.
7. Evaluate expressions and string interpolations.
8. Resolve managed `<command>.dataset` objects into dataset paths and
   prepare-dataset commands.
9. Resolve managed `<command>.config` objects into config paths, config content,
   and file-write plans.
10. Apply manifest defaults for known optional `trtllm-bench` parameters.
11. Render benchmark commands, including parameters not present in the manifest.
12. Emit resolved cases.

Resolved cases must not contain `sweep` objects, managed dataset objects,
managed config objects, or unresolved `${...}` expressions.

## Resolved Case Output

The resolver output is a list of cases. Each case contains the materialized
configuration and executable commands.

```yaml
version: autobench.resolved/v0.1
cases:
  - case_id: llama2_7b_decode__vars.batch_size=1
    metadata:
      name: llama2_7b_decode
      tags: [decode, h100]
    vars:
      batch_size: 1
    runtime:
      case_id: llama2_7b_decode__vars.batch_size=1
      run_dir: $SCRIPT_DIR
      log_path: $SCRIPT_DIR/run.log
      config_path: $SCRIPT_DIR/config.yaml
      dataset_dir: $SCRIPT_DIR/datasets
    trtllm-bench:
      model: meta-llama/Llama-2-7b-hf
      throughput:
        isl: 1024
        osl: 128
        max_batch_size: 1
        max_num_tokens: 128
        dataset: /data/llama2/i1024_o128.txt
        config: config.yaml
        warmup: 5
        iterations: 30
    commands:
      prepare_dataset: null
      write_config:
        path: config.yaml
        content:
          cuda_graph_config:
            enable_padding: true
            batch_sizes: [1]
      benchmark:
        argv:
          - trtllm-bench
          - --model
          - meta-llama/Llama-2-7b-hf
          - throughput
          - --dataset
          - /data/llama2/i1024_o128.txt
          - --config
          - config.yaml
          - --max_batch_size
          - "1"
          - --max_num_tokens
          - "128"
```

Each resolved `trtllm-bench` object must contain one subcommand mapping. Known
parameters with manifest defaults must be materialized in the resolved output.
Parameters that are not present in the manifest are preserved as-is and rendered
as CLI options.

If `<command>.dataset` is managed, `commands.prepare_dataset` must contain an
argv list. If `<command>.dataset` is already a path, `commands.prepare_dataset`
must be `null`.

If `<command>.config` is managed, `commands.write_config` must contain the config
path and content to write. If `<command>.config` is omitted, `commands.write_config`
must be `null`. If `<command>.config` is a user-managed path,
`commands.write_config` must be `null`.

The benchmark command and optional config artifact are the canonical executable
artifacts of a resolved case. `trtllm-bench` remains in the output for traceability
and validation.

## Command Rendering

Autobench YAML is not assumed to be a native TensorRT-LLM config file. The
resolved case must render TensorRT-LLM execution as command argv lists plus an
optional TensorRT-LLM `config.yaml` artifact.

### TensorRT-LLM YAML Support

TensorRT-LLM documents YAML files for selected runtime and LLM API options, such
as `--config` or the older `--extra_llm_api_options` option. These YAML files do
not replace the full `trtllm-bench` CLI command. The model, subcommand,
dataset, and most benchmark controls are still expressed as CLI options.

Therefore, autobench v0.1 treats its YAML as an orchestration protocol and emits
the pair TensorRT-LLM expects in practice:

- `trtllm-bench ... --config config.yaml` command argv.
- The generated `config.yaml` content for options that belong in TensorRT-LLM's
  YAML config surface.

### Benchmark Command

The selected benchmark subcommand is represented as a mapping key under
`trtllm-bench`. v0.1 supports:

- `throughput`
- `latency`
- `build`

Exactly one subcommand mapping is required. `trtllm-bench.command` is not supported
in v0.1.

The rendered benchmark command format is:

```text
trtllm-bench [global options] <command> [command options]
```

Root-level `trtllm-bench` fields map to global options rendered before the
subcommand. Fields under the selected subcommand map to options rendered after
the subcommand. For example, `trtllm-bench.model` maps to `--model`,
`trtllm-bench.model_path` maps to `--model_path`, and
`trtllm-bench.throughput.dataset` maps to `--dataset`.

Option names are rendered from manifest metadata. Protocol fields such as
`dataset` and `config` are consumed by autobench and rendered as the appropriate
command options after they are resolved.

Boolean options are rendered only when true, unless the TensorRT-LLM manifest
defines an explicit value-taking boolean flag.

### Config Artifact

The generated config artifact is a YAML file whose content is exactly the
resolved `commands.write_config.content` mapping. The corresponding
`<command>.config` option resolves to the artifact path.

Example:

```yaml
commands:
  write_config:
    path: config.yaml
    content:
      cuda_graph_config:
        enable_padding: true
        batch_sizes:
          - 1
          - 4
      print_iter_log: true
```

The runner must write this content before executing the benchmark command.
Config writing must be deterministic: the same resolved case must produce the
same `config.yaml` content.

### Prepare Dataset Command

For managed datasets, the rendered prepare command format is:

```text
trtllm-bench --model <model> prepare-dataset --output <dataset_path> <generator> [generator options]
```

Example:

```yaml
commands:
  prepare_dataset:
    if_missing: true
    output: /data/autobench/datasets/token_norm_dist__model=meta-llama_Llama-2-7b-hf__in=128_0__out=128_0__n=1000.txt
    argv:
      - trtllm-bench
      - --model
      - meta-llama/Llama-2-7b-hf
      - prepare-dataset
      - --output
      - /data/autobench/datasets/token_norm_dist__model=meta-llama_Llama-2-7b-hf__in=128_0__out=128_0__n=1000.txt
      - token_norm_dist
      - --num-requests
      - "1000"
      - --input-mean
      - "128"
      - --output-mean
      - "128"
      - --input-stdev
      - "0"
      - --output-stdev
      - "0"
```

## Parameter Manifest

The resolver has an internal manifest for parameter rendering hints, but the
manifest is intentionally incomplete. It must not be used as a whitelist for
TensorRT-LLM benchmark parameters.

The manifest may define:

- CLI spelling for known parameters whose YAML key differs from the CLI flag.
- Whether a known boolean option takes an explicit value.
- Default values for known optional parameters.

Validation and rendering rules:

- Missing benchmark parameters are not errors at resolver time.
- Unknown `trtllm-bench` root parameters and subcommand parameters are preserved.
- Unknown `trtllm-bench` root parameters and subcommand parameters emit English
  warnings in the resolved output and CLI stderr.
- Unknown command sections are preserved and rendered as the benchmark
  subcommand, with an English warning.
- Unknown parameters are rendered as `--<yaml_key>`.
- Known parameters use their manifest CLI spelling when one is defined.
- Input `<command>.dataset` may be a managed dataset object, but resolved
  `<command>.dataset` must be a string path.
- Input `<command>.config` may be absent, `null`, a user-managed path, or a
  managed config object. Resolved `<command>.config` must be `null` or a string
  path.
- Unknown managed dataset generators and generator arguments emit English
  warnings, but are preserved and rendered.
- Managed dataset filenames must be deterministic for the resolved generator
  fields.
- Managed config content must be a YAML mapping.
- Managed config objects resolve to the case-local path `config.yaml`.

`metadata` is not validated against the TensorRT-LLM manifest.

## Error Handling Rules

The resolver must fail the whole YAML file on any of the following errors:

- Unknown top-level section.
- Missing `metadata` or `trtllm-bench` section.
- Empty or non-list `sweep`.
- Invalid managed dataset object.
- Unknown managed dataset generator.
- Unknown managed dataset generator argument.
- Invalid managed config object.
- Managed config content that is not a YAML mapping.
- Expression syntax error.
- Reference to a nonexistent path.
- Reference to an unresolved sweep object.
- Use of a forbidden expression operator, function, import, or attribute.
- Type mismatch after expression evaluation.
- Remaining unresolved `${...}` expression in resolved output.
- Missing dataset file after prepare-dataset command execution.
- Missing config file after managed config write.

Error messages should include the YAML path that caused the error.

## Complete Example

Input YAML:

```yaml
metadata:
  name: llama2_7b_decode
  description: Decode throughput sweep on H100.
  tags: [decode, h100]
  model_family: llama2
  gap: 30
  gpu_frequency:
    min_mhz: 1410
    max_mhz: 1410
    gpu_ids: [0]
  env:
    CUDA_VISIBLE_DEVICES: 0
    TRTLLM_LOG_LEVEL: INFO

vars:
  batch_size:
    sweep: [1, 4]

trtllm-bench:
  model: meta-llama/Llama-2-7b-hf
  model_path: /mnt/engines/llama2-7b
  throughput:
    isl:
      sweep: [128, 512]

    osl: 128

    kv_cache_dtype:
      sweep: [fp16, fp8]

    dataset:
      root: /mnt/datasets/autobench
      generator: token_norm_dist
      num_requests: 1000
      input_mean: "${trtllm_bench.throughput.isl}"
      output_mean: "${trtllm_bench.throughput.osl}"
      input_stdev: 0
      output_stdev: 0

    config:
      content:
        cuda_graph_config:
          enable_padding: true
          batch_sizes: [1, 2, 4]
        kv_cache_config:
          free_gpu_memory_fraction: 0.9
        print_iter_log: true

    max_batch_size: "${vars.batch_size}"
    max_num_tokens: "${vars.batch_size * trtllm_bench.throughput.osl}"

    warmup: 5
    iterations: 30
    streaming: false
```

This input produces `2 * 2 * 2 = 8` resolved cases before manifest defaulting.
One resolved case:

```yaml
version: autobench.resolved/v0.1
cases:
  - case_id: llama2_7b_decode__vars.batch_size=4__trtllm.throughput.isl=128__trtllm.throughput.kv_cache_dtype=fp8
    metadata:
      name: llama2_7b_decode
      description: Decode throughput sweep on H100.
      tags: [decode, h100]
      model_family: llama2
      gap: 30
      gpu_frequency:
        min_mhz: 1410
        max_mhz: 1410
        gpu_ids: [0]
      env:
        CUDA_VISIBLE_DEVICES: 0
        TRTLLM_LOG_LEVEL: INFO
    vars:
      batch_size: 4
    runtime:
      case_id: llama2_7b_decode__vars.batch_size=4__trtllm.throughput.isl=128__trtllm.throughput.kv_cache_dtype=fp8
      run_dir: $SCRIPT_DIR
      log_path: $SCRIPT_DIR/run.log
      config_path: $SCRIPT_DIR/config.yaml
      dataset_dir: $SCRIPT_DIR/datasets
    trtllm-bench:
      model: meta-llama/Llama-2-7b-hf
      model_path: /mnt/engines/llama2-7b
      throughput:
        isl: 128
        osl: 128
        kv_cache_dtype: fp8
        dataset: /mnt/datasets/autobench/token_norm_dist__model=meta-llama_Llama-2-7b-hf__in=128_0__out=128_0__n=1000.txt
        config: config.yaml
        max_batch_size: 4
        max_num_tokens: 512
        warmup: 5
        iterations: 30
        streaming: false
        # Additional manifest parameters are filled here.
    commands:
      prepare_dataset:
        if_missing: true
        output: /mnt/datasets/autobench/token_norm_dist__model=meta-llama_Llama-2-7b-hf__in=128_0__out=128_0__n=1000.txt
        argv:
          - trtllm-bench
          - --model
          - meta-llama/Llama-2-7b-hf
          - prepare-dataset
          - --output
          - /mnt/datasets/autobench/token_norm_dist__model=meta-llama_Llama-2-7b-hf__in=128_0__out=128_0__n=1000.txt
          - token_norm_dist
          - --num-requests
          - "1000"
          - --input-mean
          - "128"
          - --output-mean
          - "128"
          - --input-stdev
          - "0"
          - --output-stdev
          - "0"
      write_config:
        path: config.yaml
        content:
          cuda_graph_config:
            enable_padding: true
            batch_sizes: [1, 2, 4]
          kv_cache_config:
            free_gpu_memory_fraction: 0.9
          print_iter_log: true
      benchmark:
        argv:
          - trtllm-bench
          - --model
          - meta-llama/Llama-2-7b-hf
          - --model_path
          - /mnt/engines/llama2-7b
          - throughput
          - --isl
          - "128"
          - --osl
          - "128"
          - --kv_cache_dtype
          - fp8
          - --dataset
          - /mnt/datasets/autobench/token_norm_dist__model=meta-llama_Llama-2-7b-hf__in=128_0__out=128_0__n=1000.txt
          - --config
          - config.yaml
          - --max_batch_size
          - "4"
          - --max_num_tokens
          - "512"
```

## Parser Test Scenarios

Future parser tests should cover:

- A single static configuration resolves into one complete case.
- Two sweep fields expand into a Cartesian product.
- Sweep fields under `vars` can be referenced by `trtllm-bench` expressions.
- `vars` fields are preserved in resolved output but not rendered as CLI
  options.
- Expressions can read current sweep values and preserve result types.
- String interpolation can generate dataset paths.
- Managed datasets resolve to deterministic dataset paths.
- Managed datasets render prepare-dataset commands.
- Existing dataset paths produce `prepare_dataset: null`.
- Managed config objects resolve to deterministic config paths and content.
- Managed config objects render `write_config` plans.
- User-managed config paths render `--config <path>` without a write plan.
- Benchmark commands render as argv lists.
- Unknown `trtllm-bench` parameters are preserved and rendered.
- Missing benchmark parameters do not fail resolver validation.
- Expressions referencing nonexistent paths fail validation.
- Resolved output contains no `sweep` objects, managed dataset objects, managed
  config objects, or unresolved expressions.

## Not Supported In v0.1

The following features are intentionally out of scope for v0.1:

- A separate `computed` section.
- A separate `constraints` section.
- Complex include/exclude filtering.
- Nested matrix or paired sweep semantics.
- Arbitrary Python expressions.
- Runtime TensorRT-LLM version switching.
- Passing autobench YAML directly to TensorRT-LLM as a full replacement for CLI
  options.
- Result collection, report generation, or scheduling.

## Assumptions

- TensorRT-LLM benchmark version is fixed for v0.1.
- The fixed version's parameter manifest will be defined separately.
- `metadata` is descriptive and does not directly generate command-line
  arguments.
- `trtllm-bench` is the only section that maps to TensorRT-LLM benchmark parameters,
  commands, and generated TensorRT-LLM config artifacts.
- `trtllm-bench` execution is rendered as command argv plus optional
  `config.yaml` because TensorRT-LLM does not document a single YAML file format
  that replaces the complete `trtllm-bench` CLI.
