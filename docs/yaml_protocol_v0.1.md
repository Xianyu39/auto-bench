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

- A fully materialized `trtllm` config with no missing TensorRT-LLM benchmark
  parameters needed for the fixed supported TensorRT-LLM version.
- A dataset path that exists before the benchmark command runs.
- A TensorRT-LLM `config.yaml` artifact when deep runtime or LLM API options
  are configured.
- A benchmark command expressed as an argv list.

If a managed dataset is configured and the dataset file is missing, autobench
must generate it before running the benchmark command.

## Top-Level Structure

An autobench YAML file has exactly two top-level sections:

```yaml
metadata:
  name: llama2_7b_decode
  description: Decode benchmark on H100.
  tags: [decode, h100]

trtllm:
  model: meta-llama/Llama-2-7b-hf
  command: throughput
  isl: 1024
  osl: 128
  batch_size:
    sweep: [1, 2, 4, 8]
  dataset:
    root: /data/autobench/datasets
    generator: token-norm-dist
    num_requests: 1000
    input_mean: "${trtllm.isl}"
    output_mean: "${trtllm.osl}"
    input_stdev: 0
    output_stdev: 0
  config:
    root: /data/autobench/configs
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

Additional metadata fields are allowed. They may be referenced by expressions
with `metadata.<path>`.

### `trtllm`

`trtllm` describes TensorRT-LLM benchmark parameters. It is the only section
that maps to benchmark parameters and benchmark commands.

Every key under `trtllm` must correspond to a known parameter in the fixed
TensorRT-LLM benchmark manifest or a known autobench protocol field.

`trtllm.command`, `trtllm.dataset`, and `trtllm.config` are special protocol
fields.
`trtllm.command` selects the TensorRT-LLM benchmark subcommand. In input YAML,
`trtllm.dataset` may be either a dataset path or a managed dataset
specification. In resolved output, `trtllm.dataset` must always be a dataset
path.

`trtllm.config` describes the TensorRT-LLM YAML config artifact passed with
`--config`. In resolved output, it must be either `null` or a materialized
config artifact with a path and YAML content.

## Parameter Value Types

Fields in `metadata` and `trtllm` may use the following value types.

### Scalar

A scalar is copied into each resolved case as-is.

```yaml
trtllm:
  isl: 1024
  osl: 128
  streaming: false
```

### Sweep

A sweep field expands the experiment into multiple cases.

```yaml
trtllm:
  batch_size:
    sweep: [1, 2, 4, 8]
```

After expansion, `trtllm.batch_size` is a scalar in each resolved case.

A mapping is treated as a sweep object only when it has exactly one key,
`sweep`. The value of `sweep` must be a non-empty list.

### Expression

An expression is a string whose entire value is a single `${...}` block.

```yaml
trtllm:
  max_num_tokens: "${trtllm.batch_size * trtllm.osl}"
```

When the whole YAML value is an expression, the resolved value keeps the
expression result type. For example, the result of the expression above is an
integer, not a string.

### String Interpolation

A string interpolation is a string that contains one or more `${...}` blocks but
is not itself a single expression.

```yaml
trtllm:
  dataset: "/data/llama2/i${trtllm.isl}_o${trtllm.osl}.txt"
```

Interpolated values are always resolved to strings.

### Managed Dataset

`trtllm.dataset` may describe a dataset that autobench manages. The resolver
turns this object into a deterministic dataset path, and the runner generates
the file on demand.

```yaml
trtllm:
  model: meta-llama/Llama-2-7b-hf
  isl:
    sweep: [128, 512]
  osl: 128
  dataset:
    root: /data/autobench/datasets
    generator: token-norm-dist
    num_requests: 1000
    input_mean: "${trtllm.isl}"
    output_mean: "${trtllm.osl}"
    input_stdev: 0
    output_stdev: 0
```

Required managed dataset fields:

- `root`: directory where autobench stores generated datasets.
- `generator`: TensorRT-LLM dataset generator name. v0.1 supports
  `token-norm-dist`.

Generator arguments are written as fields in the dataset object. For
`token-norm-dist`, common arguments are:

- `num_requests`
- `input_mean`
- `output_mean`
- `input_stdev`
- `output_stdev`

The resolver must generate a readable, deterministic filename from the resolved
dataset fields. The recommended filename format is:

```text
<generator>__model=<slug(trtllm.model)>__in=<input_mean>_<input_stdev>__out=<output_mean>_<output_stdev>__n=<num_requests>.txt
```

Example:

```text
/data/autobench/datasets/token-norm-dist__model=meta-llama_Llama-2-7b-hf__in=128_0__out=128_0__n=1000.txt
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

`trtllm.config` describes a per-case TensorRT-LLM `config.yaml` file. This is
used for options that TensorRT-LLM expects through `--config` rather than as
direct benchmark CLI flags.

```yaml
trtllm:
  config:
    root: /data/autobench/configs
    content:
      cuda_graph_config:
        enable_padding: true
        batch_sizes:
          - 1
          - "${trtllm.batch_size}"
      print_iter_log: true
```

Required managed config fields:

- `root`: directory where autobench stores generated config files.
- `content`: YAML mapping to write into the generated config file.

The resolver must evaluate expressions and sweep values inside `content`. The
resolved content must be valid YAML data and must not contain `sweep` objects or
unresolved `${...}` expressions.

The resolver must generate a readable, deterministic config filename from the
resolved config content and the case identity. The recommended filename format
is:

```text
<case_id>__config-<short_hash(content)>.yaml
```

The hash is included so that two cases with the same visible sweep values but
different config content cannot collide.

If `trtllm.config` is omitted or `null`, no config artifact is generated and
the benchmark command does not include `--config`.

If `trtllm.config` is a string path, autobench treats it as a user-managed
config file. The benchmark command includes `--config <path>`, but autobench
does not generate the file.

If `trtllm.config` is a managed config object, the resolved case must include
the generated config path, the generated config content, and a file-write plan.

## Path References And Expressions

Expressions may reference values with dotted paths:

- `metadata.<path>`
- `trtllm.<path>`

Examples:

```yaml
metadata:
  model_family: llama2

trtllm:
  isl: 1024
  osl: 128
  batch_size:
    sweep: [1, 2, 4]
  dataset: "/data/${metadata.model_family}/i${trtllm.isl}_o${trtllm.osl}.txt"
  max_batch_size: "${trtllm.batch_size}"
  max_num_tokens: "${trtllm.batch_size * trtllm.osl}"
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
- Path references under `metadata` and `trtllm`
- Functions: `min`, `max`, `int`, `str`, `ceil`, `floor`, `slug`

The expression evaluator must not execute arbitrary Python or shell code. File
access, imports, attribute access outside `metadata` and `trtllm`, and calls to
unlisted functions are forbidden.

`slug(value)` converts a value into a path-safe string. It is intended for
dataset filenames, config filenames, and case identifiers.

## Sweep Expansion Rules

All `sweep` fields are expanded by Cartesian product.

```yaml
trtllm:
  isl:
    sweep: [128, 512]
  batch_size:
    sweep: [1, 2, 4]
```

This produces `2 * 3 = 6` cases.

Expansion order must be deterministic:

1. Traverse `metadata` and `trtllm` in YAML order.
2. Collect sweep fields in that order.
3. Expand Cartesian products in collected order.

The resolved `case_id` should be stable for the same input. The recommended
format is:

```text
<metadata.name>__<path>=<value>__<path>=<value>
```

Example:

```text
llama2_7b_decode__trtllm.isl=128__trtllm.batch_size=4
```

If `metadata.name` is missing, the resolver must use a deterministic fallback
such as `experiment`.

## Resolution Order

A resolver must process the input in this order:

1. Parse YAML.
2. Validate that the only top-level keys are `metadata` and `trtllm`.
3. Load the fixed TensorRT-LLM benchmark parameter manifest.
4. Collect all sweep fields from `metadata` and `trtllm`.
5. Expand the Cartesian product of sweep values.
6. For each expanded case, replace sweep objects with the current scalar values.
7. Evaluate expressions and string interpolations.
8. Resolve managed `trtllm.dataset` objects into dataset paths and
   prepare-dataset commands.
9. Resolve managed `trtllm.config` objects into config paths, config content,
   and file-write plans.
10. Apply manifest defaults for missing optional `trtllm` parameters.
11. Validate required `trtllm` parameters.
12. Reject unknown `trtllm` parameters.
13. Render benchmark commands.
14. Emit resolved cases.

Resolved cases must not contain `sweep` objects, managed dataset objects,
managed config objects, or unresolved `${...}` expressions.

## Resolved Case Output

The resolver output is a list of cases. Each case contains the materialized
configuration and executable commands.

```yaml
version: autobench.resolved/v0.1
cases:
  - case_id: llama2_7b_decode__trtllm.batch_size=1
    metadata:
      name: llama2_7b_decode
      tags: [decode, h100]
    trtllm:
      model: meta-llama/Llama-2-7b-hf
      isl: 1024
      osl: 128
      batch_size: 1
      max_batch_size: 1
      max_num_tokens: 128
      dataset: /data/llama2/i1024_o128.txt
      config:
        path: /data/autobench/configs/llama2_7b_decode__trtllm.batch_size=1__config-a1b2c3d4.yaml
        content:
          cuda_graph_config:
            enable_padding: true
            batch_sizes: [1]
      warmup: 5
      iterations: 30
    commands:
      prepare_dataset: null
      write_config:
        path: /data/autobench/configs/llama2_7b_decode__trtllm.batch_size=1__config-a1b2c3d4.yaml
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
          - /data/autobench/configs/llama2_7b_decode__trtllm.batch_size=1__config-a1b2c3d4.yaml
          - --max_batch_size
          - "1"
          - --max_num_tokens
          - "128"
```

Each resolved `trtllm` object must contain all required TensorRT-LLM benchmark
parameters plus the resolved autobench protocol fields. Parameters with
manifest defaults must be materialized in the resolved output.

If `trtllm.dataset` is managed, `commands.prepare_dataset` must contain an
argv list. If `trtllm.dataset` is already a path, `commands.prepare_dataset`
must be `null`.

If `trtllm.config` is managed, `commands.write_config` must contain the config
path and content to write. If `trtllm.config` is omitted, `commands.write_config`
must be `null`. If `trtllm.config` is a user-managed path,
`commands.write_config` must be `null`.

The benchmark command and optional config artifact are the canonical executable
artifacts of a resolved case. `trtllm` remains in the output for traceability
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

- `trtllm-bench ... --config <generated-config.yaml>` command argv.
- The generated `config.yaml` content for options that belong in TensorRT-LLM's
  YAML config surface.

### Benchmark Command

`trtllm.command` selects the benchmark subcommand. v0.1 supports:

- `throughput`
- `latency`
- `build`

If `trtllm.command` is missing, the resolver defaults to `throughput`.

The rendered benchmark command format is:

```text
trtllm-bench [global options] <command> [command options]
```

`trtllm.model` maps to the global `--model` option. `trtllm.dataset` maps to
the command option `--dataset` for commands that accept datasets.
`trtllm.config.path` or a user-managed `trtllm.config` path maps to `--config`.

Option names are rendered from manifest metadata. The manifest must classify
each parameter as one of:

- `global`: rendered before the subcommand.
- `command`: rendered after the subcommand.
- `protocol`: consumed by autobench and not rendered directly.
- `config`: written into the generated TensorRT-LLM config artifact.

Boolean options are rendered only when true, unless the TensorRT-LLM manifest
defines an explicit value-taking boolean flag.

### Config Artifact

The generated config artifact is a YAML file whose content is exactly the
resolved `trtllm.config.content` mapping.

Example:

```yaml
commands:
  write_config:
    path: /data/autobench/configs/llama2_7b_decode__config-a1b2c3d4.yaml
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
same config path and byte-equivalent YAML content.

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
    output: /data/autobench/datasets/token-norm-dist__model=meta-llama_Llama-2-7b-hf__in=128_0__out=128_0__n=1000.txt
    argv:
      - trtllm-bench
      - --model
      - meta-llama/Llama-2-7b-hf
      - prepare-dataset
      - --output
      - /data/autobench/datasets/token-norm-dist__model=meta-llama_Llama-2-7b-hf__in=128_0__out=128_0__n=1000.txt
      - token-norm-dist
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

## Completeness Validation

The resolver relies on an internal manifest for the fixed supported
TensorRT-LLM benchmark version.

The manifest defines:

- Parameter names.
- Parameter types.
- Whether a parameter is required.
- Default values for optional parameters.

Validation rules:

- Missing required `trtllm` parameters are errors.
- Missing optional `trtllm` parameters with manifest defaults are filled.
- Unknown `trtllm` parameters or protocol fields are errors.
- Values must match manifest types after expression evaluation.
- Resolved `trtllm` keys must be either TensorRT-LLM manifest parameters or
  known autobench protocol fields after defaults are applied.
- Input `trtllm.dataset` may be a managed dataset object, but resolved
  `trtllm.dataset` must be a string path.
- Input `trtllm.config` may be absent, `null`, a user-managed path, or a
  managed config object. Resolved `trtllm.config` must be `null` or an object
  with `path` and `content`.
- Managed dataset generator arguments must be known for the selected generator.
- Managed dataset filenames must be deterministic for the resolved generator
  fields.
- Managed config content must be a YAML mapping.
- Managed config filenames must be deterministic for the resolved config
  content.

`metadata` is not validated against the TensorRT-LLM manifest.

## Error Handling Rules

The resolver must fail the whole YAML file on any of the following errors:

- Unknown top-level section.
- Missing `metadata` or `trtllm` section.
- Empty or non-list `sweep`.
- Invalid managed dataset object.
- Unknown managed dataset generator.
- Unknown managed dataset generator argument.
- Invalid managed config object.
- Managed config content that is not a YAML mapping.
- Unknown `trtllm` parameter.
- Missing required `trtllm` parameter.
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

trtllm:
  model: meta-llama/Llama-2-7b-hf
  command: throughput
  engine_dir: /mnt/engines/llama2-7b

  isl:
    sweep: [128, 512]

  osl: 128

  batch_size:
    sweep: [1, 4]

  kv_cache_dtype:
    sweep: [fp16, fp8]

  dataset:
    root: /mnt/datasets/autobench
    generator: token-norm-dist
    num_requests: 1000
    input_mean: "${trtllm.isl}"
    output_mean: "${trtllm.osl}"
    input_stdev: 0
    output_stdev: 0

  config:
    root: /mnt/configs/autobench
    content:
      cuda_graph_config:
        enable_padding: true
        batch_sizes: [1, 2, 4]
      kv_cache_config:
        free_gpu_memory_fraction: 0.9
      print_iter_log: true

  max_batch_size: "${trtllm.batch_size}"
  max_num_tokens: "${trtllm.batch_size * trtllm.osl}"

  warmup: 5
  iterations: 30
  streaming: false
```

This input produces `2 * 2 * 2 = 8` resolved cases before manifest defaulting.
One resolved case:

```yaml
version: autobench.resolved/v0.1
cases:
  - case_id: llama2_7b_decode__trtllm.isl=128__trtllm.batch_size=4__trtllm.kv_cache_dtype=fp8
    metadata:
      name: llama2_7b_decode
      description: Decode throughput sweep on H100.
      tags: [decode, h100]
      model_family: llama2
    trtllm:
      model: meta-llama/Llama-2-7b-hf
      command: throughput
      engine_dir: /mnt/engines/llama2-7b
      isl: 128
      osl: 128
      batch_size: 4
      kv_cache_dtype: fp8
      dataset: /mnt/datasets/autobench/token-norm-dist__model=meta-llama_Llama-2-7b-hf__in=128_0__out=128_0__n=1000.txt
      config:
        path: /mnt/configs/autobench/llama2_7b_decode__trtllm.isl=128__trtllm.batch_size=4__trtllm.kv_cache_dtype=fp8__config-a1b2c3d4.yaml
        content:
          cuda_graph_config:
            enable_padding: true
            batch_sizes: [1, 2, 4]
          kv_cache_config:
            free_gpu_memory_fraction: 0.9
          print_iter_log: true
      max_batch_size: 4
      max_num_tokens: 512
      warmup: 5
      iterations: 30
      streaming: false
      # Additional manifest parameters are filled here.
    commands:
      prepare_dataset:
        if_missing: true
        output: /mnt/datasets/autobench/token-norm-dist__model=meta-llama_Llama-2-7b-hf__in=128_0__out=128_0__n=1000.txt
        argv:
          - trtllm-bench
          - --model
          - meta-llama/Llama-2-7b-hf
          - prepare-dataset
          - --output
          - /mnt/datasets/autobench/token-norm-dist__model=meta-llama_Llama-2-7b-hf__in=128_0__out=128_0__n=1000.txt
          - token-norm-dist
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
        path: /mnt/configs/autobench/llama2_7b_decode__trtllm.isl=128__trtllm.batch_size=4__trtllm.kv_cache_dtype=fp8__config-a1b2c3d4.yaml
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
          - throughput
          - --dataset
          - /mnt/datasets/autobench/token-norm-dist__model=meta-llama_Llama-2-7b-hf__in=128_0__out=128_0__n=1000.txt
          - --config
          - /mnt/configs/autobench/llama2_7b_decode__trtllm.isl=128__trtllm.batch_size=4__trtllm.kv_cache_dtype=fp8__config-a1b2c3d4.yaml
          - --engine_dir
          - /mnt/engines/llama2-7b
          - --max_batch_size
          - "4"
          - --max_num_tokens
          - "512"
```

## Parser Test Scenarios

Future parser tests should cover:

- A single static configuration resolves into one complete case.
- Two sweep fields expand into a Cartesian product.
- Expressions can read current sweep values and preserve result types.
- String interpolation can generate dataset paths.
- Managed datasets resolve to deterministic dataset paths.
- Managed datasets render prepare-dataset commands.
- Existing dataset paths produce `prepare_dataset: null`.
- Managed config objects resolve to deterministic config paths and content.
- Managed config objects render `write_config` plans.
- User-managed config paths render `--config <path>` without a write plan.
- Benchmark commands render as argv lists.
- Unknown `trtllm` parameters fail validation.
- Missing required parameters fail validation.
- Expressions referencing nonexistent paths fail validation.
- Resolved output contains no `sweep` objects, managed dataset objects, managed
  config objects, or unresolved expressions.

## Not Supported In v0.1

The following features are intentionally out of scope for v0.1:

- A separate `vars` section.
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
- `trtllm` is the only section that maps to TensorRT-LLM benchmark parameters,
  commands, and generated TensorRT-LLM config artifacts.
- `trtllm-bench` execution is rendered as command argv plus optional
  `config.yaml` because TensorRT-LLM does not document a single YAML file format
  that replaces the complete `trtllm-bench` CLI.
