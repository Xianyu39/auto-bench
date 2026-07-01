# auto-bench

[![CI](https://github.com/Xianyu39/auto-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/Xianyu39/auto-bench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`auto-bench` 是一个面向 TensorRT-LLM benchmark 的实验编排工具。它用一份
YAML 描述实验矩阵、变量 sweep、派生参数、数据集生成和 `config.yaml`，再把
这些内容解析成可执行的 `trtllm-bench` 命令脚本。

当前项目关注的是协议解析、参数展开、命令渲染和脚本生成。真正运行 benchmark
仍然依赖目标机器上已经安装好 TensorRT-LLM、`trtllm-bench`、CUDA/NVIDIA
驱动以及可访问的模型和数据目录。

## 安装

从 GitHub 直接安装 CLI：

```bash
pip install "auto-bench @ git+https://github.com/Xianyu39/auto-bench.git@main"
```

作为 uv tool 安装：

```bash
uv tool install "git+https://github.com/Xianyu39/auto-bench.git@main"
```

不持久安装，直接运行：

```bash
uvx --from "git+https://github.com/Xianyu39/auto-bench.git@main" auto-bench --help
```

安装完成后可以使用：

```bash
auto-bench --help
auto-bench --version
ab --help
```

如果需要可复现安装，建议把 `@main` 替换成具体 release tag，例如
`@v0.1.12`。

## 快速开始

生成一份模板：

```bash
auto-bench template decode -o my_decode.yaml
auto-bench template prefill -o my_prefill.yaml
```

检查并展开 YAML，输出解析后的 case：

```bash
auto-bench resolve my_decode.yaml
auto-bench resolve my_decode.yaml -o resolved.yaml
```

渲染可执行脚本：

```bash
auto-bench render my_decode.yaml -o artifacts/my_decode
# Same as render, for plan-first workflows:
auto-bench plan my_decode.yaml -o artifacts/my_decode
```

`render` / `plan` 是 dry run：它只写出脚本和配置，不会运行 benchmark。生成的
`cmd.sh`、`profile.sh`、`run_all.sh` 和 `profile_all.sh` 都会自动带执行权限。

一键渲染并运行：

```bash
auto-bench run my_decode.yaml -o artifacts/my_decode
```

如果 YAML 配置了顶层 `nsys`，一键渲染并运行 profile：

```bash
auto-bench run my_decode.yaml -o artifacts/my_decode --profile
```

如果 YAML 只解析出一个 case，输出目录中会直接生成：

```text
artifacts/my_decode/
  resolved.yaml
  cmd.sh
  config.yaml        # 仅当 YAML 使用 managed config 时生成
```

如果 YAML 解析出多个 case，输出目录中会生成每个 case 的子目录，以及一个总控脚本：

```text
artifacts/my_decode/
  resolved.yaml
  run_all.sh
  decode_sweep__vars.batch_size=1__trtllm-bench.throughput.isl=128/
    cmd.sh
    config.yaml
  decode_sweep__vars.batch_size=1__trtllm-bench.throughput.isl=256/
    cmd.sh
    config.yaml
```

运行单个 case：

```bash
./artifacts/my_decode/decode_sweep__vars.batch_size=1__trtllm-bench.throughput.isl=128/cmd.sh
```

运行全部 case：

```bash
./artifacts/my_decode/run_all.sh
```

每个 `cmd.sh` 都会把 stdout/stderr 同时输出到终端和同目录下的 `run.log`。
多 case 的 `run_all.sh` 也会在总输出目录写入一个 `run.log`。

## YAML 文件结构

一份 autobench YAML 有两个必填顶层 section 和两个可选顶层 section：

```yaml
metadata:
  name: decode_sweep
  description: Decode throughput sweep example.
  tags: [decode]
  gap: 30
  gpu_frequency:
    min_mhz: 1410
    max_mhz: 1410
    gpu_ids: [0]
  env:
    CUDA_VISIBLE_DEVICES: 0

vars:
  batch_size:
    sweep: [1, 4]

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
    isl:
      sweep: [128, 256]
    osl: 64
    dataset:
      root: /mnt/datasets/autobench
      generator: token_norm_dist
      num_requests: 100
      input_mean: "${trtllm_bench.throughput.isl}"
      output_mean: "${trtllm_bench.throughput.osl}"
      input_stdev: 0
      output_stdev: 0
    config:
      content:
        cuda_graph_config:
          enable_padding: true
          batch_sizes: [1, "${vars.batch_size}"]
    max_batch_size: "${vars.batch_size}"
    max_num_tokens: "${vars.batch_size * trtllm_bench.throughput.osl}"
```

`metadata` 和 `trtllm-bench` 是必填 section，`vars` 和 `nsys` 可选。顶层不允许其它 section。

### metadata

`metadata` 用来描述实验和控制渲染脚本，它不会被转换成 `trtllm-bench`
参数。

常用字段：

- `name`：实验名称，也是 case id 的前缀。
- `description`：实验说明。
- `tags`：标签列表，方便搜索和归类。
- `gap`：多 case 运行时，相邻 case 之间的等待秒数。
- `env`：写入 `cmd.sh` 的环境变量。
- `gpu_frequency`：运行 benchmark 前调用 `nvidia-smi -lgc` 锁 GPU graphics clock。

`env` 会渲染成：

```bash
export CUDA_VISIBLE_DEVICES=0
export TRTLLM_LOG_LEVEL=INFO
```

`gpu_frequency` 可以写成固定数值：

```yaml
metadata:
  gpu_frequency: 1410
```

也可以写成更完整的配置：

```yaml
metadata:
  gpu_frequency:
    enabled: true
    min_mhz: 1410
    max_mhz: 1410
    gpu_ids: [0, 1]
```

如果省略 `gpu_ids`，脚本会渲染不带 `-i` 的 `nvidia-smi -lgc`。如果设置
`enabled: false`，则不会渲染锁频命令。

### nsys

`nsys` 是可选顶层 section，用来控制 Nsight Systems 采集。它可以写成简单开关：

```yaml
nsys: true
```

这样 benchmark 命令会被渲染为：

```bash
nsys profile -f true -t cuda,nvtx -o "$PROFILE_DIR/nsys_trace" \
  env AUTO_BENCH_RUN_DIR="$PROFILE_DIR" bash "$SCRIPT_DIR/cmd.sh"
```

如果需要采集 nsys trace：

```yaml
nsys:
  env:
    NSYS_STATS_PATH: "${runtime.run_dir}/stats"
    CUDA_VISIBLE_DEVICES: 0
    TLLM_PROFILE_START_STOP: 10-20
  output: "${runtime.run_dir}/nsys_trace"      # -o
  force_overwrite: true
  trace: [cuda, nvtx]                         # -t cuda,nvtx
  capture_range: cudaProfilerApi              # -c
  trace_fork_before_exec: true                # --trace-fork-before-exec true
  cuda-graph-trace: node
```

启用 `nsys` 时，render 会额外生成 `profile.sh`。`cmd.sh` 始终执行普通
benchmark；`profile.sh` 使用 nsys 包裹 `cmd.sh`，并把该次运行的 `run.log`、
`iter.log` 和其它由 `runtime.run_dir` 生成的输出写到 `profile/` 目录下，避免覆盖
普通运行。`config.yaml` 和 `datasets/` 这类共享输入仍保留在 case 根目录。
`env` 会作为只注入给 nsys 命令的环境变量，渲染成 nsys 的 `-e KEY=VALUE`
参数。`nsys` 下除
`enabled`、`env`、`output` 等保留字段外，
其它字段会自动渲染成 nsys 参数，例如 `capture_range` 会变成
`-c`，布尔值会渲染为 `true`/`false`，列表会用逗号连接。
已知 nsys profile 短参数会按官方 CLI 映射渲染：`backtrace -> -b`、
`capture_range -> -c`、`delay -> -y`、`duration -> -d`、`env -> -e`、
`force_overwrite -> -f`、`inherit_environment -> -n`、`nvtx_capture -> -p`、
`output -> -o`、`sample -> -s`、`show_output -> -w`、`start_later -> -Y`、
`stop_on_exit -> -x`、`trace -> -t`。
也可以把参数放在 `options` 下：

```yaml
nsys:
  options:
    sample: none
    capture_range: cudaProfilerApi
```

需要完全控制前缀时，仍可以用 `command_prefix` 覆盖整段 nsys 命令。

### vars

`vars` 用来放实验变量。它们可以参与 sweep，也可以被表达式引用，但不会直接
变成 `trtllm-bench` 参数。

例如：

```yaml
vars:
  batch_size:
    sweep: [1, 2, 4, 8]
  token_budget: 4096
```

引用方式：

```yaml
trtllm-bench:
  throughput:
    max_batch_size: "${vars.batch_size}"
    max_num_tokens: "${vars.batch_size * trtllm_bench.throughput.osl}"
```

### trtllm-bench

`trtllm-bench` 是映射到 `trtllm-bench` 的主体配置。

`trtllm-bench` 根下的普通字段会渲染在子命令之前，作为全局参数：

```yaml
trtllm-bench:
  model: meta-llama/Llama-2-7b-hf
  model_path: /mnt/engines/llama2-7b
```

会渲染成：

```bash
trtllm-bench \
  --model meta-llama/Llama-2-7b-hf \
  --model_path /mnt/engines/llama2-7b \
  throughput ...
```

`trtllm-bench` 中必须且只能出现一个 benchmark 子命令 section。目前支持：

- `throughput`
- `latency`
- `build`

子命令 section 下的普通字段会渲染在子命令之后：

```yaml
trtllm-bench:
  model: llama
  throughput:
    max_batch_size: 4
    warmup: 5
```

会渲染成：

```bash
trtllm-bench \
  --model llama \
  throughput \
  --max_batch_size 4 \
  --warmup 5
```

未知的 `trtllm-bench` 参数不会被丢弃，会按 YAML key 原样渲染为 `--<key>`。例如
`custom_command: 42` 会渲染为 `--custom_command 42`。已知参数可能会使用
manifest 中定义的 CLI 拼写，例如 `iteration_log` 会渲染为
`--iteration_log`。

布尔值规则：

- `true` 渲染成无值 flag，例如 `streaming: true` -> `--streaming`。
- `false` 不渲染。
- `null` 渲染成无值 flag，例如 `streaming: null` -> `--streaming`。
- 如果不想渲染某个参数，请直接省略该字段。

## Sweep 与表达式

任意 `metadata`、`vars`、`nsys`、`trtllm-bench` 下的字段都可以写成
sweep：

```yaml
vars:
  batch_size:
    sweep: [1, 4]

trtllm-bench:
  throughput:
    isl:
      sweep: [128, 256]
```

多个 sweep 会做笛卡尔积。上面的配置会生成 `2 * 2 = 4` 个 case。case id
格式为：

```text
<metadata.name>__<path>=<value>__<path>=<value>
```

如果需要固定参数组合，而不是让多个字段互相展开成笛卡尔积，可以使用
`cases`。`cases` 的每个条目必须是非空 mapping：

```yaml
vars:
  profile:
    cases:
      - batch_size: 1
        isl: 128
      - batch_size: 4
        isl: 256

trtllm-bench:
  throughput:
    max_batch_size: "${vars.profile.batch_size}"
    dataset:
      input_mean: "${vars.profile.isl}"
```

这会生成 2 个 case：`(batch_size=1, isl=128)` 和
`(batch_size=4, isl=256)`。`cases` 本身作为一个 sweep 维度参与展开，所以仍然
可以和普通 `sweep` 组合；例如再加一个 `backend.sweep: [pytorch, tensorrt]`
会生成 `2 * 2 = 4` 个 case。`cases` 生成的 case id 会展开条目字段，例如：

```text
<metadata.name>__vars.profile.batch_size=1__vars.profile.isl=128
```

表达式使用 `${...}`：

```yaml
max_num_tokens: "${vars.batch_size * trtllm_bench.throughput.osl}"
```

如果整个 YAML 值就是一个表达式，解析结果会保留原始类型。上例解析后是数字，
不是字符串。

字符串插值也支持：

```yaml
dataset: "/data/i${trtllm_bench.throughput.isl}_o${trtllm_bench.throughput.osl}.txt"
```

字符串插值的结果始终是字符串。

表达式可以引用这些命名空间：

- `metadata.<path>`
- `vars.<path>`
- `trtllm_bench.<path>`
- `runtime.<path>`

内置 `runtime` 值：

- `runtime.run_dir`：当前 case 目录，渲染为 `$SCRIPT_DIR`。
- `runtime.log_path`：当前 case 日志路径，渲染为 `$SCRIPT_DIR/run.log`。
- `runtime.config_path`：当前 case config 路径，渲染为 `$SCRIPT_DIR/config.yaml`。
- `runtime.dataset_dir`：当前 case 数据集目录，渲染为 `$SCRIPT_DIR/datasets`。

支持的表达式能力：

- 算术：`+`、`-`、`*`、`/`、`//`、`%`、`**`
- 比较：`==`、`!=`、`<`、`<=`、`>`、`>=`
- 布尔逻辑：`and`、`or`、`not`
- 函数：`min`、`max`、`int`、`str`、`ceil`、`floor`、`slug`

表达式不会执行任意 Python 或 shell 代码。不支持 import、文件访问和未列出的函数。

## 数据集配置

子命令下的 `dataset` 有两种写法。

第一种是用户自己准备好的数据集路径：

```yaml
trtllm-bench:
  model: llama
  throughput:
    dataset: /mnt/datasets/static.txt
```

这种写法只会在 benchmark 命令里加入 `--dataset /mnt/datasets/static.txt`，
不会生成 prepare-dataset 步骤。

第二种是 managed dataset，由 auto-bench 按需生成：

```yaml
trtllm-bench:
  model: meta-llama/Llama-2-7b-hf
  throughput:
    dataset:
      root: /mnt/datasets/autobench
      generator: token_norm_dist
      num_requests: 100
      input_mean: "${trtllm_bench.throughput.isl}"
      output_mean: "${trtllm_bench.throughput.osl}"
      input_stdev: 0
      output_stdev: 0
```

当前支持的 generator 是 `dataset`、`token_norm_dist` 和 `token_unif_dist`。
`token_norm_dist` 可用参数：

- `num_requests`
- `input_mean`
- `output_mean`
- `input_stdev`
- `output_stdev`

解析时会生成确定性的文件名：

```text
<root>/token_norm_dist__model=<slug(model)>__in=<input_mean>_<input_stdev>__out=<output_mean>_<output_stdev>__n=<num_requests>.txt
```

渲染后的 `cmd.sh` 会先检查这个文件是否存在。如果不存在，会运行：

```bash
trtllm-bench \
  --model meta-llama/Llama-2-7b-hf \
  prepare-dataset \
  --output /mnt/datasets/autobench/token_norm_dist__model=meta-llama_Llama-2-7b-hf__in=128_0__out=64_0__n=100.txt \
  token_norm_dist \
  --num-requests 100 \
  --input-mean 128 \
  --output-mean 64 \
  --input-stdev 0 \
  --output-stdev 0
```

prepare-dataset 发生在 benchmark 命令之前，不属于 benchmark 测量区间。

## config.yaml 配置

子命令下的 `config` 也有两种写法。

第一种是用户自己维护的 config 文件路径：

```yaml
trtllm-bench:
  throughput:
    config: /mnt/configs/static.yaml
```

这种写法只会在 benchmark 命令里加入 `--config /mnt/configs/static.yaml`，
不会生成文件。

第二种是 managed config：

```yaml
trtllm-bench:
  throughput:
    config:
      content:
        cuda_graph_config:
          enable_padding: true
          batch_sizes: [1, "${vars.batch_size}"]
        enable_attention_dp: true
```

`content` 必须是 YAML mapping。auto-bench 会在每个 case 目录下生成
`config.yaml`，内容就是表达式解析后的 `content`。benchmark 命令中对应的
`--config` 会自动改成 case-local 路径：

```bash
--config "$SCRIPT_DIR/config.yaml"
```

如果省略 `config` 或写成 `config: null`，则不会生成 config 文件，也不会渲染
`--config`。

## 解析输出

`resolve` 命令会输出标准化后的 YAML：

```bash
auto-bench resolve examples/decode_sweep.yaml -o artifacts/decode_resolved.yaml
```

输出结构：

```yaml
version: autobench.resolved/v0.1
cases:
  - case_id: decode_sweep__vars.batch_size=1__trtllm-bench.throughput.isl=128
    metadata: ...
    vars: ...
    runtime: ...
    trtllm-bench: ...
    commands:
      prepare_dataset: ...
      write_config: ...
      benchmark:
        argv:
          - trtllm-bench
          - --model
          - meta-llama/Llama-2-7b-hf
          - throughput
          - --dataset
          - /mnt/datasets/...
```

重点字段：

- `case_id`：稳定的 case 名称，也用于多 case 渲染时的目录名。
- `runtime`：脚本运行时路径。
- `trtllm-bench`：展开 sweep 和表达式后的配置，便于审计。
- `commands.prepare_dataset`：managed dataset 的准备命令；非 managed dataset 时为 `null`。
- `commands.write_config`：managed config 的写文件计划；用户自管 config 或无 config 时为 `null`。
- `commands.benchmark.argv`：最终 benchmark 命令的 argv 列表。

## 渲染产物说明

`render` 命令会写入：

- `resolved.yaml`：完整解析结果。
- `cmd.sh`：单个 case 的可执行脚本。
- `profile.sh`：当顶层 `nsys` 启用时生成，用 nsys 包裹同目录下的 `cmd.sh`。
- `config.yaml`：managed config 内容，仅在需要时生成。
- `run_all.sh`：多 case 时生成，用于按顺序运行所有 case。
- `profile_all.sh`：多 case 且存在 nsys case 时生成，只按顺序运行各 case 的
  `profile.sh`。
- `run.log`：执行脚本时产生，不是 render 阶段生成。

`render` 只生成产物，不执行 benchmark；生成的 shell 脚本都会自动设置执行权限。
需要渲染后立即运行时，可以使用 `auto-bench run`。加 `--profile` 时会运行
`profile.sh` 或 `profile_all.sh`，否则运行 `cmd.sh` 或 `run_all.sh`。

每个 `cmd.sh` 做的事情：

1. 设置 `set -euo pipefail`。
2. 计算 `SCRIPT_DIR`。
3. 清空并写入当前 case 的 `run.log`。
4. 导出 `metadata.env` 中的环境变量。
5. 按 `metadata.gpu_frequency` 锁 GPU frequency。
6. 如果使用 managed dataset 且数据集文件不存在，先运行 `prepare-dataset`。
7. 执行最终 `trtllm-bench` benchmark 命令。

每个 `profile.sh` 会创建 `profile/` 目录，把 nsys trace 写到该目录，并通过
`AUTO_BENCH_RUN_DIR` 让内部的 `cmd.sh` 把本次 profile 运行的日志和产物也写到
`profile/` 下。

多 case 的 `run_all.sh` 会按解析顺序执行每个 case 的 `cmd.sh`。如果
`metadata.gap` 大于 0，会在相邻 case 之间 sleep 对应秒数。
默认情况下，任意 case 失败都会让 `run_all.sh` 停止。渲染时加
`--continue-on-error` 可以改为记录失败并继续运行后续 case，脚本最后仍会用
非零退出码表示至少有一个 case 失败：

```bash
auto-bench render examples/decode_sweep.yaml -o artifacts/decode_sweep --continue-on-error
```

## 收集结果

运行完成后，可以从 render 输出目录中收集 benchmark 指标：

```bash
auto-bench collect_results artifacts/prefill_sweep --framework trtllm-bench
```

当前只支持 `trtllm-bench`，因此必须显式指定：

```bash
--framework trtllm-bench
```

默认输出 CSV 到 stdout。写入文件：

```bash
auto-bench collect_results artifacts/prefill_sweep \
  --framework trtllm-bench \
  -o artifacts/prefill_sweep/results.csv
```

也可以输出 YAML：

```bash
auto-bench collect_results artifacts/prefill_sweep \
  --framework trtllm-bench \
  --format yaml \
  -o artifacts/prefill_sweep/results.yaml
```

`collect_results` 会读取输出目录中的 `resolved.yaml` 来确定 case 列表，然后读取
每个 case 的日志。普通 case 读取 `run.log`；配置了 `nsys` 的 case 会读取
`run.log` 和 `profile/run.log`，并在结果中用 `variant` 区分两份数据。如果只
运行了其中一个脚本，另一行会显示 `missing_log`。

CSV 中会包含：

- `case_id`：case 名称。
- `variant`：`default` 或 `profile`。
- `status`：`ok`、`missing_log` 或 `no_metrics`。
- `log_path`：读取的日志路径。
- `metadata.*`：从 resolved case 中展开的 metadata 字段。
- `vars.*`：从 resolved case 中展开的变量字段。
- `nsys.*`：从 resolved case 中展开的 nsys 采集配置字段。
- `metrics.*`：从 `trtllm-bench` 日志中提取出的指标。

目前支持解析常见的 `trtllm-bench` key-value 或简单表格输出，例如：

```text
Request throughput (req/sec): 123.4
Output token throughput (tokens/sec): 5678
| Average latency (ms) | 12.25 |
```

## 常见用法

### Decode sweep

仓库中提供了 decode 示例：

```bash
auto-bench resolve examples/decode_sweep.yaml
auto-bench render examples/decode_sweep.yaml -o artifacts/decode_sweep
bash artifacts/decode_sweep/run_all.sh
```

它会 sweep `vars.batch_size` 和 `trtllm-bench.throughput.isl`，并用表达式计算
`max_batch_size`、`max_num_tokens` 和数据集长度。

### Prefill sweep

仓库中提供了 prefill 示例：

```bash
auto-bench resolve examples/prefill_sweep.yaml
auto-bench render examples/prefill_sweep.yaml -o artifacts/prefill_sweep
bash artifacts/prefill_sweep/run_all.sh
```

prefill 示例主要 sweep batch size，并展示了 `ep`、`tp`、`warmup`、
`iteration_log`、`enable_attention_dp` 等字段的写法。

### 使用当前目录作为 case-local 数据目录

如果希望数据集放在每个 case 目录下，可以使用 `runtime.dataset_dir`：

```yaml
trtllm-bench:
  model: llama
  throughput:
    dataset:
      root: "${runtime.dataset_dir}"
      generator: token_norm_dist
      num_requests: 100
      input_mean: 128
      output_mean: 64
      input_stdev: 0
      output_stdev: 0
```

渲染后的脚本会使用 `$SCRIPT_DIR/datasets/...`，移动整个 artifacts 目录后仍然可运行。

### 传递尚未内置到 manifest 的 TensorRT-LLM 参数

直接写在 `trtllm-bench` 根或子命令下即可：

```yaml
trtllm-bench:
  model: llama
  custom_global: root-value
  throughput:
    dataset: /mnt/datasets/static.txt
    custom_command: 42
```

会渲染为：

```bash
trtllm-bench \
  --model llama \
  --custom_global root-value \
  throughput \
  --dataset /mnt/datasets/static.txt \
  --custom_command 42
```

这些参数会被保留并渲染，同时 `resolve` / `render` 会在 stderr 和
`resolved.yaml` 的 `warnings` 字段里提示：

```text
Warning: option 'trtllm-bench.custom_global': This option is not documented for TensorRT-LLM 1.3.0rc13 or is not supported by auto-bench yet.
```

未知子命令、未知 dataset generator、未知 dataset generator 参数也采用同样策略：
给出英文 warning，但继续展开和渲染。

## 排错

常见错误：

- 顶层 section 写错：只允许 `metadata`、`vars`、`trtllm-bench`。
- 缺少 `metadata` 或 `trtllm-bench`。
- 同时写了多个子命令，例如同时存在 `throughput` 和 `latency`。
- `sweep` 不是非空 list。
- 表达式引用了不存在的路径。
- 表达式引用了尚未展开的 sweep object。
- managed dataset 缺少 `root` 或 `generator`。
- managed config 缺少 `content`，或 `content` 不是 mapping。
- 解析后仍残留 `${...}`。

可以先用 `resolve` 检查 YAML，不急着生成可执行脚本：

```bash
auto-bench resolve my_experiment.yaml
```

错误信息会尽量包含出错的 YAML path。

## 本地开发

安装开发依赖：

```bash
uv sync --dev
```

运行测试和检查：

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

使用本地源码运行 CLI：

```bash
uv run auto-bench --version
uv run auto-bench resolve examples/decode_sweep.yaml
uv run auto-bench render examples/decode_sweep.yaml -o artifacts/decode_sweep
```

生成模板：

```bash
uv run auto-bench template minimal
uv run auto-bench template decode -o examples/my_decode.yaml
uv run auto-bench template prefill -o examples/my_prefill.yaml
```

## 协议文档

更底层的协议定义见 [YAML protocol v0.1](docs/yaml_protocol_v0.1.md)。

## 许可证

本项目使用 [MIT License](LICENSE)。
