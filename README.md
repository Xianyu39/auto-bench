# auto-bench

Automated TensorRT-LLM benchmark experiment orchestration.

This repository is starting with the YAML protocol that describes benchmark
experiments and resolves them into command-plus-config execution plans. The
current machine is not expected to run TensorRT-LLM itself; local development
focuses on protocol parsing, validation, command rendering, and tests.

## Installation

Install the CLI directly from GitHub with pip:

```bash
pip install "auto-bench @ git+https://github.com/Xianyu39/auto-bench.git@main"
```

Install it as a uv-managed tool:

```bash
uv tool install "git+https://github.com/Xianyu39/auto-bench.git@main"
```

Run it without a persistent install:

```bash
uvx --from "git+https://github.com/Xianyu39/auto-bench.git@main" auto-bench --help
```

After installing, the CLI is available as:

```bash
auto-bench --help
```

For reproducible installs, replace `@main` with a release tag such as `@v0.1.0`
after tagging a release.

## Development

Install dependencies with uv:

```bash
uv sync --dev
```

Run the local checks:

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

Show the CLI entrypoint:

```bash
uv run auto-bench --version
```

Resolve an experiment YAML into command-plus-config cases:

```bash
uv run auto-bench resolve examples/decode_sweep.yaml
```

Render executable artifacts:

```bash
uv run auto-bench render examples/decode_sweep.yaml -o artifacts/decode_sweep
```

For each resolved case, render creates a `cmd.sh`. If the case uses a managed
config, render also creates a local `config.yaml` next to `cmd.sh`.

Generate a starter YAML template:

```bash
uv run auto-bench template decode -o examples/my_decode.yaml
uv run auto-bench template prefill
```

## Documentation

- [YAML protocol v0.1](docs/yaml_protocol_v0.1.md)

## Release Notes

This project is installable from GitHub as a Python package. Before making the
repository public, choose and add an open-source license. After adding a
license, also add the matching `license` metadata to `pyproject.toml`.
