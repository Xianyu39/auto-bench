# auto-bench

Automated TensorRT-LLM benchmark experiment orchestration.

This repository is starting with the YAML protocol that describes benchmark
experiments and resolves them into command-plus-config execution plans. The
current machine is not expected to run TensorRT-LLM itself; local development
focuses on protocol parsing, validation, command rendering, and tests.

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

## Documentation

- [YAML protocol v0.1](docs/yaml_protocol_v0.1.md)
