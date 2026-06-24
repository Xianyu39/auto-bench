from __future__ import annotations

from pathlib import Path

from auto_bench.renderer import render_resolved


def test_render_single_case_cmd_and_config(tmp_path: Path) -> None:
    case_dir = tmp_path / "rendered"
    render_resolved(_resolved_payload(), case_dir)

    cmd = case_dir / "cmd.sh"
    config = case_dir / "config.yaml"
    assert cmd.exists()
    assert config.exists()
    assert cmd.stat().st_mode & 0o111
    cmd_text = cmd.read_text()
    assert 'LOG_FILE="$SCRIPT_DIR/run.log"' in cmd_text
    assert ': > "$LOG_FILE"' in cmd_text
    assert 'exec > >(tee -a "$LOG_FILE") 2>&1' in cmd_text
    assert "trtllm-bench \\" in cmd_text
    assert "  --model llama \\" in cmd_text
    assert "  throughput \\" in cmd_text
    assert '  --config "$SCRIPT_DIR/config.yaml"' in cmd_text
    assert "cuda_graph_config:" in config.read_text()


def test_render_multi_case_directories_and_run_all(tmp_path: Path) -> None:
    payload = _resolved_payload()
    payload["cases"][0]["metadata"]["gap"] = 15
    payload["cases"].append(
        {
            **payload["cases"][0],
            "case_id": "case_two",
        }
    )

    case_dirs = render_resolved(payload, tmp_path)

    assert case_dirs == [tmp_path / "case_one", tmp_path / "case_two"]
    assert (tmp_path / "case_one" / "cmd.sh").exists()
    assert (tmp_path / "case_two" / "cmd.sh").exists()
    assert (tmp_path / "run_all.sh").exists()
    run_all = (tmp_path / "run_all.sh").read_text()
    assert "case_one/cmd.sh" in run_all
    assert "sleep 15" in run_all


def test_render_run_all_can_continue_on_error(tmp_path: Path) -> None:
    payload = _resolved_payload()
    payload["cases"].append(
        {
            **payload["cases"][0],
            "case_id": "case_two",
        }
    )

    render_resolved(payload, tmp_path, continue_on_error=True)

    run_all = (tmp_path / "run_all.sh").read_text()
    assert "set -uo pipefail" in run_all
    assert "set -euo pipefail" not in run_all
    assert "FAILED=0" in run_all
    assert 'echo "auto-bench: case failed: ${case_name} (exit ${status})"' in run_all
    assert 'run_case case_one "$SCRIPT_DIR/case_one/cmd.sh"' in run_all
    assert 'run_case case_two "$SCRIPT_DIR/case_two/cmd.sh"' in run_all
    assert 'exit "$FAILED"' in run_all


def test_render_gpu_frequency_lock(tmp_path: Path) -> None:
    payload = _resolved_payload()
    payload["cases"][0]["metadata"]["gpu_frequency"] = {
        "min_mhz": 1410,
        "max_mhz": 1410,
        "gpu_ids": [0],
    }

    render_resolved(payload, tmp_path)

    cmd_text = (tmp_path / "cmd.sh").read_text()
    assert "nvidia-smi \\" in cmd_text
    assert "  -i 0 \\" in cmd_text
    assert "  -lgc 1410,1410" in cmd_text


def test_render_environment_variables(tmp_path: Path) -> None:
    payload = _resolved_payload()
    payload["cases"][0]["metadata"]["env"] = {
        "CUDA_VISIBLE_DEVICES": 0,
        "TRTLLM_LOG_LEVEL": "INFO",
    }

    render_resolved(payload, tmp_path)

    cmd_text = (tmp_path / "cmd.sh").read_text()
    assert "export CUDA_VISIBLE_DEVICES=0" in cmd_text
    assert "export TRTLLM_LOG_LEVEL=INFO" in cmd_text


def test_render_nsys_wraps_benchmark_command(tmp_path: Path) -> None:
    payload = _resolved_payload()
    payload["cases"][0]["nsys"] = {
        "enabled": True,
        "trace": "cuda,nvtx,osrt",
        "output": "$SCRIPT_DIR/profile",
    }

    render_resolved(payload, tmp_path)

    cmd_text = (tmp_path / "cmd.sh").read_text()
    assert "nsys \\" in cmd_text
    assert "  profile \\" in cmd_text
    assert "  --trace cuda,nvtx,osrt \\" in cmd_text
    assert '  -o "$SCRIPT_DIR/profile" \\' in cmd_text
    assert "  trtllm-bench \\" in cmd_text


def test_render_nsys_compare_runs_baseline_and_profile(tmp_path: Path) -> None:
    payload = _resolved_payload()
    payload["cases"][0]["commands"]["benchmark"]["argv"].extend(
        [
            "--iteration_log",
            "$SCRIPT_DIR/iter.log",
            "--artifact_dir",
            "$SCRIPT_DIR/output",
            "--dataset",
            "$SCRIPT_DIR/datasets/data.txt",
        ]
    )
    payload["cases"][0]["nsys"] = {
        "compare": True,
        "command_prefix": [
            "nsys",
            "profile",
            "--sample",
            "none",
            "-o",
            "$SCRIPT_DIR/nsys_trace",
        ],
    }

    render_resolved(payload, tmp_path)

    cmd_text = (tmp_path / "cmd.sh").read_text()
    assert 'BASELINE_DIR="$SCRIPT_DIR/baseline"' in cmd_text
    assert 'NSYS_DIR="$SCRIPT_DIR/nsys"' in cmd_text
    assert 'mkdir -p "$BASELINE_DIR" "$NSYS_DIR"' in cmd_text
    assert 'BASELINE_LOG_FILE="$BASELINE_DIR/run.log"' in cmd_text
    assert 'NSYS_LOG_FILE="$NSYS_DIR/run.log"' in cmd_text
    assert 'echo "auto-bench: running baseline"' in cmd_text
    assert 'echo "auto-bench: running nsys"' in cmd_text
    assert '} > >(tee -a "$BASELINE_LOG_FILE") 2>&1' in cmd_text
    assert '} > >(tee -a "$NSYS_LOG_FILE") 2>&1' in cmd_text
    assert cmd_text.index("  trtllm-bench \\") < cmd_text.index("  nsys \\")
    assert '  --iteration_log "$BASELINE_DIR/iter.log" \\' in cmd_text
    assert '  --iteration_log "$NSYS_DIR/iter.log" \\' in cmd_text
    assert '  --artifact_dir "$BASELINE_DIR/output" \\' in cmd_text
    assert '  --artifact_dir "$NSYS_DIR/output" \\' in cmd_text
    assert '  -o "$NSYS_DIR/nsys_trace" \\' in cmd_text
    assert '  --config "$SCRIPT_DIR/config.yaml" \\' in cmd_text
    assert '  --dataset "$SCRIPT_DIR/datasets/data.txt"' in cmd_text


def test_render_internal_script_dir_paths_expand_in_shell(tmp_path: Path) -> None:
    payload = _resolved_payload()
    payload["cases"][0]["commands"]["prepare_dataset"]["output"] = (
        "$SCRIPT_DIR/datasets/data.txt"
    )
    payload["cases"][0]["commands"]["prepare_dataset"]["argv"] = [
        "trtllm-bench",
        "prepare-dataset",
        "--output",
        "$SCRIPT_DIR/datasets/data.txt",
    ]
    payload["cases"][0]["commands"]["benchmark"]["argv"].extend(
        ["--artifact_dir", "$SCRIPT_DIR/artifacts"]
    )

    render_resolved(payload, tmp_path)

    cmd_text = (tmp_path / "cmd.sh").read_text()
    assert 'if [ ! -f "$SCRIPT_DIR/datasets/data.txt" ]; then' in cmd_text
    assert 'mkdir -p "$SCRIPT_DIR/datasets"' in cmd_text
    assert '  --output "$SCRIPT_DIR/datasets/data.txt"' in cmd_text
    assert '  --artifact_dir "$SCRIPT_DIR/artifacts"' in cmd_text


def _resolved_payload() -> dict:
    return {
        "version": "autobench.resolved/v0.1",
        "cases": [
            {
                "case_id": "case_one",
                "metadata": {"name": "case_one"},
                "trtllm-bench": {
                    "model": "llama",
                    "dataset": "/datasets/data.txt",
                    "config": {
                        "path": "/configs/generated.yaml",
                        "content": {
                            "cuda_graph_config": {"enable_padding": True}
                        },
                    },
                },
                "commands": {
                    "prepare_dataset": {
                        "if_missing": True,
                        "output": "/datasets/data.txt",
                        "argv": [
                            "trtllm-bench",
                            "--model",
                            "llama",
                            "prepare-dataset",
                            "--output",
                            "/datasets/data.txt",
                            "token_norm_dist",
                        ],
                    },
                    "write_config": {
                        "path": "/configs/generated.yaml",
                        "content": {
                            "cuda_graph_config": {"enable_padding": True}
                        },
                    },
                    "benchmark": {
                        "argv": [
                            "trtllm-bench",
                            "--model",
                            "llama",
                            "throughput",
                            "--dataset",
                            "/datasets/data.txt",
                            "--config",
                            "/configs/generated.yaml",
                        ]
                    },
                },
            }
        ],
    }
