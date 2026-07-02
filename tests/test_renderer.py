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
    assert 'RUN_DIR="${AUTO_BENCH_RUN_DIR:-$SCRIPT_DIR}"' in cmd_text
    assert 'LOG_FILE="$RUN_DIR/run.log"' in cmd_text
    assert ': > "$LOG_FILE"' in cmd_text
    assert 'if [ "${AUTO_BENCH_QUIET:-}" = "1" ]; then' in cmd_text
    assert '  exec > "$LOG_FILE" 2>&1' in cmd_text
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
    assert (tmp_path / "case_one" / "cmd.sh").stat().st_mode & 0o111
    assert (tmp_path / "case_two" / "cmd.sh").stat().st_mode & 0o111
    assert (tmp_path / "run_all.sh").stat().st_mode & 0o111
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
        "env": {
            "CUDA_VISIBLE_DEVICES": 0,
            "NSYS_OUTPUT_DIR": "$SCRIPT_DIR/nsys_env",
        },
        "trace": ["cuda", "nvtx", "osrt"],
        "sample": "none",
        "capture_range": "cudaProfilerApi",
        "capture_range_end": "stop-shutdown",
        "trace_fork_before_exec": True,
        "cuda_memory_usage": True,
        "output": "$SCRIPT_DIR/profile",
    }

    render_resolved(payload, tmp_path)

    cmd_text = (tmp_path / "cmd.sh").read_text()
    assert "nsys" not in cmd_text
    assert "  trtllm-bench \\" in cmd_text
    profile = tmp_path / "profile.sh"
    assert profile.exists()
    assert profile.stat().st_mode & 0o111
    profile_text = profile.read_text()
    assert 'PROFILE_DIR="$SCRIPT_DIR/profile"' in profile_text
    assert 'PROFILE_LOG_FILE="$PROFILE_DIR/profile.log"' in profile_text
    assert 'if [ "${AUTO_BENCH_QUIET:-}" = "1" ]; then' in profile_text
    assert '  exec > "$PROFILE_LOG_FILE" 2>&1' in profile_text
    assert "nsys \\" in profile_text
    assert "  profile \\" in profile_text
    assert "  -e CUDA_VISIBLE_DEVICES=0 \\" in profile_text
    assert '  -e NSYS_OUTPUT_DIR="$PROFILE_DIR/nsys_env" \\' in profile_text
    assert "  -f true \\" in profile_text
    assert "  -t cuda,nvtx,osrt \\" in profile_text
    assert "  -s none \\" in profile_text
    assert "  -c cudaProfilerApi \\" in profile_text
    assert "  --capture-range-end stop-shutdown \\" in profile_text
    assert "  --trace-fork-before-exec true \\" in profile_text
    assert "  --cuda-memory-usage true \\" in profile_text
    assert '  -o "$PROFILE_DIR/profile" \\' in profile_text
    assert "  AUTO_BENCH_RUN_DIR=\"$PROFILE_DIR\" \\" in profile_text
    assert "  bash \\" in profile_text
    assert '  "$SCRIPT_DIR/cmd.sh"' in profile_text


def test_render_nsys_profile_script_isolates_outputs(tmp_path: Path) -> None:
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
        "sample": "none",
    }

    render_resolved(payload, tmp_path)

    cmd_text = (tmp_path / "cmd.sh").read_text()
    assert '  --iteration_log "$RUN_DIR/iter.log" \\' in cmd_text
    assert '  --artifact_dir "$RUN_DIR/output" \\' in cmd_text
    assert '  --config "$SCRIPT_DIR/config.yaml" \\' in cmd_text
    assert '  --dataset "$SCRIPT_DIR/datasets/data.txt"' in cmd_text
    assert "nsys" not in cmd_text

    profile_text = (tmp_path / "profile.sh").read_text()
    assert '  -o "$PROFILE_DIR/nsys_trace" \\' in profile_text
    assert '  AUTO_BENCH_RUN_DIR="$PROFILE_DIR" \\' in profile_text
    assert "  bash \\" in profile_text
    assert '  "$SCRIPT_DIR/cmd.sh"' in profile_text


def test_render_nsys_nested_options(tmp_path: Path) -> None:
    payload = _resolved_payload()
    payload["cases"][0]["nsys"] = {
        "options": {
            "sample": "none",
            "capture_range": "cudaProfilerApi",
        },
    }

    render_resolved(payload, tmp_path)

    profile_text = (tmp_path / "profile.sh").read_text()
    assert "  -s none \\" in profile_text
    assert "  -c cudaProfilerApi \\" in profile_text
    assert '  -o "$PROFILE_DIR/nsys_trace" \\' in profile_text


def test_render_nsys_known_short_options(tmp_path: Path) -> None:
    payload = _resolved_payload()
    payload["cases"][0]["nsys"] = {
        "backtrace": "fp",
        "delay": 5,
        "duration": 30,
        "inherit_environment": False,
        "nvtx_capture": "range@domain",
        "sample": "none",
        "show_output": True,
        "start_later": True,
        "stop_on_exit": False,
        "capture_range_end": "stop-shutdown",
    }

    render_resolved(payload, tmp_path)

    profile_text = (tmp_path / "profile.sh").read_text()
    assert "  -b fp \\" in profile_text
    assert "  -y 5 \\" in profile_text
    assert "  -d 30 \\" in profile_text
    assert "  -n false \\" in profile_text
    assert "  -p range@domain \\" in profile_text
    assert "  -s none \\" in profile_text
    assert "  -w true \\" in profile_text
    assert "  -Y true \\" in profile_text
    assert "  -x false \\" in profile_text
    assert "  --capture-range-end stop-shutdown \\" in profile_text


def test_render_nsys_null_options_are_omitted(tmp_path: Path) -> None:
    payload = _resolved_payload()
    payload["cases"][0]["nsys"] = {
        "trace": None,
        "force_overwrite": None,
        "sample": "none",
    }

    render_resolved(payload, tmp_path)

    profile_text = (tmp_path / "profile.sh").read_text()
    assert "--trace" not in profile_text
    assert "--force-overwrite" not in profile_text
    assert "  -t " not in profile_text
    assert "  -f " not in profile_text
    assert "  -s none \\" in profile_text


def test_render_nsys_profile_rewrites_nsys_env_paths(tmp_path: Path) -> None:
    payload = _resolved_payload()
    payload["cases"][0]["nsys"] = {
        "env": {
            "NSYS_STATS_PATH": "$SCRIPT_DIR/stats",
        },
    }

    render_resolved(payload, tmp_path)

    profile_text = (tmp_path / "profile.sh").read_text()
    assert '  -e NSYS_STATS_PATH="$PROFILE_DIR/stats" \\' in profile_text


def test_render_multi_case_profile_all(tmp_path: Path) -> None:
    payload = _resolved_payload()
    payload["cases"][0]["nsys"] = {"sample": "none"}
    payload["cases"][0]["metadata"]["gap"] = 7
    payload["cases"].append(
        {
            **payload["cases"][0],
            "case_id": "case_two",
        }
    )

    render_resolved(payload, tmp_path)

    assert (tmp_path / "case_one" / "profile.sh").stat().st_mode & 0o111
    assert (tmp_path / "case_two" / "profile.sh").stat().st_mode & 0o111
    assert (tmp_path / "profile_all.sh").stat().st_mode & 0o111
    profile_all = (tmp_path / "profile_all.sh").read_text()
    assert "case_one/profile.sh" in profile_all
    assert "case_two/profile.sh" in profile_all
    assert "case_one/cmd.sh" not in profile_all
    assert "sleep 7" in profile_all


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
    assert '  --artifact_dir "$RUN_DIR/artifacts"' in cmd_text


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
