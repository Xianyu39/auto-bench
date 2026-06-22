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


def _resolved_payload() -> dict:
    return {
        "version": "autobench.resolved/v0.1",
        "cases": [
            {
                "case_id": "case_one",
                "metadata": {"name": "case_one"},
                "trtllm": {
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
                            "token-norm-dist",
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
