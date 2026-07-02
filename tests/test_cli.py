import subprocess
import tomllib
from pathlib import Path

from auto_bench.cli import main
from auto_bench.resolver import resolve
from auto_bench.templates import get_template


def test_cli_smoke() -> None:
    assert main([]) == 0


def test_cli_version(capsys) -> None:
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "auto-bench 0.1.13" in capsys.readouterr().out


def test_cli_has_ab_entrypoint_alias() -> None:
    with open("pyproject.toml", "rb") as handle:
        pyproject = tomllib.load(handle)

    assert pyproject["project"]["scripts"]["auto-bench"] == "auto_bench.cli:main"
    assert pyproject["project"]["scripts"]["ab"] == "auto_bench.cli:main"


def test_cli_template_stdout(capsys) -> None:
    assert main(["template", "prefill"]) == 0
    output = capsys.readouterr().out
    assert "name: prefill_sweep" in output
    assert "min_mhz: 1400" in output
    assert "tp: 4" in output
    assert "backend: pytorch" in output
    assert "NSYS_STATS_PATH: ${runtime.run_dir}/stats" in output
    assert "CUDA_VISIBLE_DEVICES: 0" in output
    assert "TLLM_PROFILE_START_STOP: 10-20" in output
    assert "cuda-graph-trace: node" in output
    assert "iteration_log: ${runtime.run_dir}/iter.log" in output


def test_prefill_template_resolves() -> None:
    result = resolve(get_template("prefill"))
    case = result["cases"][0]
    assert len(result["cases"]) == 6
    assert case["metadata"]["gpu_frequency"]["min_mhz"] == 1400
    assert case["trtllm-bench"]["throughput"]["tp"] == 4
    assert case["trtllm-bench"]["throughput"]["backend"] == "pytorch"
    assert case["trtllm-bench"]["throughput"]["max_num_tokens"] == 1025
    assert case["trtllm-bench"]["throughput"]["iteration_log"] == "$SCRIPT_DIR/iter.log"
    assert case["nsys"]["env"]["NSYS_STATS_PATH"] == "$SCRIPT_DIR/stats"
    assert case["nsys"]["env"]["CUDA_VISIBLE_DEVICES"] == 0
    assert case["nsys"]["env"]["TLLM_PROFILE_START_STOP"] == "10-20"
    assert case["nsys"]["cuda-graph-trace"] == "node"


def test_cli_template_output_file(tmp_path) -> None:
    output = tmp_path / "experiment.yaml"
    assert main(["template", "decode", "-o", str(output)]) == 0
    assert output.exists()
    assert "name: decode_sweep" in output.read_text()


def test_cli_render_continue_on_error(tmp_path) -> None:
    experiment = tmp_path / "experiment.yaml"
    output_dir = tmp_path / "artifacts"
    experiment.write_text(
        """
metadata:
  name: render_continue
vars:
  batch_size:
    sweep: [1, 2]
trtllm-bench:
  model: llama
  throughput:
    dataset: /datasets/static.txt
    batch_size: ${vars.batch_size}
""".lstrip(),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "render",
                str(experiment),
                "-o",
                str(output_dir),
                "--continue-on-error",
            ]
        )
        == 0
    )
    assert "FAILED=0" in (output_dir / "run_all.sh").read_text()


def test_cli_plan_alias_renders_artifacts(tmp_path) -> None:
    experiment = tmp_path / "experiment.yaml"
    output_dir = tmp_path / "artifacts"
    experiment.write_text(
        """
metadata:
  name: plan_alias
trtllm-bench:
  model: llama
  throughput:
    dataset: /datasets/static.txt
""".lstrip(),
        encoding="utf-8",
    )

    assert main(["plan", str(experiment), "-o", str(output_dir)]) == 0

    assert (output_dir / "cmd.sh").exists()
    assert (output_dir / "resolved.yaml").exists()


def test_cli_run_renders_and_executes_cmd(tmp_path, monkeypatch) -> None:
    experiment = tmp_path / "experiment.yaml"
    output_dir = tmp_path / "artifacts"
    experiment.write_text(
        """
metadata:
  name: run_cmd
trtllm-bench:
  model: llama
  throughput:
    dataset: /datasets/static.txt
""".lstrip(),
        encoding="utf-8",
    )
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(argv, *, check, env):
        calls.append((argv, env))
        assert check is False
        assert env["AUTO_BENCH_QUIET"] == "1"
        (output_dir / "run.log").write_text("ok\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert main(["run", str(experiment), "-o", str(output_dir), "--no-progress"]) == 0

    assert [call[0] for call in calls] == [[str(output_dir / "cmd.sh")]]
    assert (output_dir / "cmd.sh").exists()


def test_cli_run_profile_executes_profile_script(tmp_path, monkeypatch) -> None:
    experiment = tmp_path / "experiment.yaml"
    output_dir = tmp_path / "artifacts"
    experiment.write_text(
        """
metadata:
  name: run_profile
nsys:
  sample: none
trtllm-bench:
  model: llama
  throughput:
    dataset: /datasets/static.txt
""".lstrip(),
        encoding="utf-8",
    )
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(argv, *, check, env):
        calls.append((argv, env))
        assert check is False
        assert env["AUTO_BENCH_QUIET"] == "1"
        profile_dir = output_dir / "profile"
        profile_dir.mkdir()
        (profile_dir / "profile.log").write_text("failed\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 7)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert (
        main(
            [
                "run",
                str(experiment),
                "-o",
                str(output_dir),
                "--profile",
                "--no-progress",
            ]
        )
        == 7
    )

    assert [call[0] for call in calls] == [[str(output_dir / "profile.sh")]]
    assert (output_dir / "profile.sh").exists()


def test_cli_run_multi_case_writes_total_log_and_stops_on_failure(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    experiment = tmp_path / "experiment.yaml"
    output_dir = tmp_path / "artifacts"
    experiment.write_text(
        """
metadata:
  name: run_multi
vars:
  batch_size:
    sweep: [1, 2]
trtllm-bench:
  model: llama
  throughput:
    dataset: /datasets/static.txt
    batch_size: ${vars.batch_size}
""".lstrip(),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(argv, *, check, env):
        calls.append(argv)
        assert check is False
        assert env["AUTO_BENCH_QUIET"] == "1"
        case_dir = Path(argv[0]).parent
        (case_dir / "run.log").write_text("line one\nline two\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 3)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert (
        main(["run", str(experiment), "-o", str(output_dir), "--no-progress"])
        == 3
    )

    assert len(calls) == 1
    total_log = (output_dir / "run.log").read_text(encoding="utf-8")
    assert "===== auto-bench case run_multi__vars.batch_size=1 start" in total_log
    assert "line one\nline two" in total_log
    assert "exit=3" in total_log
    captured = capsys.readouterr()
    assert "auto-bench: case failed: run_multi__vars.batch_size=1 (exit 3)" in (
        captured.err
    )
    assert "auto-bench: last log lines:" in captured.err
    assert "line two" in captured.err


def test_cli_run_continue_on_error_runs_remaining_cases(
    tmp_path: Path, monkeypatch
) -> None:
    experiment = tmp_path / "experiment.yaml"
    output_dir = tmp_path / "artifacts"
    experiment.write_text(
        """
metadata:
  name: run_continue
vars:
  batch_size:
    sweep: [1, 2]
trtllm-bench:
  model: llama
  throughput:
    dataset: /datasets/static.txt
    batch_size: ${vars.batch_size}
""".lstrip(),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(argv, *, check, env):
        calls.append(argv)
        assert check is False
        assert env["AUTO_BENCH_QUIET"] == "1"
        case_dir = Path(argv[0]).parent
        (case_dir / "run.log").write_text(
            f"log for {case_dir.name}\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(argv, 5 if len(calls) == 1 else 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert (
        main(
            [
                "run",
                str(experiment),
                "-o",
                str(output_dir),
                "--continue-on-error",
                "--no-progress",
            ]
        )
        == 5
    )

    assert len(calls) == 2
    total_log = (output_dir / "run.log").read_text(encoding="utf-8")
    assert "run_continue__vars.batch_size=1" in total_log
    assert "run_continue__vars.batch_size=2" in total_log


def test_cli_resolve_emits_warnings_to_stderr(tmp_path, capsys) -> None:
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        """
metadata:
  name: warn
trtllm-bench:
  model: llama
  custom_global: value
  throughput:
    dataset: /datasets/static.txt
""".lstrip(),
        encoding="utf-8",
    )

    assert main(["resolve", str(experiment)]) == 0
    captured = capsys.readouterr()
    assert "warnings:" in captured.out
    assert "Warning: option 'trtllm-bench.custom_global'" in captured.err
