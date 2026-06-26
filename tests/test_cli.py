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
    assert "auto-bench 0.1.9" in capsys.readouterr().out


def test_cli_template_stdout(capsys) -> None:
    assert main(["template", "prefill"]) == 0
    output = capsys.readouterr().out
    assert "name: prefill_sweep" in output
    assert "min_mhz: 1400" in output
    assert "tp: 4" in output
    assert "backend: pytorch" in output
    assert "iteration_log: ${runtime.run_dir}/iter.log" in output
    assert "CUDA_VISIBLE_DEVICES" not in output


def test_prefill_template_resolves() -> None:
    result = resolve(get_template("prefill"))
    case = result["cases"][0]
    assert len(result["cases"]) == 6
    assert case["metadata"]["gpu_frequency"]["min_mhz"] == 1400
    assert case["trtllm-bench"]["throughput"]["tp"] == 4
    assert case["trtllm-bench"]["throughput"]["backend"] == "pytorch"
    assert case["trtllm-bench"]["throughput"]["max_num_tokens"] == 1025
    assert case["trtllm-bench"]["throughput"]["iteration_log"] == "$SCRIPT_DIR/iter.log"


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
