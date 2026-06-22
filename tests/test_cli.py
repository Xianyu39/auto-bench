from auto_bench.cli import main


def test_cli_smoke() -> None:
    assert main([]) == 0


def test_cli_version(capsys) -> None:
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "auto-bench 0.1.3" in capsys.readouterr().out


def test_cli_template_stdout(capsys) -> None:
    assert main(["template", "prefill"]) == 0
    output = capsys.readouterr().out
    assert "name: prefill_sweep" in output
    assert "iteration_log:" in output


def test_cli_template_output_file(tmp_path) -> None:
    output = tmp_path / "experiment.yaml"
    assert main(["template", "decode", "-o", str(output)]) == 0
    assert output.exists()
    assert "name: decode_sweep" in output.read_text()
