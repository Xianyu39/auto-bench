from auto_bench.cli import main


def test_cli_smoke() -> None:
    assert main([]) == 0


def test_cli_version(capsys) -> None:
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "auto-bench 0.1.0" in capsys.readouterr().out
