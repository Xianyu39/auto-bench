from auto_bench.cli import main


def test_cli_smoke() -> None:
    assert main([]) == 0
