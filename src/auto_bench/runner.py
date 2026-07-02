from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from auto_bench.errors import AutobenchError

LOG_TAIL_LINES = 40


@dataclass(frozen=True)
class RunCase:
    case_id: str
    case: dict[str, Any]
    case_dir: Path
    script: Path
    log_path: Path


@dataclass(frozen=True)
class CaseResult:
    case: RunCase
    returncode: int
    started_at: datetime
    ended_at: datetime
    duration: float


class Reporter(Protocol):
    def start_case(self, run_case: RunCase, index: int, total: int) -> None: ...

    def finish_case(self, result: CaseResult) -> None: ...

    def wait_gap(self, seconds: int | float, after_case_id: str) -> None: ...

    def error(self, result: CaseResult, tail: list[str]) -> None: ...


def run_cases(
    resolved: dict[str, Any],
    output_dir: str | Path,
    case_dirs: list[Path],
    *,
    profile: bool,
    continue_on_error: bool,
    progress: bool,
) -> int:
    root = Path(output_dir)
    cases = _run_cases(resolved, case_dirs, profile=profile)
    if not cases:
        script = root / "profile_all.sh"
        raise AutobenchError(
            f"{script}: missing profile script; add top-level nsys config first"
        )

    total_log = _total_log_path(root, cases, profile=profile)
    if total_log is not None:
        total_log.write_text("", encoding="utf-8")

    reporter = RichReporter() if progress else PlainReporter()
    failed = 0
    total = len(cases)

    with reporter_context(reporter):
        for index, run_case in enumerate(cases, start=1):
            reporter.start_case(run_case, index, total)
            started_at = datetime.now().astimezone()
            started = time.monotonic()
            completed = subprocess.run(
                [str(run_case.script)],
                check=False,
                env=_quiet_env(),
            )
            ended_at = datetime.now().astimezone()
            result = CaseResult(
                case=run_case,
                returncode=int(completed.returncode),
                started_at=started_at,
                ended_at=ended_at,
                duration=time.monotonic() - started,
            )
            _append_total_log(total_log, result)
            reporter.finish_case(result)

            if result.returncode != 0:
                if failed == 0:
                    failed = result.returncode
                reporter.error(result, _tail_lines(result.case.log_path))
                if not continue_on_error:
                    return failed

            if index < total:
                gap = _metadata_gap(run_case.case)
                if gap > 0:
                    reporter.wait_gap(gap, run_case.case_id)
                    time.sleep(gap)

    return failed


@contextmanager
def reporter_context(reporter: Reporter) -> Iterator[None]:
    if isinstance(reporter, RichReporter):
        with reporter:
            yield
        return
    yield


class PlainReporter:
    def start_case(self, run_case: RunCase, index: int, total: int) -> None:
        sys.stderr.write(f"auto-bench: [{index}/{total}] start {run_case.case_id}\n")

    def finish_case(self, result: CaseResult) -> None:
        status = "ok" if result.returncode == 0 else f"failed ({result.returncode})"
        sys.stderr.write(
            f"auto-bench: {status}: {result.case.case_id} "
            f"in {_format_duration(result.duration)}\n"
        )

    def wait_gap(self, seconds: int | float, after_case_id: str) -> None:
        sys.stderr.write(
            f"auto-bench: waiting {_format_duration(float(seconds))} "
            f"after {after_case_id}\n"
        )

    def error(self, result: CaseResult, tail: list[str]) -> None:
        _write_error(sys.stderr, result, tail)


class RichReporter:
    def __init__(self) -> None:
        self.console = Console(stderr=True)
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        self.task_id: TaskID | None = None

    def __enter__(self) -> RichReporter:
        self.progress.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.progress.__exit__(exc_type, exc, traceback)

    def start_case(self, run_case: RunCase, index: int, total: int) -> None:
        description = f"[{index}/{total}] running {run_case.case_id}"
        if self.task_id is None:
            self.task_id = self.progress.add_task(description, total=total)
        else:
            self.progress.update(self.task_id, description=description)

    def finish_case(self, result: CaseResult) -> None:
        if self.task_id is None:
            return
        status = "done" if result.returncode == 0 else "failed"
        self.progress.update(
            self.task_id,
            advance=1,
            description=(
                f"{status} {result.case.case_id} "
                f"({_format_duration(result.duration)})"
            ),
        )

    def wait_gap(self, seconds: int | float, after_case_id: str) -> None:
        if self.task_id is not None:
            self.progress.update(
                self.task_id,
                description=(
                    f"waiting {_format_duration(float(seconds))} after {after_case_id}"
                ),
            )

    def error(self, result: CaseResult, tail: list[str]) -> None:
        _write_error(self.console.file, result, tail)


def _run_cases(
    resolved: dict[str, Any],
    case_dirs: list[Path],
    *,
    profile: bool,
) -> list[RunCase]:
    cases = resolved.get("cases")
    if not isinstance(cases, list):
        raise TypeError("resolved payload must contain a cases list")

    run_cases: list[RunCase] = []
    for case, case_dir in zip(cases, case_dirs, strict=True):
        if not isinstance(case, dict):
            raise TypeError("case must be a mapping")
        case_id = str(case["case_id"])
        script = case_dir / ("profile.sh" if profile else "cmd.sh")
        if profile and not script.exists():
            continue
        if not script.exists():
            raise AutobenchError(f"{script}: missing run script")
        run_cases.append(
            RunCase(
                case_id=case_id,
                case=case,
                case_dir=case_dir,
                script=script,
                log_path=_case_log_path(case_dir, profile=profile),
            )
        )
    return run_cases


def _case_log_path(case_dir: Path, *, profile: bool) -> Path:
    if profile:
        return case_dir / "profile" / "profile.log"
    return case_dir / "run.log"


def _total_log_path(root: Path, cases: list[RunCase], *, profile: bool) -> Path | None:
    if len(cases) <= 1:
        return None
    return root / ("profile_all.log" if profile else "run.log")


def _quiet_env() -> dict[str, str]:
    env = os.environ.copy()
    env["AUTO_BENCH_QUIET"] = "1"
    return env


def _append_total_log(total_log: Path | None, result: CaseResult) -> None:
    if total_log is None:
        return
    with total_log.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n"
            f"===== auto-bench case {result.case.case_id} start "
            f"{result.started_at.isoformat()} =====\n"
        )
        if result.case.log_path.exists():
            handle.write(result.case.log_path.read_text(encoding="utf-8"))
            if result.case.log_path.stat().st_size > 0:
                handle.write("\n")
        else:
            handle.write(f"auto-bench: missing log: {result.case.log_path}\n")
        handle.write(
            f"===== auto-bench case {result.case.case_id} end "
            f"{result.ended_at.isoformat()} "
            f"exit={result.returncode} "
            f"duration={_format_duration(result.duration)} =====\n"
        )


def _metadata_gap(case: dict[str, Any]) -> int | float:
    metadata = case.get("metadata", {})
    if not isinstance(metadata, dict):
        return 0
    gap = metadata.get("gap", 0)
    if isinstance(gap, int | float) and gap > 0:
        return gap
    return 0


def _tail_lines(path: Path, limit: int = LOG_TAIL_LINES) -> list[str]:
    if not path.exists():
        return [f"auto-bench: missing log: {path}"]
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-limit:]


def _write_error(stream: Any, result: CaseResult, tail: list[str]) -> None:
    stream.write(
        f"auto-bench: case failed: {result.case.case_id} "
        f"(exit {result.returncode})\n"
    )
    stream.write(f"auto-bench: log: {result.case.log_path}\n")
    if tail:
        stream.write("auto-bench: last log lines:\n")
        for line in tail:
            stream.write(f"{line}\n")


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{remainder:02d}s"
    return f"{minutes:d}m{remainder:02d}s"
