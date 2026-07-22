#!/usr/bin/env python3
"""MI355 sandbox adapter for scripts/run_kernel_ab.py."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _wiki_root import WIKI_ROOT, configure_utf8_stdio  # noqa: E402
from evaluate_kernel_ab import load_kernel_tasks  # noqa: E402


def _run_sandbox(
    sandbox: Path,
    workspace: Path,
    output: Path,
    gpu: int,
    command: str,
    timeout: int,
) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(
        [str(sandbox), str(workspace), str(output), str(gpu), command],
        cwd=WIKI_ROOT,
        env={
            key: value
            for key, value in os.environ.items()
            if key not in {"GH_TOKEN", "GITHUB_TOKEN", "SSH_AUTH_SOCK"}
        },
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _hash_files(root: Path) -> tuple[str, dict[str, str]]:
    aggregate = hashlib.sha256()
    hashes = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[relative] = digest
        aggregate.update(f"{relative}:{digest}\n".encode("utf-8"))
    return aggregate.hexdigest(), hashes


def main() -> int:
    try:
        package = json.loads(sys.stdin.read())
        task_id = str((package.get("task") or {}).get("id") or "")
        tasks = load_kernel_tasks(
            WIKI_ROOT / "data" / "evals" / "kernel-tasks.yaml"
        )
        if task_id not in tasks:
            raise ValueError(f"unknown trusted task {task_id!r}")
        task = tasks[task_id]
        workspace = Path(str(package.get("workspace") or "")).resolve()
        if not (workspace / ".git").exists() and not (
            workspace / ".git"
        ).is_file():
            raise ValueError("workspace is not a git worktree")
        arm = str(package.get("mode") or "")
        if arm not in {"without_skill", "with_skill"}:
            raise ValueError("invalid A/B arm")
        gpu = int(os.environ.get("ROCM_WIKI_MI355_GPU", "0"))
        timeout = int(os.environ.get("ROCM_WIKI_KERNEL_TASK_TIMEOUT", "1800"))
        sandbox = WIKI_ROOT / "ops" / "mi355" / "run-command-sandbox.sh"
        evidence_root = Path(
            os.environ.get(
                "ROCM_WIKI_KERNEL_EVIDENCE_ROOT",
                str(WIKI_ROOT / ".evidence" / "kernel-ab"),
            )
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = evidence_root / f"{task_id}-{arm}-{timestamp}"
        output.mkdir(parents=True, exist_ok=False)

        commands = []
        environment = _run_sandbox(
            sandbox, workspace, output, gpu, "hipcc --version", timeout
        )
        commands.append(environment)
        if environment["returncode"] != 0:
            raise RuntimeError("sandbox toolchain probe failed")
        build_command = task.get("build_command")
        if build_command:
            build = _run_sandbox(
                sandbox, workspace, output, gpu, str(build_command), timeout
            )
            commands.append(build)
            if build["returncode"] != 0:
                correct = False
                correctness = build
            else:
                correctness = _run_sandbox(
                    sandbox,
                    workspace,
                    output,
                    gpu,
                    str(task["correctness_command"]),
                    timeout,
                )
                commands.append(correctness)
                correct = correctness["returncode"] == 0
        else:
            correctness = _run_sandbox(
                sandbox,
                workspace,
                output,
                gpu,
                str(task["correctness_command"]),
                timeout,
            )
            commands.append(correctness)
            correct = correctness["returncode"] == 0

        metric = None
        metric_spec = task["metric"]
        benchmark = correctness
        if correct and task.get("benchmark_command"):
            benchmark = _run_sandbox(
                sandbox,
                workspace,
                output,
                gpu,
                str(task["benchmark_command"]),
                timeout,
            )
            commands.append(benchmark)
            correct = benchmark["returncode"] == 0
        if correct and metric_spec.get("source") == "wall-clock":
            metric = float(benchmark["duration_seconds"])
        elif correct:
            match = re.search(
                str(metric_spec["regex"]),
                benchmark["stdout"],
                re.IGNORECASE | re.DOTALL,
            )
            if not match:
                raise ValueError(f"{task_id}: metric regex did not match")
            metric = float(match.group(1)) * float(metric_spec.get("scale", 1.0))

        for index, command in enumerate(commands, 1):
            (output / f"{index:02d}.stdout.txt").write_text(
                command.pop("stdout"), encoding="utf-8"
            )
            (output / f"{index:02d}.stderr.txt").write_text(
                command.pop("stderr"), encoding="utf-8"
            )
        (output / "commands.json").write_text(
            json.dumps(commands, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Use the persisted toolchain probe after command normalization.
        environment_fingerprint = hashlib.sha256(
            (
                os.environ["ROCM_WIKI_MI355_IMAGE"]
                + "\n"
                + (output / "01.stdout.txt").read_text(encoding="utf-8")
                + f"\ngpu={gpu}"
            ).encode("utf-8")
        ).hexdigest()
        digest, hashes = _hash_files(output)
        evidence = {
            "schema_version": 1,
            "task": task_id,
            "arm": arm,
            "environment_fingerprint": environment_fingerprint,
            "bundle_sha256": digest,
            "files": hashes,
            "correct": correct,
            "metric": metric,
        }
        (output / "EVIDENCE.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "correct": correct,
                    "metric": metric,
                    "evidence_bundle": str(output),
                    "environment_fingerprint": environment_fingerprint,
                }
            )
        )
    except (
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    configure_utf8_stdio()
    raise SystemExit(main())
