#!/usr/bin/env python3
"""Trusted MI355 evidence worker for exact, maintainer-approved PR SHAs.

This is a node-local controller, not a GitHub Actions runner. Controller and
sandbox code always come from the trusted checkout. Candidate code is mounted
read-only, receives no credentials, and is run only after an exact-SHA approval
from a collaborator with write/maintain/admin permission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _wiki_root import WIKI_ROOT, configure_utf8_stdio  # noqa: E402

APPROVAL_RE = re.compile(r"(?m)^\s*/mi355-approve\s+([0-9a-f]{40})\s*$")
WRITE_PERMISSIONS = {"admin", "maintain", "write"}
CONTROL_PLANE_PATHS = {
    "validation/run.py",
    "scripts/evolve/mi355_worker.py",
}
CONTROL_PLANE_PREFIXES = (
    ".github/",
    "ops/mi355/",
)
CHECK_NAME = "mi355-evidence"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def find_approval(
    comments: list[dict[str, Any]],
    *,
    head_sha: str,
    permission_lookup: Callable[[str], str],
) -> dict[str, str]:
    for comment in sorted(
        comments, key=lambda item: str(item.get("created_at") or ""), reverse=True
    ):
        match = APPROVAL_RE.search(str(comment.get("body") or ""))
        if not match or match.group(1) != head_sha:
            continue
        login = str((comment.get("user") or {}).get("login") or "")
        if not login:
            continue
        permission = permission_lookup(login)
        if permission in WRITE_PERMISSIONS:
            return {
                "login": login,
                "permission": permission,
                "sha": head_sha,
                "approved_at": str(comment.get("created_at") or ""),
            }
    raise ValueError(
        "no exact-SHA /mi355-approve comment from a write-authorized maintainer"
    )


def validate_changed_paths(paths: list[str]) -> None:
    for path in paths:
        normalized = Path(path).as_posix().lstrip("/")
        if normalized in CONTROL_PLANE_PATHS or normalized.startswith(
            CONTROL_PLANE_PREFIXES
        ):
            raise ValueError(
                f"candidate changes MI355 control-plane path: {normalized}"
            )


def _run(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
    input_text: str | None = None,
    timeout: int = 300,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)} failed ({result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def _gh_json(arguments: list[str], *, cwd: Path = WIKI_ROOT) -> Any:
    output = _run(["gh", *arguments], cwd=cwd)
    return json.loads(output or "null")


def _repo_name() -> str:
    payload = _gh_json(["repo", "view", "--json", "nameWithOwner"])
    return str(payload["nameWithOwner"])


def _permission(repo: str, login: str) -> str:
    payload = _gh_json(["api", f"repos/{repo}/collaborators/{login}/permission"])
    return str(payload.get("permission") or "none")


def _comments(repo: str, pr: int) -> list[dict[str, Any]]:
    pages = _gh_json(
        [
            "api",
            f"repos/{repo}/issues/{pr}/comments",
            "--paginate",
            "--slurp",
        ]
    )
    return [
        comment
        for page in pages or []
        for comment in (page if isinstance(page, list) else [])
        if isinstance(comment, dict)
    ]


def _pull(repo: str, pr: int) -> dict[str, Any]:
    return _gh_json(
        [
            "pr",
            "view",
            str(pr),
            "--repo",
            repo,
            "--json",
            (
                "number,title,url,headRefOid,headRefName,headRepositoryOwner,"
                "headRepository,baseRefName,isCrossRepository,labels"
            ),
        ]
    )


def _changed_paths(repo: str, pr: int) -> list[str]:
    output = _run(
        ["gh", "pr", "diff", str(pr), "--repo", repo, "--name-only"],
        cwd=WIKI_ROOT,
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def _health_snapshot(command: str | None, gpu: int, destination: Path) -> None:
    if not command:
        destination.write_text("health command not configured\n", encoding="utf-8")
        return
    argv = [part.format(gpu=gpu) for part in shlex.split(command)]
    result = subprocess.run(
        argv,
        cwd=WIKI_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    destination.write_text(
        f"returncode={result.returncode}\n{result.stdout}\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError("MI355 health check failed")


def _bundle_digest(bundle: Path) -> tuple[str, dict[str, str]]:
    hashes = {}
    aggregate = hashlib.sha256()
    for path in sorted(bundle.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(bundle).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[relative] = digest
        aggregate.update(f"{relative}:{digest}\n".encode("utf-8"))
    return aggregate.hexdigest(), hashes


def _copy_compact_bundle(
    run_output: Path,
    evidence_root: Path,
    *,
    pr: int,
    head_sha: str,
    controller_sha: str,
    approval: dict[str, str],
    artifact_uri: str | None,
) -> tuple[Path, dict[str, Any]]:
    destination = evidence_root / f"pr-{pr}-{head_sha[:12]}"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for name in ("manifest.json", "verdicts.json", "summary.txt"):
        source = run_output / name
        if not source.is_file():
            raise RuntimeError(f"validation output is missing {name}")
        shutil.copy2(source, destination / name)
    for name in ("health-before.txt", "health-after.txt"):
        source = run_output / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    digest, hashes = _bundle_digest(destination)
    evidence = {
        "schema_version": 1,
        "created_at": utc_now(),
        "pr": pr,
        "head_sha": head_sha,
        "controller_sha": controller_sha,
        "approval": approval,
        "artifact_uri": artifact_uri,
        "bundle_sha256": digest,
        "files": hashes,
    }
    (destination / "EVIDENCE.yaml").write_text(
        yaml.safe_dump(evidence, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination, evidence


def _publish_check(
    repo: str,
    *,
    head_sha: str,
    evidence: dict[str, Any],
    verdicts: dict[str, Any],
    approval: dict[str, str],
    dry_run: bool,
) -> dict[str, Any]:
    overall = str(verdicts.get("overall_status") or "fail")
    conclusion = "success" if overall == "pass" else "failure"
    counts = verdicts.get("counts") or {}
    summary = (
        f"Exact SHA `{head_sha}` approved by @{approval['login']} "
        f"({approval['permission']}).\n\n"
        f"Verdicts: {counts.get('pass', 0)} pass, {counts.get('fail', 0)} fail, "
        f"{counts.get('recorded', 0)} source-recorded.\n\n"
        f"Evidence digest: `{evidence['bundle_sha256']}`\n\n"
        f"Artifact: {evidence.get('artifact_uri') or 'node-local (URI not configured)'}"
    )
    payload = {
        "name": CHECK_NAME,
        "head_sha": head_sha,
        "status": "completed",
        "conclusion": conclusion,
        "completed_at": utc_now(),
        "output": {
            "title": f"MI355 evidence: {overall}",
            "summary": summary,
        },
    }
    if dry_run:
        return payload
    return json.loads(
        _run(
            [
                "gh",
                "api",
                "--method",
                "POST",
                f"repos/{repo}/check-runs",
                "--input",
                "-",
            ],
            cwd=WIKI_ROOT,
            input_text=json.dumps(payload),
        )
    )


def process_pr(args: argparse.Namespace, repo: str, pr_number: int) -> dict[str, Any]:
    pull = _pull(repo, pr_number)
    labels = {
        str(label.get("name"))
        for label in (pull.get("labels") or [])
        if isinstance(label, dict)
    }
    if args.required_label not in labels:
        raise ValueError(f"PR lacks required label {args.required_label!r}")
    head_sha = str(pull["headRefOid"])
    approval = find_approval(
        _comments(repo, pr_number),
        head_sha=head_sha,
        permission_lookup=lambda login: _permission(repo, login),
    )
    changed_paths = _changed_paths(repo, pr_number)
    validate_changed_paths(changed_paths)
    if args.dry_run:
        return {
            "status": "approved-dry-run",
            "pr": pr_number,
            "head_sha": head_sha,
            "approval": approval,
            "changed_paths": changed_paths,
        }

    controller_sha = _run(["git", "rev-parse", "HEAD"], cwd=WIKI_ROOT)
    evidence_root = args.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rocm-wiki-mi355-") as directory:
        temporary = Path(directory)
        candidate = temporary / "candidate"
        run_output = temporary / "run-output"
        fetch_ref = f"refs/rocm-wiki/mi355/{head_sha}"
        _run(
            [
                "git",
                "fetch",
                "origin",
                f"pull/{pr_number}/head:{fetch_ref}",
            ],
            cwd=WIKI_ROOT,
        )
        _run(
            ["git", "worktree", "add", "--detach", str(candidate), fetch_ref],
            cwd=WIKI_ROOT,
        )
        try:
            run_output.mkdir()
            _health_snapshot(
                args.health_command,
                args.gpu,
                run_output / "health-before.txt",
            )
            sandbox = args.sandbox.resolve()
            environment = {
                key: value
                for key, value in os.environ.items()
                if key not in {"GH_TOKEN", "GITHUB_TOKEN", "SSH_AUTH_SOCK"}
            }
            _run(
                [
                    str(sandbox),
                    str(WIKI_ROOT / "validation" / "run.py"),
                    str(candidate / "validation"),
                    str(candidate),
                    str(run_output),
                    str(args.gpu),
                ],
                cwd=WIKI_ROOT,
                environment=environment,
                timeout=args.timeout,
            )
            _health_snapshot(
                args.health_command,
                args.gpu,
                run_output / "health-after.txt",
            )
        finally:
            _run(
                ["git", "worktree", "remove", "--force", str(candidate)],
                cwd=WIKI_ROOT,
            )
        validation_output = run_output / "validation"
        for health_name in ("health-before.txt", "health-after.txt"):
            shutil.copy2(run_output / health_name, validation_output / health_name)
        verdicts = json.loads(
            (validation_output / "verdicts.json").read_text(encoding="utf-8")
        )
        bundle, evidence = _copy_compact_bundle(
            validation_output,
            evidence_root,
            pr=pr_number,
            head_sha=head_sha,
            controller_sha=controller_sha,
            approval=approval,
            artifact_uri=(
                args.artifact_uri.rstrip("/") + "/" + f"pr-{pr_number}-{head_sha[:12]}"
                if args.artifact_uri
                else None
            ),
        )
        check = _publish_check(
            repo,
            head_sha=head_sha,
            evidence=evidence,
            verdicts=verdicts,
            approval=approval,
            dry_run=False,
        )
        return {
            "status": verdicts.get("overall_status"),
            "pr": pr_number,
            "head_sha": head_sha,
            "bundle": str(bundle),
            "check_url": check.get("html_url"),
        }


def _labeled_prs(repo: str, label: str) -> list[int]:
    pulls = _gh_json(
        [
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--label",
            label,
            "--json",
            "number",
        ]
    )
    return [int(pull["number"]) for pull in pulls or []]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo")
    parser.add_argument("--pr", type=int)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--required-label", default="mi355-approved")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--sandbox",
        type=Path,
        default=WIKI_ROOT / "ops" / "mi355" / "run-sandbox.sh",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=WIKI_ROOT / ".evidence" / "mi355",
    )
    parser.add_argument(
        "--artifact-uri",
        default=os.environ.get("ROCM_WIKI_MI355_ARTIFACT_URI"),
    )
    parser.add_argument(
        "--health-command",
        default=os.environ.get("ROCM_WIKI_MI355_HEALTH_COMMAND"),
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        repo = args.repo or _repo_name()
        prs = [args.pr] if args.pr else _labeled_prs(repo, args.required_label)
        if not prs:
            print(json.dumps({"status": "no-op", "reason": "no approved PRs"}))
            return 0
        results = [process_pr(args, repo, pr) for pr in prs[:1] if pr is not None]
        print(json.dumps({"results": results}, indent=2, sort_keys=True))
    except (
        OSError,
        KeyError,
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
