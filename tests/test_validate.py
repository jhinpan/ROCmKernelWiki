#!/usr/bin/env python3
"""Smoke tests for ROCmKernelWiki: the validator passes and the query tools run.

Run: python3 -m pytest tests/  (or just: python3 tests/test_validate.py)
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT,
                          capture_output=True, text=True)


def test_validator_passes():
    r = run("scripts/validate.py")
    assert r.returncode == 0, f"validate.py failed:\n{r.stdout}\n{r.stderr}"
    assert "0 errors" in r.stdout


def test_query_runs():
    r = run("scripts/query.py", "--tag", "mfma", "--type", "kernel", "--compact")
    assert r.returncode == 0
    assert "result" in r.stdout


def test_get_page_by_id():
    r = run("scripts/get_page.py", "hw-mfma", "--frontmatter-only")
    assert r.returncode == 0
    assert "hw-mfma" in r.stdout


def test_alias_architecture():
    # MI355X must alias to gfx950
    r = run("scripts/query.py", "--architecture", "MI355X", "--type", "hardware",
            "--compact")
    assert r.returncode == 0
    assert "result" in r.stdout


def test_rerank_surfaces_synthesis_first():
    # cp.async (alias) must surface the migration synthesis page, not PR noise
    r = run("scripts/query.py", "how port cuda cp.async to rocm", "--limit", "3",
            "--compact")
    assert r.returncode == 0
    assert "migration-cuda-to-hip" in r.stdout


def test_synthesis_flag_excludes_prs():
    r = run("scripts/query.py", "fp8 gemm", "--synthesis", "--limit", "8", "--compact")
    assert r.returncode == 0
    assert "source-pr" not in r.stdout


def test_pr_wiki_bridge():
    # a kernel page should list implementing PRs
    r = run("scripts/get_page.py", "kernel-fp8-gemm")
    assert r.returncode == 0
    assert "Implementing PRs" in r.stdout


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
