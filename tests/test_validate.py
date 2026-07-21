#!/usr/bin/env python3
"""Smoke tests for ROCmKernelWiki: the validator passes and the query tools run.

Run: python3 -m pytest tests/  (or just: python3 tests/test_validate.py)
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
GUIDE_ID = "blog-amdgpu-kernel-opt-guide"
GUIDE_COMMIT = "efa471aeef66a260c85983cc41e833bfa769dade"


def run(*args):
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    return subprocess.run([sys.executable, *args], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8",
                          env=env)


def frontmatter(relpath):
    text = (ROOT / relpath).read_text(encoding="utf-8")
    assert text.startswith("---\n")
    return yaml.safe_load(text.split("---", 2)[1])


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


def test_amdgpu_guide_sync_regression():
    source = frontmatter("sources/blogs/blog-amdgpu-kernel-opt-guide.md")
    assert f"/blob/{GUIDE_COMMIT}/" in source["url"]
    assert str(source["retrieved_at"]) == "2026-07-20"

    consumers = [
        "wiki/hardware/chiplet-xcd.md",
        "wiki/hardware/wavefront.md",
        "wiki/hardware/lds.md",
        "wiki/hardware/cross-lane.md",
        "wiki/hardware/memory-instructions.md",
        "wiki/techniques/buffer-oob-guard.md",
        "wiki/techniques/occupancy-tuning.md",
        "wiki/techniques/profiling-workflow.md",
        "wiki/techniques/vectorized-loads.md",
        "wiki/techniques/vgpr-budgeting.md",
        "wiki/techniques/wave-reduce.md",
    ]
    for relpath in consumers:
        assert GUIDE_ID in frontmatter(relpath).get("sources", []), relpath

    coverage = {
        "wiki/hardware/wavefront.md":
            ("Vulkan", "CUDA", "AMDGPU", "thread-block cluster", "tgsplit"),
        "wiki/hardware/chiplet-xcd.md":
            ("5.3248 TB/s", "8.192 TB/s", "L1D", "MALL", "64 MiB/IOD",
             "2048 sets/channel", "explicitly managed"),
        "wiki/hardware/lds.md":
            ("ds_read2_b64", "ds_read_b128", "ds_write_b128", "0–63",
             "1280-byte", "16 addresses per SIMD", "lowest-thread-id"),
        "wiki/hardware/cross-lane.md":
            ("rocdl.update.dpp", "GCNDPPCombine", "row_mask", "bank_mask",
             "DPP8/DPP16", "unclear provenance"),
        "wiki/hardware/memory-instructions.md":
            ("4 GiB", "descriptor-window limit", "range-checked independently",
             "0x31004000"),
        "wiki/techniques/buffer-oob-guard.md":
            ("dwordx2/x3/x4", "mapped memory", "0x31004000"),
        "wiki/techniques/vectorized-loads.md":
            ("1 KiB", "64 lanes × 16 B"),
    }
    for relpath, markers in coverage.items():
        body = (ROOT / relpath).read_text(encoding="utf-8")
        for marker in markers:
            assert marker in body, f"{relpath}: missing {marker!r}"

    corrected_occupancy_pages = [
        "wiki/techniques/occupancy-tuning.md",
        "wiki/techniques/vgpr-budgeting.md",
        "wiki/patterns/low-occupancy.md",
        "wiki/patterns/vgpr-pressure.md",
    ]
    for relpath in corrected_occupancy_pages:
        body = (ROOT / relpath).read_text(encoding="utf-8")
        for stale in ("40 waves", "10 waves", "min(10", "ceiling 40",
                      "separate AGPR bank", "first to AGPRs",
                      "104 SGPRs/workgroup"):
            assert stale not in body, f"{relpath}: stale occupancy claim {stale!r}"

    vectorized = (ROOT / "wiki/techniques/vectorized-loads.md").read_text(
        encoding="utf-8"
    )
    assert "512 B at once" not in vectorized

    source_body = (ROOT / "sources/blogs/blog-amdgpu-kernel-opt-guide.md").read_text(
        encoding="utf-8"
    )
    for marker in (
        "docs-6.1.1/how-to/llm-fine-tuning-optimization",
        "how-to/rocm-for-ai/inference-optimization",
        "iree-turbine",
        "llvm-ir-attributes",
        "lds-bank-conflict",
        "rdna3-shader-instruction-set-architecture",
        "GCNDPPCombine.cpp",
        "rocprofiler",
    ):
        assert marker in source_body, f"source anchor missing linked resource {marker!r}"
    assert "/en/latest/how-to/llm-fine-tuning-optimization/" not in source_body

    generator = (ROOT / "scripts/generate-indices.py").read_text(encoding="utf-8")
    assert ".as_posix()" in generator
    for query in (ROOT / "queries").glob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", query.read_text(encoding="utf-8")):
            assert "\\" not in target, f"{query.name}: non-portable link {target!r}"

    source_generator = (ROOT / "scripts/gen_source_anchors.py").read_text(
        encoding="utf-8"
    )
    assert "40 waves/CU occupancy (four pools × 10 waves)" not in source_generator
    assert "Keeps the full CDNA3 MFMA/SMFMAC set" not in source_generator
    assert "`v_permlane16_*` cross-lane ops" not in source_generator

    mfma = (ROOT / "wiki/hardware/mfma.md").read_text(encoding="utf-8")
    assert "keeps the full CDNA3 set" not in mfma
    assert "keeps most of the CDNA3 set" in mfma

    phase_consumers = [
        "wiki/techniques/bank-conflict-avoidance.md",
        "wiki/patterns/bank-conflicts.md",
        "wiki/techniques/lds-swizzling.md",
        "wiki/kernels/transpose-lds.md",
    ]
    for relpath in phase_consumers:
        assert "vs-lds-phase-groups-gfx942-gfx950" in frontmatter(relpath).get(
            "version_sensitive", []
        ), relpath
        body = (ROOT / relpath).read_text(encoding="utf-8")
        for stale in ("4 cycles, 16 lanes", "16 lanes per cycle", "dispatched over 4 different"):
            assert stale not in body, f"{relpath}: stale universal phase claim {stale!r}"

    cross_lane = (ROOT / "wiki/hardware/cross-lane.md").read_text(encoding="utf-8")
    wave_reduce = (ROOT / "wiki/techniques/wave-reduce.md").read_text(encoding="utf-8")
    migration = (ROOT / "wiki/migration/gfx942-to-gfx950.md").read_text(
        encoding="utf-8"
    )
    assert "(lane & 16) ? sw[0] : sw[1]" in cross_lane
    assert "(lane & 16) ? p16[0] : p16[1]" in wave_reduce
    assert "(lane & 32) ? p32[0] : p32[1]" in wave_reduce
    assert "(lane & 16) ? sw[0] : sw[1]" in migration
    for body in (cross_lane, wave_reduce, migration):
        assert "r[1] is the partner" not in body
        assert "second element is the swapped partner" not in body
        assert "auto sw = __builtin_amdgcn_permlanex16" not in body

    bandwidth = (ROOT / "wiki/kernels/bandwidth-microbench.md").read_text(
        encoding="utf-8"
    )
    assert "using f32x4 = float __attribute__((ext_vector_type(4)))" in bandwidth
    assert "sizeof(float4)" not in bandwidth
    assert "float4 *in" not in bandwidth
    assert "stream past L2" not in bandwidth

    swizzle = (ROOT / "wiki/techniques/lds-swizzling.md").read_text(
        encoding="utf-8"
    )
    assert "((row >> 2) & 3)" in swizzle
    assert "((row >> 1) & 3)" in swizzle
    assert "!defined(__HIP_DEVICE_COMPILE__)" in swizzle
    assert "__half2_8" not in swizzle
    assert "row & (COLS_V - 1)" not in swizzle

    transpose = (ROOT / "examples/transpose-lds/transpose_lds.cpp").read_text(
        encoding="utf-8"
    )
    assert "#define LDS_PAD 2" in transpose
    assert "#define LDS_PAD 1" in transpose
    assert "tile[TILE][TILE + LDS_PAD]" in transpose

    tf32_pages = [
        "wiki/hardware/mfma.md",
        "wiki/migration/gfx942-to-gfx950.md",
    ]
    for relpath in tf32_pages:
        body = (ROOT / relpath).read_text(encoding="utf-8")
        assert "emulated via BF16" not in body
        assert "will still produce results" not in body

    for relpath in (
        "wiki/patterns/low-occupancy.md",
        "wiki/patterns/vgpr-pressure.md",
    ):
        body = (ROOT / relpath).read_text(encoding="utf-8")
        assert "32 on gfx10+" not in body

    for relpath in ("wiki/hardware/wmma.md", "wiki/migration/wmma-vs-mfma.md"):
        body = (ROOT / relpath).read_text(encoding="utf-8")
        assert "AGPRs (separate bank)" not in body
        assert "separate accumulator register bank" not in body


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
