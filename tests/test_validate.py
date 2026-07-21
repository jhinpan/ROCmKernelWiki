#!/usr/bin/env python3
"""Smoke tests for ROCmKernelWiki: the validator passes and the query tools run.

Run: python3 -m pytest tests/  (or just: python3 tests/test_validate.py)
"""
import os
import re
import subprocess
import sys
import tempfile
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

    scripts_dir = str(ROOT / "scripts")
    sys.path.insert(0, scripts_dir)
    configured_cache = os.environ.pop("ROCM_WIKI_CACHE_DIR", None)
    try:
        from query import query_cache_path

        default_cache = query_cache_path()
    finally:
        if configured_cache is not None:
            os.environ["ROCM_WIKI_CACHE_DIR"] = configured_cache
        sys.path.remove(scripts_dir)
    user_cache_root = default_cache.parents[1]
    assert user_cache_root.name.startswith("rocm-kernel-wiki-")
    getuid = getattr(os, "getuid", None)
    if getuid is not None:
        assert user_cache_root.name == f"rocm-kernel-wiki-uid-{getuid()}"


def test_query_recovers_from_invalid_cache():
    import json

    scripts_dir = str(ROOT / "scripts")
    sys.path.insert(0, scripts_dir)
    try:
        import query
    finally:
        sys.path.remove(scripts_dir)

    original_root = query.WIKI_ROOT
    configured_cache = os.environ.get("ROCM_WIKI_CACHE_DIR")
    try:
        with tempfile.TemporaryDirectory() as workspace:
            workspace = Path(workspace)
            wiki_dir = workspace / "wiki"
            wiki_dir.mkdir()
            (wiki_dir / "cache-test.md").write_text(
                "---\nid: cache-test\ntitle: Cache Test\n"
                "page_type: wiki-technique\n---\ncache recovery body\n",
                encoding="utf-8",
            )
            os.environ["ROCM_WIKI_CACHE_DIR"] = str(workspace / "cache")
            query.WIKI_ROOT = workspace
            cache_path = query.query_cache_path()
            expected_pages = query.load_all_pages(use_cache=True)
            expected_cache = json.loads(cache_path.read_text(encoding="utf-8"))
            assert [page["fm"]["id"] for page in expected_pages] == ["cache-test"]

            bad_caches = {
                "malformed JSON": "{broken json",
                "stale signature": json.dumps(
                    {"sig": "stale", "pages": [{"sentinel": True}]}
                ),
            }
            for case, contents in bad_caches.items():
                cache_path.write_text(contents, encoding="utf-8")
                pages = query.load_all_pages(use_cache=True)
                rebuilt = json.loads(cache_path.read_text(encoding="utf-8"))
                assert pages == expected_pages, case
                assert rebuilt["sig"] == expected_cache["sig"], case
                assert rebuilt["pages"] == expected_pages, case
    finally:
        query.WIKI_ROOT = original_root
        if configured_cache is None:
            os.environ.pop("ROCM_WIKI_CACHE_DIR", None)
        else:
            os.environ["ROCM_WIKI_CACHE_DIR"] = configured_cache


def test_codex_skill_contract():
    skill = frontmatter("SKILL.md")
    skill_text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert set(skill) == {"name", "description"}
    assert skill["name"] == "rocm-kernel-wiki"
    assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", skill["name"])
    assert len(skill["description"]) <= 1024
    assert "merged-PR" in skill["description"]
    for repo in (ROOT / "sources/prs").iterdir():
        if repo.is_dir():
            assert repo.name.lower() in skill["description"].lower(), repo.name
    assert len(skill_text.splitlines()) < 500
    assert "~/.claude/skills" not in skill_text
    assert "$HOME/.agents/skills" in readme
    assert ".venv/bin/python" in readme
    assert ".venv\\Scripts\\python.exe" in readme
    assert "same clone-and-venv procedure" in readme
    query_docs = readme.split("## Query Tools", 1)[1].split("## Architecture", 1)[0]
    assert "python3 scripts/" not in query_docs

    metadata = yaml.safe_load(
        (ROOT / "agents/openai.yaml").read_text(encoding="utf-8")
    )
    assert metadata["interface"]["display_name"] == "ROCm Kernel Wiki"
    assert "$rocm-kernel-wiki" in metadata["interface"]["default_prompt"]
    assert metadata["policy"]["allow_implicit_invocation"] is True


def test_documented_corpus_inventory():
    counts = {
        "prs": len(list((ROOT / "sources/prs").glob("*/*.md"))),
        "wiki": len(list((ROOT / "wiki").glob("**/*.md"))),
        "docs_blogs": sum(
            len(list((ROOT / "sources" / subdir).glob("*.md")))
            for subdir in ("docs", "blogs")
        ),
        "refs": len(list((ROOT / "sources/refs").glob("*.md"))),
    }
    assert counts == {"prs": 7454, "wiki": 57, "docs_blogs": 21, "refs": 9}

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    architecture = (ROOT / "docs/architecture.svg").read_text(encoding="utf-8")
    readme_inventory = readme.split("## What's Here", 1)[1].split(
        "## Install as a Codex CLI Skill", 1
    )[0]
    assert "ROCm/hipBLASLt" not in readme_inventory
    for marker in (
        "7,454 PR reference pages",
        "57 synthesized wiki pages",
        "21 doc/blog summaries",
        "9 reference-repository studies",
    ):
        assert marker in readme, marker
    for marker in (
        "7,454 merged-PR references",
        "57 wiki synthesis pages",
        "21 doc/blog summaries",
        "9 reference-repository studies",
    ):
        assert marker in skill, marker
    assert "7,454 merged PRs" in architecture


def test_query_runs_outside_skill_directory():
    env = os.environ.copy()
    env.pop("PYTHONUTF8", None)
    env["PYTHONIOENCODING"] = "cp1252"
    commands = [
        [
            str(ROOT / "scripts/query.py"),
            "avoid LDS bank conflicts on MI300",
            "--limit",
            "3",
        ],
        [str(ROOT / "scripts/get_page.py"), "hw-mfma", "--body-only"],
        [
            str(ROOT / "scripts/grep_wiki.py"),
            "v_mfma_f32_16x16x16",
            "--only",
            "wiki",
        ],
    ]
    with tempfile.TemporaryDirectory() as cwd:
        cache_root = Path(cwd) / "query-cache"
        env["ROCM_WIKI_CACHE_DIR"] = str(cache_root)
        outputs = {}
        for command in commands:
            r = subprocess.run(
                [sys.executable, *command],
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
            )
            assert r.returncode == 0, (
                f"{Path(command[0]).name} failed:\n{r.stdout}\n{r.stderr}"
            )
            assert r.stdout
            outputs[Path(command[0]).name] = r.stdout
        assert "→" in outputs["grep_wiki.py"]
        assert len(list(cache_root.glob("*/query-index.json"))) == 1


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
    for relpath in (
        "wiki/hardware/cross-lane.md",
        "wiki/hardware/lds.md",
        "wiki/techniques/wave-reduce.md",
    ):
        assert "vs-ds-bpermute-address-cdna3-cdna4" in frontmatter(relpath).get(
            "version_sensitive", []
        ), relpath
        body = (ROOT / relpath).read_text(encoding="utf-8")
        assert "out-of-range source yields 0" not in body
        assert "out-of-range source reads 0" not in body
        assert "% 64" in body

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
