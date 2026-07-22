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
    assert counts == {"prs": 7454, "wiki": 57, "docs_blogs": 21, "refs": 10}

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    architecture = (ROOT / "docs/architecture.svg").read_text(encoding="utf-8")
    readme_inventory = readme.split("## What's Here", 1)[1].split(
        "## Install as a Codex CLI Skill", 1
    )[0]
    assert "ROCm/hipBLASLt" not in readme_inventory
    for marker in (
        "7,454 PR reference pages",
        "54 active synthesized wiki pages",
        "21 doc/blog summaries",
        "10 reference-repository studies",
    ):
        assert marker in readme, marker
    for marker in (
        "7,454 merged-PR references",
        "54 synthesis pages",
        "21 doc/blog summaries",
        "10 reference-repository studies",
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


def test_get_page_scope_and_index():
    blocked = run("scripts/get_page.py", "hw-wmma", "--frontmatter-only")
    recovery = run(
        "scripts/get_page.py",
        "hw-wmma",
        "--frontmatter-only",
        "--include-out-of-scope",
    )
    assert blocked.returncode == 1
    assert "outside active" in blocked.stderr
    assert recovery.returncode == 0
    assert "hw-wmma" in recovery.stdout

    scripts_dir = str(ROOT / "scripts")
    sys.path.insert(0, scripts_dir)
    try:
        from _index import id_index

        assert len(id_index()) == 7542
    finally:
        sys.path.remove(scripts_dir)



def test_id_and_scope_caches_invalidate_in_process():
    scripts_dir = str(ROOT / "scripts")
    sys.path.insert(0, scripts_dir)
    try:
        import _index
        import _scope

        original_index_root = _index.WIKI_ROOT
        original_scope_root = _scope.WIKI_ROOT
        configured_cache = os.environ.get("ROCM_WIKI_CACHE_DIR")
        with tempfile.TemporaryDirectory() as workspace:
            workspace = Path(workspace)
            (workspace / "wiki").mkdir()
            (workspace / "sources").mkdir()
            (workspace / "data").mkdir()
            os.environ["ROCM_WIKI_CACHE_DIR"] = str(workspace / "cache")
            _index.WIKI_ROOT = workspace
            _index._MEMORY_CACHE = None
            _scope.WIKI_ROOT = workspace
            _scope._SCOPE_CACHE = None

            (workspace / "wiki/a.md").write_text(
                "\ufeff---\nid: page-a\n---\nA\n", encoding="utf-8"
            )
            assert set(_index.id_index()) == {"page-a"}
            (workspace / "wiki/b.md").write_text(
                "---\nid: page-b\n---\nB\n", encoding="utf-8"
            )
            assert set(_index.id_index(refresh=True)) == {"page-a", "page-b"}
            assert set(_index.id_index(use_cache=False)) == {"page-a", "page-b"}

            scope_path = workspace / "data/scope.yaml"
            scope_path.write_text(
                "in_scope_architectures: [gfx950]\n"
                "quarantined_architectures: [gfx942]\n"
                "quarantined_pages: []\n",
                encoding="utf-8",
            )
            assert _scope.in_scope_architectures() == {"gfx950"}
            scope_path.write_text(
                "in_scope_architectures: [gfx942, gfx950]\n"
                "quarantined_architectures: []\n"
                "quarantined_pages: []\n",
                encoding="utf-8",
            )
            assert _scope.in_scope_architectures() == {"gfx942", "gfx950"}
    finally:
        _index.WIKI_ROOT = original_index_root
        _index._MEMORY_CACHE = None
        _scope.WIKI_ROOT = original_scope_root
        _scope._SCOPE_CACHE = None
        if configured_cache is None:
            os.environ.pop("ROCM_WIKI_CACHE_DIR", None)
        else:
            os.environ["ROCM_WIKI_CACHE_DIR"] = configured_cache
        sys.path.remove(scripts_dir)


def test_alias_architecture():
    # MI355X must alias to gfx950
    r = run("scripts/query.py", "--architecture", "MI355X", "--type", "hardware",
            "--compact")
    assert r.returncode == 0
    assert "result" in r.stdout


def test_active_scope_contract():
    scope = yaml.safe_load((ROOT / "data/scope.yaml").read_text(encoding="utf-8"))
    assert scope["in_scope_architectures"] == ["gfx942", "gfx950"]
    active = set(scope["in_scope_architectures"])
    quarantined = set(scope["quarantined_pages"])

    for path in (ROOT / "wiki").rglob("*.md"):
        fm = frontmatter(path.relative_to(ROOT))
        if fm["id"] in quarantined:
            continue
        assert set(fm.get("architectures") or []) <= active, path.relative_to(ROOT)


def test_active_synthesis_has_no_unsupported_architecture_prose():
    scope = yaml.safe_load((ROOT / "data/scope.yaml").read_text(encoding="utf-8"))
    quarantined = set(scope["quarantined_pages"])
    unsupported = re.compile(
        r"gfx1201|gfx1250|gfx1100|gfx90a|RDNA3|RDNA4|MI400|MI450|"
        r"RX 9070|R9700|wave32",
        re.IGNORECASE,
    )
    for path in (ROOT / "wiki").rglob("*.md"):
        fm = frontmatter(path.relative_to(ROOT))
        if fm["id"] in quarantined:
            continue
        assert not unsupported.search(path.read_text(encoding="utf-8")), path


def test_grep_and_implementation_links_obey_scope():
    blocked = run("scripts/grep_wiki.py", "gfx1201", "--only", "wiki")
    recovery = run(
        "scripts/grep_wiki.py",
        "gfx1201",
        "--only",
        "wiki",
        "--include-out-of-scope",
    )
    assert blocked.returncode == 2
    assert "unsupported architecture" in blocked.stderr
    assert recovery.returncode == 0
    assert "hw-wmma" in recovery.stdout

    active_page = run("scripts/get_page.py", "kernel-fp8-gemm")
    blocked_pr = run("scripts/get_page.py", "pr-aiter-3228", "--frontmatter-only")
    retained_pr = run(
        "scripts/get_page.py",
        "pr-aiter-3228",
        "--frontmatter-only",
        "--include-out-of-scope",
    )
    assert active_page.returncode == retained_pr.returncode == 0
    assert blocked_pr.returncode == 1
    assert "pr-aiter-3228" not in active_page.stdout
    assert "pr-aiter-3228" in retained_pr.stdout


def test_scope_quarantine_query_and_indices():
    quarantined = {"hw-wmma", "lang-rocwmma", "migration-wmma-vs-mfma"}
    default = run("scripts/query.py", "wmma", "--synthesis", "--limit", "100",
                  "--compact", "--no-cache")
    recovery = run("scripts/query.py", "wmma", "--synthesis", "--limit", "100",
                   "--compact", "--no-cache", "--include-out-of-scope")
    assert default.returncode == recovery.returncode == 0
    assert not any(page_id in default.stdout for page_id in quarantined)
    assert all(page_id in recovery.stdout for page_id in quarantined)

    indices = "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "queries").glob("*.md")
    )
    assert not any(page_id in indices for page_id in quarantined)


def test_out_of_scope_query_refusal_and_recovery():
    queries = (
        "WMMA tuning on RDNA4 gfx1201",
        "MI400 gfx1250 MFMA instruction reference",
        "Optimize wave32 kernels on RX9070",
    )
    for query in queries:
        blocked = run("scripts/query.py", query, "--synthesis", "--compact")
        recovery = run(
            "scripts/query.py",
            query,
            "--synthesis",
            "--compact",
            "--include-out-of-scope",
        )
        assert blocked.returncode == 2, (query, blocked.stdout, blocked.stderr)
        assert "unsupported architecture" in blocked.stderr
        assert recovery.returncode == 0, (query, recovery.stderr)


def test_gfx950_first_examples_and_claims():
    for build_script in (ROOT / "examples").glob("*/build.sh"):
        body = build_script.read_text(encoding="utf-8")
        assert "--offload-arch=gfx1201" not in body, build_script
        assert ':-gfx1201}' not in body, build_script

    stale_output = ("RX 9070", "warpSize=32", "captured on gfx1201")
    for page in (ROOT / "wiki/kernels").glob("*.md"):
        body = page.read_text(encoding="utf-8")
        for marker in stale_output:
            assert marker not in body, f"{page}: stale output {marker!r}"


def test_corrected_register_dma_and_mxfp_guidance():
    register_pages = [
        "wiki/hardware/wavefront.md",
        "wiki/techniques/occupancy-tuning.md",
        "wiki/techniques/vgpr-budgeting.md",
        "wiki/patterns/low-occupancy.md",
        "wiki/patterns/vgpr-pressure.md",
    ]
    for relpath in register_pages:
        body = (ROOT / relpath).read_text(encoding="utf-8")
        assert "round_up(round_up(.vgpr_count, 4) + .agpr_count, 8)" not in body
        assert "round_up(round_up(vgpr_count, 4) + agpr_count, 8)" not in body

    wavefront = (ROOT / register_pages[0]).read_text(encoding="utf-8")
    assert "metadata `.vgpr_count` already reports the combined total" in wavefront
    assert "NumVgprs" in wavefront and "NumAgprs" in wavefront

    async_copy = (ROOT / "wiki/hardware/async-copy-lds.md").read_text(
        encoding="utf-8"
    )
    assert "&tile[lane]" not in async_copy
    assert "wave-uniform LDS base" in async_copy
    assert "out[global_index] = tile[threadIdx.x]" in async_copy
    waitcnt = (ROOT / "wiki/hardware/s-waitcnt.md").read_text(encoding="utf-8")
    assert "buffer_load_dwordx4" not in waitcnt

    mxfp = (ROOT / "wiki/hardware/mxfp.md").read_text(encoding="utf-8")
    assert "comes directly from the wider-K opcodes" not in mxfp
    assert "datatype-dependent execution rate" in mxfp


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
            ("4 GiB", "descriptor-window limit", "range-checked independently"),
        "wiki/techniques/buffer-oob-guard.md":
            ("dwordx2/x3/x4", "mapped memory"),
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
