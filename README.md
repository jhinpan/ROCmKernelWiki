# ROCmKernelWiki — AMD CDNA Kernel Optimization Knowledge Base

A structured, agent-queryable knowledge base of **AMD Instinct GPU kernel
optimization** for CDNA3 (gfx942 / MI300) and CDNA4
(gfx950 / MI350–MI355X). It is packaged as a **Codex CLI skill**, remains
compatible with Claude Code, and uses the repository root as the skill
directory so one `git pull` updates both tooling and corpus.

ROCmKernelWiki couples architecture-scoped synthesis to merged-PR and
primary-source provenance, [real-silicon validation](VERIFICATION.md),
and a [review-gated evidence flywheel](#self-evolution-boundary). Automation can
discover, triage, and propose evidence in Draft PRs, but it cannot approve facts
or merge changes.

> **Corpus freshness:** the generated
> [`data/corpus-manifest.yaml`](data/corpus-manifest.yaml) records the last
> complete baseline PR-harvest cutoff from
> [`data/refresh-cutoff.yaml`](data/refresh-cutoff.yaml). Incremental refreshes
> may add selected evidence after that date;
> [`data/evolution-state.yaml`](data/evolution-state.yaml) records each source's
> latest discovery position, not a complete corpus through-date. Doc/blog
> retrieval dates and the guide-sync boundary advance independently. The nod-ai
> AMDGPU optimization guide is synced through commit `efa471ae` on
> **2026-07-20**. Tool versions remain pinned in
> [`data/tool-versions.yaml`](data/tool-versions.yaml). gfx950 facts and examples
> were verified on MI350X, with guide-specific device/LDS checks repeated on
> MI355X — see below.

## Hardware Scope

| Marketing | gfx | Arch | FP8 | Matrix unit | Wave |
|---|---|---|---|---|---|
| MI300A / MI300X / MI325X | `gfx942` | CDNA3 | **FNUZ** | MFMA | wave64 |
| **MI350X / MI355X** | **`gfx950`** | **CDNA4** | **OCP** + FP6/FP4/MX | MFMA | wave64 |

Raw PR/source material for other architectures is retained for recovery but is
excluded from the active skill, default query results, and generated indices.

> The headline portability gotcha: **gfx942 FP8 (FNUZ) is not bit-compatible with
> gfx950 FP8 (OCP)**. See [`wiki/migration/gfx942-to-gfx950.md`](wiki/migration/gfx942-to-gfx950.md).

## Validated on real silicon (MI350X and MI355X / gfx950)

Unlike a docs-only wiki, the gfx950 claims here were **checked on an actual AMD Instinct
MI350X** (ROCm 7.2) by compiling, running, and disassembling code — each finding re-run by
an adversarial second pass. Full evidence: [`VERIFICATION.md`](VERIFICATION.md) and
[`data/hardware-verified.yaml`](data/hardware-verified.yaml).

The 2026-07-20 guide sync additionally ran on MI355X/ROCm 7.1.1: HIP reported
256 CUs, wave64, 32 waves/CU, and 160 KiB LDS; the upstream empirical-LDS
harness reproduced 64 banks plus the b32/b64 phase groups. Its b128 classifier
was inconclusive and MI300X SSH access was unavailable, both recorded as limits
in [`VERIFICATION.md`](VERIFICATION.md).

- **Hardware facts re-grounded on silicon** and corrected where the GPU disagreed with the
  docs: gfx950 cross-lane is `v_permlane16_swap` (not the RDNA selector form); **32 waves/CU**
  (not 40); direct-to-LDS is ≤16 B on gfx950 / ≤4 B on gfx942; compute modes SPX/DPX/QPX/CPX,
  memory NPS1/NPS2; native `xf32` MFMA *fails to select* on gfx950.
- **All 12 runnable examples** build with `--offload-arch=gfx950` **and execute** on the GPU
  (11/12 self-check; `fp8-gemm`'s `main()` only verifies the emitted MFMA, no numeric check).
- **First-party FlyDSL kernel sweep on MI350X** — every major FlyDSL gfx950 kernel was
  profiled with rocprofv3 ATT + counters against matched AITER/CK/hipBLASLt baselines.
  The detailed verdict table, root-cause notes, and dashboard links live in the
  canonical [`ref-flydsl-kernel-profiling`](sources/refs/ref-flydsl-kernel-profiling.md)
  source page; synthesized pages link back to it instead of duplicating the full summary.

## What's Here

- Merged-PR reference pages from allowlisted ROCm repositories and
  ROCm-filtered ecosystem projects
- Active synthesized wiki pages plus quarantined material retained for recovery
- Official-doc/blog summaries and reference-repository studies retained for provenance
- Per-source candidate ledgers in `candidates/` recording the include/defer/exclude
  decision for every scanned PR
- Auto-generated cross-reference indices under `queries/`
- Byte-capped upstream PR diffs under `artifacts/prs/<repo>/PR-<N>/`, SHA-256-pinned via `PROVENANCE.yaml`
- gfx950-first example suites under `examples/`; compiler-only subparts
  are labeled explicitly (see [`VERIFICATION.md`](VERIFICATION.md))

Current counts and cutoffs are generated from the checkout into
[`data/corpus-manifest.yaml`](data/corpus-manifest.yaml); documentation does not
maintain a second inventory.

## Install as a Codex CLI Skill

Codex discovers personal skills under
[`$HOME/.agents/skills`](https://developers.openai.com/codex/skills#where-to-save-skills).
Clone this repo there using the lowercase skill name from `SKILL.md`:

```bash
ROCM_WIKI_SKILL="$HOME/.agents/skills/rocm-kernel-wiki"
mkdir -p "$HOME/.agents/skills"
git clone --depth 1 https://github.com/jhinpan/ROCmKernelWiki \
  "$ROCM_WIKI_SKILL"
python3 -m venv "$ROCM_WIKI_SKILL/.venv"
"$ROCM_WIKI_SKILL/.venv/bin/python" -m pip install -r \
  "$ROCM_WIKI_SKILL/requirements.txt"
```

PowerShell:

```powershell
$RocmWikiSkill = Join-Path $HOME '.agents\skills\rocm-kernel-wiki'
New-Item -ItemType Directory -Force (Split-Path $RocmWikiSkill) | Out-Null
git clone --depth 1 https://github.com/jhinpan/ROCmKernelWiki $RocmWikiSkill
$RocmWikiBootstrap = $null
$RocmWikiBootstrapArgs = @()
foreach ($RocmWikiCandidate in @(
  @{ Name = 'python'; Args = @() },
  @{ Name = 'python3'; Args = @() },
  @{ Name = 'py'; Args = @('-3') }
)) {
  $RocmWikiCommand = Get-Command $RocmWikiCandidate.Name -ErrorAction SilentlyContinue
  if (-not $RocmWikiCommand) { continue }
  $RocmWikiCandidateArgs = @($RocmWikiCandidate.Args)
  $RocmWikiProbeOk = $false
  try {
    $LASTEXITCODE = 1
    & $RocmWikiCommand.Source @RocmWikiCandidateArgs -c "import sys; raise SystemExit(sys.version_info < (3, 9))" 2>$null
    $RocmWikiProbeOk = ($LASTEXITCODE -eq 0)
  } catch {}
  if ($RocmWikiProbeOk) {
    $RocmWikiBootstrap = $RocmWikiCommand.Source
    $RocmWikiBootstrapArgs = $RocmWikiCandidateArgs
    break
  }
}
if (-not $RocmWikiBootstrap) { throw 'Install Python 3 and add its launcher to PATH.' }
& $RocmWikiBootstrap @RocmWikiBootstrapArgs -m venv (Join-Path $RocmWikiSkill '.venv')
$RocmWikiPython = Join-Path $RocmWikiSkill '.venv\Scripts\python.exe'
& $RocmWikiPython -m pip install -r (Join-Path $RocmWikiSkill 'requirements.txt')
```

Start a new Codex CLI session, run `/skills`, and select
`$rocm-kernel-wiki`, or invoke it directly:

```text
$rocm-kernel-wiki find the best LDS swizzle for this gfx950 transpose kernel
```

Codex may also activate it automatically when a request matches the skill
description. If a newly installed skill does not appear, restart Codex. Update
the corpus later with:

```bash
ROCM_WIKI_SKILL="$HOME/.agents/skills/rocm-kernel-wiki"
git -C "$ROCM_WIKI_SKILL" pull --ff-only
"$ROCM_WIKI_SKILL/.venv/bin/python" -m pip install -r \
  "$ROCM_WIKI_SKILL/requirements.txt"
```

PowerShell update:

```powershell
$RocmWikiSkill = Join-Path $HOME '.agents\skills\rocm-kernel-wiki'
$RocmWikiPython = Join-Path $RocmWikiSkill '.venv\Scripts\python.exe'
git -C $RocmWikiSkill pull --ff-only
& $RocmWikiPython -m pip install -r (Join-Path $RocmWikiSkill 'requirements.txt')
```

The query scripts resolve the wiki root from their own location, so Codex can
run them by absolute path without changing the user's project directory. No
environment variable is required. Optional overrides are
`ROCM_WIKI_ROOT=/path/to/ROCmKernelWiki` and
`ROCM_WIKI_CACHE_DIR=/writable/cache/path`; by default, the query cache lives
under the OS temporary directory rather than modifying the skill checkout.

Smoke test:

```bash
ROCM_WIKI_SKILL="$HOME/.agents/skills/rocm-kernel-wiki"
ROCM_WIKI_PYTHON="$ROCM_WIKI_SKILL/.venv/bin/python"
"$ROCM_WIKI_PYTHON" "$ROCM_WIKI_SKILL/scripts/query.py" \
  --tag mfma --type hardware --compact
"$ROCM_WIKI_PYTHON" "$ROCM_WIKI_SKILL/scripts/get_page.py" \
  kernel-flydsl-flash-attention --frontmatter-only
```

PowerShell smoke test:

```powershell
$RocmWikiSkill = Join-Path $HOME '.agents\skills\rocm-kernel-wiki'
$RocmWikiPython = Join-Path $RocmWikiSkill '.venv\Scripts\python.exe'
& $RocmWikiPython (Join-Path $RocmWikiSkill 'scripts\query.py') --tag mfma --type hardware --compact
```

For Claude Code, use the same clone-and-venv procedure with
`~/.claude/skills/rocm-kernel-wiki` as the skill path; install the same
`requirements.txt`. The shared `SKILL.md` stays compatible.

## Query Tools

| Tool | Purpose |
|---|---|
| `scripts/query.py` | Unified search (keywords + filters + alias-aware) |
| `scripts/get_page.py` | Fetch any page by `id` or path; `--follow-sources` |
| `scripts/grep_wiki.py` | Regex text search across wiki bodies and PR pages |

```bash
ROCM_WIKI_SKILL="$HOME/.agents/skills/rocm-kernel-wiki"
ROCM_WIKI_PYTHON="$ROCM_WIKI_SKILL/.venv/bin/python"
"$ROCM_WIKI_PYTHON" "$ROCM_WIKI_SKILL/scripts/query.py" "flash attention ck-tile" --limit 5
"$ROCM_WIKI_PYTHON" "$ROCM_WIKI_SKILL/scripts/query.py" --architecture MI355X --type kernel
"$ROCM_WIKI_PYTHON" "$ROCM_WIKI_SKILL/scripts/get_page.py" kernel-flash-attention-ck --follow-sources
"$ROCM_WIKI_PYTHON" "$ROCM_WIKI_SKILL/scripts/grep_wiki.py" "v_mfma_f32_16x16x128_f8f6f4" --only wiki
```

## Architecture

Three layers (after MIT Han Lab's KernelWiki, in turn after Karpathy's LLM-wiki):

<p align="center"><img src="docs/architecture.svg" alt="ROCmKernelWiki three-layer architecture: sources → wiki → queries, gated by data/ and scripts/" width="780"></p>

1. **`sources/`** — Raw data. Immutable summaries of PRs, docs, blogs, and reference
   repos. Cross-referenced by `id`.
2. **`wiki/`** — Synthesized knowledge pages with YAML frontmatter (subfolders:
   `hardware`, `techniques`, `kernels`, `patterns`, `languages`, `migration`).
3. **`queries/`** — Auto-generated cross-reference indices. Do not edit by hand;
   regenerate via `scripts/generate-indices.py`.

Supporting files: `data/` holds the schema and controlled vocabulary
(`schemas.yaml`, `tags.yaml`, `aliases.yaml`, `inclusion-policy.yaml`,
`scope.yaml`, `sources.yaml`, `tool-versions.yaml`, `refresh-cutoff.yaml`,
`hardware-verified.yaml`);
`candidates/` holds per-repo PR ledgers; `references/` holds the primer, schema, and
worked examples.

## Maintenance Tooling

[`data/sources.yaml`](data/sources.yaml) is the canonical registry for the
refresh pipeline. Each run caps candidates, generated source pages, and stored
diff bytes so the rolling Draft PR remains reviewable. The daily worker updates
that PR from a disposable clone. Approved hardware tasks are evaluated
separately on the trusted MI355 node against an exact candidate SHA.

| Script | Purpose |
|---|---|
| `scripts/evolve/discover.py` | Incrementally discover PR/tree evidence from `data/sources.yaml` |
| `scripts/harvest_prs.py` | Compatibility wrapper for merge-safe incremental discovery |
| `scripts/evolve/gaps.py` | Turn uncovered evidence clusters into synthesis proposals |
| `scripts/evolve/synthesize.py` | Validate a credential-free, path-bounded synthesis adapter |
| `scripts/evolve/refresh.py` | Run one budgeted discovery→triage→eval→validation refresh |
| `scripts/evolve/corpus.py` | Generate or check the canonical corpus inventory and cutoffs |
| `scripts/evolve/daily_worker.py` | Update the rolling `bot/evolution` Draft PR in a disposable clone |
| `scripts/evolve/mi355_worker.py` | Run exact-SHA approved evidence tasks on the trusted MI355 node |
| `scripts/backfill_diffs.py` | Fetch real upstream diffs for top-ranked kernel PRs |
| `scripts/enrich_facets.py` | Infer techniques/hardware_features/kernel_types from paths + diffs |
| `scripts/link_prs.py` | Build the bidirectional PR↔wiki bridge |
| `scripts/generate-indices.py` | Regenerate `queries/*.md` from frontmatter |
| `scripts/evaluate_skill.py` | Score held-out retrieval, citations, and architecture safety |
| `scripts/evaluate_answers.py` | Check reference-answer facts and source citations |
| `scripts/verify_provenance.py` | Re-hash artifact bundles and backfill immutable merge SHAs |
| `scripts/validate.py` | Validate pages, evidence, candidates, manifests, claims, and provenance |

CI (`.github/workflows/ci.yml`) gates every push on the validator, the query-tool
smoke tests, scored retrieval/answer evals, provenance, and index freshness.
`main` is protected by required checks, CODEOWNER review, one human approval,
linear history, and resolved conversations.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/validate.py            # schema + vocabulary + link integrity
.venv/bin/python scripts/generate-indices.py    # regenerate query indices
.venv/bin/python scripts/evaluate_skill.py --check
```

### Self-evolution boundary

The system discovers, proposes, evaluates, and collects evidence; it does not
decide its own truth. Every automated change stays in a Draft PR and the bot has
no approval or merge capability. PR/blog text and stored diffs are rendered as
`UNTRUSTED-UPSTREAM-*` data. The public repository is not connected to a
persistent self-hosted runner: [`ops/mi355/`](ops/mi355/) documents the
node-local, exact-SHA approval and sandbox contract.

### Quality Gates

- 0 validation errors (schema, controlled vocabulary, link integrity)
- Every hardware fact traces to an official AMD ISA doc / whitepaper
- Every technique/kernel/language page has a compilable code snippet
- Every PR page carries `inclusion_reason` and `status: merged`
- `verified` pages carry `evidence_basis` (official-doc + upstream-code/paper)
- Machine-authored pages cannot self-promote to `verified`
- Performance claims link a reproduction bundle or say `unreproduced: true`
- Stored PR diffs are re-hashed against `PROVENANCE.yaml`
- 0 dangling internal references (frontmatter ids **and** in-body relative links)
- **gfx950 hardware/numeric claims re-verified on real MI350X silicon (ROCm 7.2)** —
  see [`VERIFICATION.md`](VERIFICATION.md) and [`data/hardware-verified.yaml`](data/hardware-verified.yaml)

## License

Tooling and scripts are released under **Apache-2.0** (see [`LICENSE`](LICENSE)).
Wiki synthesis pages are derivative works that cite their upstream sources; PR
summary pages link to and summarize publicly available upstream PR metadata, with
the upstream repositories remaining the authoritative source of truth. AMD,
Instinct, Radeon, CDNA, and ROCm are trademarks of Advanced Micro Devices, Inc.;
this project is unaffiliated with AMD. It is **not** an official AMD or ROCm product.

## Acknowledgements & Citation

This project is **inspired by and modeled on** the excellent
[**KernelWiki**](https://github.com/mit-han-lab/KernelWiki) from **MIT Han Lab** —
their structured, agent-queryable knowledge base for NVIDIA Blackwell/Hopper kernel
optimization. ROCmKernelWiki adapts the same three-layer architecture
(`sources/` → `wiki/` → `queries/`), the YAML-frontmatter page schema, and the skill
packaging, retargeting all content to the AMD/ROCm ecosystem. The KernelWiki three-layer
design itself follows
[Karpathy's LLM-wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

If you use this knowledge base, please cite both:

```bibtex
@misc{rocmkernelwiki2026,
  title  = {ROCmKernelWiki: An AMD CDNA GPU Kernel Optimization Knowledge Base},
  author = {ROCmKernelWiki contributors},
  year   = {2026},
  howpublished = {\url{https://github.com/jhinpan/ROCmKernelWiki}},
  note   = {Inspired by MIT Han Lab's KernelWiki}
}

@misc{kernelwiki2026,
  title  = {KernelWiki: Blackwell \& Hopper Kernel Optimization Knowledge Base},
  author = {MIT Han Lab},
  year   = {2026},
  howpublished = {\url{https://github.com/mit-han-lab/KernelWiki}}
}
```
