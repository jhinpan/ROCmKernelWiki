# ROCmKernelWiki — AMD CDNA / RDNA Kernel Optimization Knowledge Base

> **Knowledge cutoff: 2026-05-15.** All upstream PRs, doc snapshots, and blog
> summaries are anchored to upstream state on or before this date (recorded in
> [`data/refresh-cutoff.yaml`](data/refresh-cutoff.yaml)). Tool versions are
> pinned in [`data/tool-versions.yaml`](data/tool-versions.yaml) (ROCm 7.0.2,
> Composable Kernel 1.1.0, Triton 3.4.0, …). To advance the cutoff, re-run
> `scripts/harvest_prs.py`, regenerate indices, and bump the cutoff file.

A structured knowledge base of **AMD Instinct & Radeon GPU kernel optimization**
for CDNA3 (gfx942 / MI300), CDNA4 (gfx950 / MI350–MI355X), and RDNA4 (gfx1201),
packaged as a Claude Code skill. The repository root **is** the skill directory —
clone it into `~/.claude/skills/` and it works out of the box.

## Acknowledgements & Citation

This project is **inspired by and modeled on** the excellent
[**KernelWiki**](https://github.com/mit-han-lab/KernelWiki) from
**MIT Han Lab** — their structured, agent-queryable knowledge base for NVIDIA
Blackwell/Hopper kernel optimization. ROCmKernelWiki adapts the same three-layer
architecture (`sources/` → `wiki/` → `queries/`), the YAML-frontmatter page
schema, and the skill packaging, retargeting all content to the AMD/ROCm
ecosystem. The KernelWiki three-layer design itself follows
[Karpathy's LLM-wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

If you use this knowledge base, please cite both:

```bibtex
@misc{rocmkernelwiki2026,
  title  = {ROCmKernelWiki: An AMD CDNA/RDNA GPU Kernel Optimization Knowledge Base},
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

This is a community project. It is **not** an official AMD or ROCm product.

## Install as a Claude Code Skill

```bash
git clone https://github.com/jhinpan/ROCmKernelWiki ~/.claude/skills/ROCmKernelWiki
pip install -r ~/.claude/skills/ROCmKernelWiki/requirements.txt
```

The skill auto-registers (`SKILL.md` lives at the clone root) and the query
scripts auto-resolve the wiki root to their own directory — no environment
variable required.

Smoke test:

```bash
cd ~/.claude/skills/ROCmKernelWiki
python3 scripts/query.py --tag mfma --type hardware --compact
python3 scripts/get_page.py kernel-fp8-gemm --frontmatter-only
```

Optional override for relocating the scripts:

```bash
export ROCM_WIKI_ROOT=/path/to/ROCmKernelWiki
```

## What's Here

- **7,400+ PR reference pages** from ROCm/composable_kernel, ROCm/aiter,
  ROCm/hipBLASLt, ROCm/Tensile, ROCm/rocBLAS, ROCm/flash-attention, ROCm/FlyDSL,
  ROCm/triton, plus ROCm-filtered vllm-project/vllm and sgl-project/sglang
- **~53 synthesized wiki pages** — hardware features, optimization techniques,
  kernel case studies, problem patterns, DSL/language guides, migration guides
- **20 doc/blog summaries** (AMD CDNA3/CDNA4 ISA, whitepapers, ROCm blogs) and
  **8 reference-repository studies** (FlyDSL, gcnasm, Composable Kernel, rocWMMA,
  AITER, hipBLASLt, Tensile, the Matrix Instruction Calculator)
- **9 candidate ledgers** in `candidates/` recording the include/defer/exclude
  decision for every scanned PR
- **6 auto-generated cross-reference indices** under `queries/`

## Hardware Scope

| Marketing | gfx | Arch | FP8 | Matrix unit | Wave |
|---|---|---|---|---|---|
| MI300A / MI300X / MI325X | `gfx942` | CDNA3 | **FNUZ** | MFMA | wave64 |
| MI350X / MI355X | `gfx950` | CDNA4 | **OCP** + FP6/FP4/MX | MFMA | wave64 |
| Radeon AI PRO R9700 | `gfx1201` | RDNA4 | OCP | **WMMA** | wave32/64 |

> The headline portability gotcha: **gfx942 FP8 (FNUZ) is not bit-compatible
> with gfx950 FP8 (OCP)**. See [`wiki/migration/gfx942-to-gfx950.md`](wiki/migration/gfx942-to-gfx950.md).

## Query Tools

| Tool | Purpose |
|---|---|
| `scripts/query.py` | Unified search (keywords + filters + alias-aware) |
| `scripts/get_page.py` | Fetch any page by `id` or path; `--follow-sources` |
| `scripts/grep_wiki.py` | Regex text search across wiki bodies and PR pages |

```bash
python3 scripts/query.py "flash attention ck-tile" --limit 5
python3 scripts/query.py --tag XDLOP --type hardware --compact     # alias → mfma
python3 scripts/query.py --architecture MI355X --type kernel       # alias → gfx950
python3 scripts/get_page.py kernel-flash-attention-ck --follow-sources
python3 scripts/grep_wiki.py "v_mfma_f32_16x16x128_f8f6f4" --only wiki
```

## Architecture

Three layers (after MIT Han Lab's KernelWiki, in turn after Karpathy's LLM-wiki):

1. **`sources/`** — Raw data. Immutable summaries of PRs, docs, blogs, and
   reference repos. Cross-referenced by `id`.
2. **`wiki/`** — Synthesized knowledge pages with YAML frontmatter
   (subfolders: `hardware`, `techniques`, `kernels`, `patterns`, `languages`,
   `migration`).
3. **`queries/`** — Auto-generated cross-reference indices. Do not edit by hand;
   regenerate via `scripts/generate-indices.py`.

Supporting files:
- `data/schemas.yaml` — required/optional fields per page type
- `data/tags.yaml` — controlled vocabulary (validator-enforced)
- `data/aliases.yaml` — canonical → synonym map (MI300→gfx942, XDLOP→mfma, …)
- `data/inclusion-policy.yaml` — PR harvest classification policy
- `data/tool-versions.yaml`, `data/refresh-cutoff.yaml` — version/cutoff anchors
- `candidates/` — per-repo PR candidate ledgers
- `references/` — primer, schema, worked examples

## Maintenance Tooling

| Script | Purpose |
|---|---|
| `scripts/harvest_prs.py` | Harvest merged PRs from tracked ROCm repos (gh GraphQL) |
| `scripts/gen_source_anchors.py` | (Re)generate doc/blog/ref source anchor pages |
| `scripts/generate-indices.py` | Regenerate `queries/*.md` from frontmatter |
| `scripts/validate.py` | Validate frontmatter, vocabulary, and link integrity |

```bash
pip install -r requirements.txt
python3 scripts/validate.py            # schema + vocabulary + link integrity
python3 scripts/generate-indices.py    # regenerate query indices
```

## Quality Gates (cutoff 2026-05-15)

- 0 validation errors (schema, controlled vocabulary, link integrity)
- Every hardware fact traces to an official AMD ISA doc / whitepaper
- Every technique/kernel/language page has a compilable code snippet
- Every PR page carries `inclusion_reason` and `status: merged`
- `verified` pages carry `evidence_basis` (official-doc + upstream-code/paper)
- 0 dangling internal references

## License

Tooling and scripts are released under **Apache-2.0** (see [`LICENSE`](LICENSE)).
Wiki synthesis pages are derivative works that cite their upstream sources; PR
summary pages link to and summarize publicly available upstream PR metadata, with
the upstream repositories remaining the authoritative source of truth. AMD,
Instinct, Radeon, CDNA, and ROCm are trademarks of Advanced Micro Devices, Inc.;
this project is unaffiliated with AMD.
