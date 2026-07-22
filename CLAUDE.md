# CLAUDE.md — Agent Guide for ROCmKernelWiki

This repository is an **agent skill for Codex CLI and Claude Code** and an
LLM-queryable knowledge base of AMD GPU kernel optimization (CDNA3 gfx942 /
CDNA4 gfx950). Read `SKILL.md` for the user-facing trigger
conditions; this file is the extended schema + navigation reference for agents
working *in* the repo.

## Layout

```
sources/      raw immutable summaries (cross-referenced by id)
  prs/<repo>/PR-<N>.md     7,454 merged-PR reference pages
  docs/                    official AMD/ROCm doc + paper summaries (doc-*)
  blogs/                   ROCm/community blog summaries (blog-*)
  refs/                    reference-repository studies (ref-*)
wiki/         synthesized pages with YAML frontmatter
  hardware/ techniques/ kernels/ patterns/ languages/ migration/
queries/      auto-generated indices (regenerate; never hand-edit)
data/         schemas, controlled vocabulary, aliases, policies, version anchors
candidates/   per-repo PR ledgers (include/defer/exclude decisions)
references/   primer.md, schema.md, examples.md
scripts/      query/get_page/grep + validate/generate/harvest tooling
```

## How to answer a question (recommended order)

1. If broad → read `references/primer.md` (topic map → page id).
2. `python3 scripts/query.py "<question>" [--tag --type --architecture …]`
3. `python3 scripts/get_page.py <id> --follow-sources`
4. `python3 scripts/grep_wiki.py "<regex>"` for instruction-level text search.
5. Use `queries/by-*.md` for structured cross-refs.

## Frontmatter schema (summary; full in data/schemas.yaml)

- **Every page** has a unique `id`. Prefixes: `pr-`, `doc-`, `blog-`, `ref-`,
  `hw-`, `technique-`, `kernel-`, `pattern-`, `lang-`, `migration-`.
- **wiki-kernel** must have `performance_claims` (each: gpu, dtype, metric,
  value, source_id).
- **wiki-technique / wiki-kernel / wiki-language** must have `reproducibility >=
  snippet` and a fenced code block.
- **wiki-pattern** must have `symptoms` and `candidate_techniques`.
- **wiki-migration** must have `cross_vendor_note`.
- `confidence: verified` requires `evidence_basis` with ≥1 `official-doc` + ≥1
  `upstream-code`/`paper`.

## Controlled vocabulary (validator-enforced)

All `architectures`, `hardware_features`, `techniques`, `kernel_types`,
`languages`, and `tags` values must exist in `data/tags.yaml`. Add new terms
there first. See `references/schema.md` for the current lists and
`data/aliases.yaml` for the alias map.

`data/scope.yaml` is the separate source of truth for active publication.
Vocabulary entries and raw PRs outside gfx942/gfx950 remain valid but are
excluded from default queries and generated indices.

## Critical correctness rules for AMD content

- **Always name the architecture.** gfx942 ≠ gfx950.
- **FP8 encoding differs by generation**: gfx942 = FNUZ, gfx950 = OCP. They are
  not bit-compatible. Never claim FP8 weights port unchanged across them.
- **LDS sizing differs**: 64 kB / 32 banks (gfx942) vs 160 kB / 64 banks (gfx950).
- **`v_permlane16_*` is gfx950-only.** On gfx942, reductions use DPP + ds_bpermute.
- **CDNA is wave64-only.**
- **No TMA / no mbarrier on AMD.** The async-copy analog is direct-to-LDS
  (`buffer_load…lds` / `global_load_lds`), gated by `s_waitcnt vmcnt(N)`.
- Prefer citing `doc-*`/`blog-*`/`ref-*` anchors (guaranteed to exist) over
  guessing PR ids.

## Maintenance

```bash
python3 scripts/validate.py            # must report 0 errors before committing
python3 scripts/generate-indices.py    # after adding/editing wiki pages
python3 scripts/harvest_prs.py --all   # refresh PR corpus (advances cutoff)
```

## Provenance & scope

PR pages summarize publicly available upstream PR metadata; the upstream repo is
the source of truth. This is a community knowledge base, not an AMD product.
Inspired by and modeled on MIT Han Lab's
[KernelWiki](https://github.com/mit-han-lab/KernelWiki) — see README for citation.
