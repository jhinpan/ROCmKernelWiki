# Codex Skill Contract Test Results

Validated on 2026-07-21 with Python and PyYAML on Windows. The AMD kernel
content has separate hardware evidence in [`VERIFICATION.md`](../VERIFICATION.md);
this check covers Codex skill packaging and query portability. It does not
exercise the interactive `/skills` picker.

## Install (per README)

```bash
ROCM_WIKI_SKILL="$HOME/.agents/skills/rocm-kernel-wiki"
git clone --depth 1 https://github.com/jhinpan/ROCmKernelWiki \
  "$ROCM_WIKI_SKILL"
python3 -m venv "$ROCM_WIKI_SKILL/.venv"
"$ROCM_WIKI_SKILL/.venv/bin/python" -m pip install -r \
  "$ROCM_WIKI_SKILL/requirements.txt"
```

The root `SKILL.md` passes the Codex skill validator. Its `name` matches the
install directory, `agents/openai.yaml` provides UI metadata, and scripts resolve
the wiki root without changing the user's working directory.

## Results

| Check | Result |
|---|---|
| Codex `SKILL.md` contract | PASS |
| `agents/openai.yaml` contract | PASS |
| Query launched from an unrelated working directory | PASS |
| Forced cp1252 child process emits UTF-8 safely | PASS |
| User-scoped query cache writes outside the skill checkout | PASS |
| Corrupted and stale query caches rebuild automatically | PASS |
| `tests/test_validate.py` (12 tests) | PASS x12 |
| `scripts/validate.py` | 0 errors / 7,541 pages |
| Natural-language query ("avoid LDS bank conflicts on MI300") | returns pattern + technique + kernel, correctly ranked |
| Alias arch filter (`--architecture MI355X` -> gfx950) + `--repo` | returns real gfx950 CK PRs |
| Symptom path (`--symptom bank-conflicts`) | -> pattern-bank-conflicts |
| `get_page.py` by id | frontmatter + body resolve |
| `grep_wiki.py` ISA mnemonic across 7,541 pages | correct hits |

Conclusion: the repository satisfies the published Codex discovery contract,
and its query tools work from a separate project directory.
