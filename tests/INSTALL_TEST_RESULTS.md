# Skill Installation Test Results

Tested on: AMD Radeon RX 9070 XT (gfx1201, RDNA4), ROCm 7.2.3, Python 3.14, PyYAML 6.0.3.

## Install (per README)
```bash
ln -s ~/ROCmKernelWiki ~/.claude/skills/ROCmKernelWiki   # (or git clone)
pip install -r ~/.claude/skills/ROCmKernelWiki/requirements.txt
```
Skill auto-registers via SKILL.md at clone root; scripts auto-resolve the wiki
root (no env var needed). Verified by running every tool from the installed path.

## Results (all from ~/.claude/skills/ROCmKernelWiki)

| Check | Result |
|---|---|
| `tests/test_validate.py` (4 tests) | PASS x4 |
| `scripts/validate.py` | 0 errors / 7,535 pages |
| Natural-language query ("avoid LDS bank conflicts on MI300") | returns pattern + technique + kernel, correctly ranked |
| Alias arch filter (`--architecture MI355X` -> gfx950) + `--repo` | returns real gfx950 CK PRs |
| Symptom path (`--symptom bank-conflicts`) | -> pattern-bank-conflicts |
| `get_page.py` by id | frontmatter + body resolve |
| `grep_wiki.py` ISA mnemonic across 7,500 pages | 0.2 s, correct hits |

Conclusion: installs cleanly and is fully functional as a Claude Code skill.
