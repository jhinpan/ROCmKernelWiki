"""Shared wiki-root resolution for ROCmKernelWiki skill scripts.

The skill is a flat repository — SKILL.md, data/, wiki/, sources/, queries/
all live at the same root. So by default the wiki root is this file's
grandparent directory:

    <wiki-root>/scripts/_wiki_root.py  ->  <wiki-root>

No environment variable is required for the common case. An optional
ROCM_WIKI_ROOT override is honored for advanced setups (e.g. scripts running
from a separate checkout). Any detected root is validated; misconfiguration
hard-errors rather than silently returning wrong results.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_utf8_stdio() -> None:
    """Make Unicode-rich wiki output safe on legacy Windows code pages."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            # Some embedded/captured streams expose reconfigure but reject it.
            pass


def _looks_like_wiki_root(p: Path) -> bool:
    return (p / "data" / "tags.yaml").is_file() and (p / "wiki").is_dir()


def _error(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def resolve_wiki_root() -> Path:
    # 1. explicit env override (advanced use)
    env = os.environ.get("ROCM_WIKI_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if _looks_like_wiki_root(p):
            return p
        _error(
            f"ROCM_WIKI_ROOT={env!r} does not point at a valid "
            f"ROCm kernel wiki (missing data/tags.yaml or wiki/)."
        )

    # 2. default: script-file grandparent == skill/wiki root
    default_root = Path(__file__).resolve().parent.parent
    if _looks_like_wiki_root(default_root):
        return default_root

    # 3. legacy: walk up from script location and from cwd
    seen = set()
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for candidate in [start, *start.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if _looks_like_wiki_root(candidate):
                return candidate

    _error(
        "Could not locate the ROCm kernel wiki root.\n"
        "       Expected a directory containing `data/tags.yaml` and `wiki/`.\n"
        "       Fix: run scripts from inside the cloned skill directory, or\n"
        "       set ROCM_WIKI_ROOT to its absolute path."
    )
    return Path()  # unreachable


WIKI_ROOT = resolve_wiki_root()
