#!/usr/bin/env python3
"""Mark legacy performance claims that lack a committed reproduction bundle."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from _wiki_root import WIKI_ROOT, configure_utf8_stdio


def main() -> int:
    changed_pages = 0
    changed_claims = 0
    for path in sorted((WIKI_ROOT / "wiki" / "kernels").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        match = re.match(
            r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)", text, re.DOTALL
        )
        if not match:
            continue
        frontmatter = yaml.safe_load(match.group(1)) or {}
        dirty = False
        for claim in frontmatter.get("performance_claims") or []:
            if "reproduction_id" not in claim and "unreproduced" not in claim:
                claim["unreproduced"] = True
                dirty = True
                changed_claims += 1
        if dirty:
            rendered = yaml.safe_dump(
                frontmatter,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )
            path.write_text(
                f"---\n{rendered}---\n{match.group(2)}",
                encoding="utf-8",
            )
            changed_pages += 1
    print(f"pages={changed_pages} claims={changed_claims}")
    return 0


if __name__ == "__main__":
    configure_utf8_stdio()
    raise SystemExit(main())
