#!/usr/bin/env python3
"""Build the bidirectional PR <-> wiki bridge.

Measured before this script: 0/53 wiki pages cited any of the 7,454 real PRs, and
0 PR pages linked back to a synthesis page. The two corpora were disconnected
islands. This linker scores every merged PR against every wiki page using shared
facets + title keywords, then writes:

  - wiki page frontmatter:  implemented_by: [pr-<repo>-<N>, ...]   (top matches)
  - PR page frontmatter:    related: [<wiki-id>, ...]              (reverse links)

So an agent reading kernel-fp8-gemm can jump straight to the real PRs that built
it, and from any PR page back to the synthesis that explains it.

Idempotent / re-runnable (like generate-indices.py). Only links merged PRs; skips
revert/sync/chore/bump/typo titles. Drops links below a score threshold.

Run:
  python3 scripts/link_prs.py
  python3 scripts/link_prs.py --dry-run
  python3 scripts/link_prs.py --max-per-wiki 8 --max-per-pr 3
"""
import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT  # noqa: E402
from _scope import is_active  # noqa: E402

WIKI_DIR = WIKI_ROOT / "wiki"
PRS_DIR = WIKI_ROOT / "sources" / "prs"

SKIP_TITLE = re.compile(r"\b(revert|sync|chore|bump|typo|copyright|changelog|"
                        r"readme|lint|pre-commit|formatting|whitespace|"
                        r"clang-format|version)\b", re.I)
STOP = {"the", "a", "an", "on", "for", "of", "to", "and", "with", "in", "via",
        "amd", "gpu", "kernel", "kernels", "support", "add", "fix", "use", "cdna"}
WIKI_ID_PREFIXES = (
    "hw-", "technique-", "kernel-", "pattern-", "lang-", "migration-",
)


def split_fm(text):
    m = re.match(r'^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)', text, re.DOTALL)
    if not m:
        return None, None
    try:
        return yaml.safe_load(m.group(1)), m.group(2)
    except yaml.YAMLError:
        return None, None


def write_fm(md, fm, body):
    new_yaml = yaml.dump(fm, sort_keys=False, allow_unicode=True,
                         default_flow_style=False)
    md.write_text(f"---\n{new_yaml}---\n{body}", encoding="utf-8")


def facets(fm):
    s = set()
    for k in ("hardware_features", "techniques", "kernel_types"):
        for v in (fm.get(k) or []):
            s.add(f"{k[0]}:{v}")          # namespace so hw 'fp8' != kt 'fp8'
    return s


def architectures_compatible(left, right):
    left_arches = set(left.get("architectures") or [])
    right_arches = set(right.get("architectures") or [])
    return (
        not left_arches
        or not right_arches
        or not left_arches.isdisjoint(right_arches)
    )


def concept_set(fm):
    """Architecture-independent concept terms drawn from facets AND tags, so
    hardware/pattern/language/migration pages (which carry their concepts in
    `tags` rather than typed facet fields) can still match PRs that share those
    concepts. Returns a flat lowercase set."""
    s = set()
    for k in ("hardware_features", "techniques", "kernel_types", "tags"):
        for v in (fm.get(k) or []):
            s.add(str(v).lower())
    return s


def title_keywords(title):
    return {w for w in re.findall(r"[a-z0-9_]{3,}", str(title).lower())
            if w not in STOP}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-per-wiki", type=int, default=8)
    ap.add_argument("--max-per-pr", type=int, default=3)
    ap.add_argument("--min-score", type=float, default=6.0)
    ap.add_argument("--include-out-of-scope", action="store_true")
    args = ap.parse_args()

    # load wiki pages
    wikis = []
    for md in sorted(WIKI_DIR.rglob("*.md")):
        fm, body = split_fm(md.read_text(encoding="utf-8"))
        if not fm or not fm.get("id"):
            continue
        if not args.include_out_of_scope and not is_active(fm):
            continue
        wikis.append({
            "md": md, "fm": fm, "body": body, "id": fm["id"],
            "facets": facets(fm),
            "concepts": concept_set(fm),
            "kw": title_keywords(fm.get("title", "")),
            "tags": {str(t).lower() for t in (fm.get("tags") or [])},
        })

    # load PRs
    all_prs = []
    prs = []
    for md in PRS_DIR.rglob("PR-*.md"):
        fm, body = split_fm(md.read_text(encoding="utf-8"))
        if not fm or not fm.get("id"):
            continue
        pr = {"md": md, "fm": fm, "body": body, "id": fm["id"]}
        all_prs.append(pr)
        if fm.get("status") != "merged":
            continue
        if not args.include_out_of_scope and not is_active(fm):
            continue
        if SKIP_TITLE.search(str(fm.get("title", ""))):
            continue
        prf = facets(fm)
        if not prf:                       # nothing to match on
            continue
        prs.append({
            **pr,
            "facets": prf,
            "concepts": concept_set(fm),
            "kw": title_keywords(fm.get("title", "")),
            "repo": str(fm.get("repo", "")),
        })
    print(f"loaded {len(wikis)} wiki pages, {len(prs)} candidate merged PRs")

    # score each (wiki, pr) pair
    # wiki -> list of (score, pr_id); pr -> list of (score, wiki_id)
    wiki_links = {w["id"]: [] for w in wikis}
    pr_links = {}

    for w in wikis:
        wfac, wkw, wcon = w["facets"], w["kw"], w["concepts"]
        for pr in prs:
            if not architectures_compatible(w["fm"], pr["fm"]):
                continue
            shared = wfac & pr["facets"]
            score = 0.0
            # Strong signal: shared typed facets (kernel_type weighed most).
            for f in shared:
                score += 5.0 if f.startswith("k:") else 3.0
            # Fallback/boost: shared concept terms (covers tag-only wiki pages
            # like hw-*/pattern-*/lang-*/migration-* that have no typed facets).
            con_shared = wcon & pr["concepts"]
            score += 1.5 * len(con_shared)
            # Title keyword overlap.
            score += 2.0 * len(wkw & pr["kw"])
            if score >= args.min_score:
                wiki_links[w["id"]].append((score, pr["id"]))
                pr_links.setdefault(pr["id"], []).append((score, w["id"]))

    # trim to top-N each direction
    wiki_top = {wid: [pid for _, pid in sorted(v, reverse=True)[:args.max_per_wiki]]
                for wid, v in wiki_links.items()}
    pr_top = {pid: [wid for _, wid in sorted(v, reverse=True)[:args.max_per_pr]]
              for pid, v in pr_links.items()}

    n_wiki_linked = sum(1 for v in wiki_top.values() if v)
    n_pr_linked = sum(1 for v in pr_top.values() if v)
    total_edges = sum(len(v) for v in wiki_top.values())
    print(f"wiki pages with >=1 PR link: {n_wiki_linked}/{len(wikis)}")
    print(f"PR pages back-linked:        {n_pr_linked}")
    print(f"total wiki->PR edges:        {total_edges}")
    if args.dry_run:
        print("(dry-run; nothing written)")
        # show a sample
        for w in wikis[:6]:
            if wiki_top[w["id"]]:
                print(f"  {w['id']}: {wiki_top[w['id']][:5]}")
        return

    # write wiki frontmatter: implemented_by
    for w in wikis:
        links = wiki_top[w["id"]]
        fm = w["fm"]
        if links:
            fm["implemented_by"] = links
        elif "implemented_by" in fm:
            del fm["implemented_by"]
        write_fm(w["md"], fm, w["body"])

    # write PR frontmatter: related (merge, keep unique, wiki ids only here)
    for pr in all_prs:
        links = pr_top.get(pr["id"], [])
        fm = pr["fm"]
        existing = [r for r in (fm.get("related") or [])
                    if not str(r).startswith(WIKI_ID_PREFIXES)]
        merged = links + existing
        if merged:
            if fm.get("related") != merged:
                fm["related"] = merged
                write_fm(pr["md"], fm, pr["body"])
        elif "related" in fm:
            del fm["related"]
            write_fm(pr["md"], fm, pr["body"])

    print("wrote implemented_by (wiki) + related (PR) links.")


if __name__ == "__main__":
    main()
