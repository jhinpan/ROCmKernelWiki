#!/usr/bin/env python3
"""Unified query tool for the ROCm kernel wiki.

Supports natural-language keyword queries, tag filters, repo filters, and type filters.

Usage:
    query.py "how to pipeline MFMA on MI300"
    query.py --tag mfma --type kernel
    query.py --repo composable_kernel --limit 20
    query.py --language gcn-asm
    query.py --symptom bank-conflicts

Returns a ranked list of matching pages with titles, paths, and key frontmatter fields.
"""

import argparse
import re
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT, configure_utf8_stdio  # noqa: E402


_ALIAS_CACHE = None
_QUERY_CACHE_VERSION = 1


def query_cache_path() -> Path:
    """Return a writable, checkout-specific cache path outside the skill tree."""
    import hashlib
    import os
    import tempfile

    configured = os.environ.get("ROCM_WIKI_CACHE_DIR")
    if configured:
        cache_root = Path(configured).expanduser()
    else:
        getuid = getattr(os, "getuid", None)
        if getuid is not None:
            user_key = f"uid-{getuid()}"
        else:
            identity = (
                os.environ.get("USERNAME")
                or os.environ.get("USER")
                or str(Path.home())
            )
            user_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        cache_root = Path(tempfile.gettempdir()) / f"rocm-kernel-wiki-{user_key}"
    root_key = hashlib.sha256(str(WIKI_ROOT).encode("utf-8")).hexdigest()[:16]
    return cache_root / root_key / "query-index.json"


def load_alias_expansions():
    """Return a dict mapping lowercased alias → canonical term, from data/aliases.yaml."""
    global _ALIAS_CACHE
    if _ALIAS_CACHE is not None:
        return _ALIAS_CACHE
    out = {}
    aliases_path = WIKI_ROOT / "data" / "aliases.yaml"
    try:
        raw = yaml.safe_load(aliases_path.read_text(encoding="utf-8")) or {}
    except Exception:
        _ALIAS_CACHE = {}
        return _ALIAS_CACHE
    for canonical, variants in raw.items():
        if not isinstance(canonical, str):
            continue
        out.setdefault(canonical.lower(), canonical)
        for v in (variants or []):
            if isinstance(v, str):
                out.setdefault(v.lower(), canonical)
    _ALIAS_CACHE = out
    return out


def expand_keyword(kw):
    """Return search variants for a keyword: original plus any canonical term."""
    aliases = load_alias_expansions()
    canonical = aliases.get(kw.lower())
    if canonical and canonical.lower() != kw.lower():
        return [kw, canonical]
    return [kw]


def load_frontmatter(path):
    """Parse YAML frontmatter. Returns (fm_dict, body_str) or (None, None)."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return None, None
    m = re.match(r'^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)', content, re.DOTALL)
    if not m:
        return None, None
    try:
        fm = yaml.safe_load(m.group(1))
        if not isinstance(fm, dict):
            return None, None
        return fm, m.group(2)
    except yaml.YAMLError:
        return None, None


def load_all_pages(use_cache=True):
    """Load frontmatter + body for every sources/*.md and wiki/*.md file.

    Results are cached to a checkout-specific JSON index under the OS temporary
    directory (or ROCM_WIKI_CACHE_DIR), keyed by the max corpus mtime. Keeping
    the cache outside the skill tree supports read-only and sandboxed installs.
    """
    import json
    import tempfile

    cache_path = query_cache_path()
    md_files = []
    for subdir in ("sources", "wiki"):
        base = WIKI_ROOT / subdir
        if base.exists():
            md_files.extend(base.rglob("*.md"))
    if not md_files:
        return []
    latest = max(f.stat().st_mtime for f in md_files)
    sig = f"v{_QUERY_CACHE_VERSION}:{len(md_files)}:{latest:.3f}"

    if use_cache and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("sig") == sig:
                return cached["pages"]
        except Exception:
            pass

    pages = []
    for md in md_files:
        fm, body = load_frontmatter(md)
        if fm is None:
            continue
        pages.append({
            "path": str(md.relative_to(WIKI_ROOT)),
            "fm": fm,
            "body": body or "",
        })
    if use_cache:
        temporary_path = None
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=cache_path.parent,
                prefix=f".{cache_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                json.dump({"sig": sig, "pages": pages}, temporary_file)
                temporary_path = Path(temporary_file.name)
            temporary_path.replace(cache_path)
        except Exception:
            pass
        finally:
            try:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return pages


# Common English + markup stopwords that should not drive ranking.
STOPWORDS = {
    "a", "an", "the", "to", "of", "for", "on", "in", "is", "are", "how", "do",
    "does", "i", "my", "with", "and", "or", "what", "which", "when", "where",
    "use", "using", "can", "vs", "from", "into", "this", "that", "it", "be",
    "as", "at", "by", "if", "we", "you", "want", "need", "should", "make",
}

# Multiplicative priors: surface curated synthesis + runnable assets above raw PRs.
PTYPE_PRIOR = {
    "wiki-kernel": 1.7, "wiki-technique": 1.6, "wiki-hardware": 1.6,
    "wiki-pattern": 1.6, "wiki-language": 1.5, "wiki-migration": 1.5,
    "source-doc": 1.3, "source-ref": 1.3, "source-blog": 1.2,
    "source-pr": 1.0,
}


def build_idf(pages):
    """Document-frequency-based inverse weights over the searchable text of each
    page, so a rare term like `cp.async` outweighs a ubiquitous one like `gemm`."""
    import math
    n = len(pages) or 1
    df = {}
    for p in pages:
        fm = p["fm"]
        text = (str(fm.get("title", "")) + " " +
                " ".join(str(v) for k in ("tags", "techniques", "hardware_features",
                                          "kernel_types", "languages", "aliases",
                                          "symptoms") for v in (fm.get(k) or [])) +
                " " + p["body"]).lower()
        toks = set(re.findall(r"[a-z0-9_.+-]{2,}", text))
        for t in toks:
            df[t] = df.get(t, 0) + 1
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}, n


def detect_page_type(fm, path):
    """Return a page-type label for filtering."""
    if "type" in fm:
        return f"wiki-{fm['type']}"
    parts = path.split("/")
    if parts[0] == "sources" and len(parts) > 1:
        return f"source-{parts[1].rstrip('s')}"  # prs → source-pr, refs → source-ref
    return "unknown"


def _page_signal_prior(fm, ptype):
    """Combine page-type, runnable-asset, and low-signal priors into one factor."""
    factor = PTYPE_PRIOR.get(ptype, 1.0)
    # Reward pages backed by a runnable/benchmarked example.
    repro = str(fm.get("reproducibility", ""))
    if repro in ("runnable", "benchmarked"):
        factor *= 1.25
    # Damp low-signal PR placeholders: no inferred facets at all.
    if ptype == "source-pr":
        if not (fm.get("hardware_features") or fm.get("kernel_types")
                or fm.get("techniques")):
            factor *= 0.5
    return factor


def score_keyword_match(fm, body, keywords, idf=None, ptype="unknown"):
    """Score a page by alias-aware, IDF-weighted keyword matches in title, tags,
    and body, scaled by a page-type / runnable / signal prior.

    title hit = 10, tag/facet hit = 5, body hits = up to 3 — each multiplied by the
    keyword's IDF weight so rare, discriminative terms dominate over stopwords."""
    title_text = str(fm.get("title", "")).lower()
    tag_text = " ".join(
        str(v) for k in ("tags", "techniques", "hardware_features", "kernel_types",
                          "languages", "aliases", "symptoms")
        for v in (fm.get(k) or [])
    ).lower()
    body_lower = body.lower()
    raw = 0.0
    for kw in keywords:
        if kw.lower() in STOPWORDS:
            continue
        best = 0.0
        for variant in expand_keyword(kw):
            v_l = variant.lower()
            w = (idf.get(v_l, 1.0) if idf else 1.0)
            vs = 0.0
            if v_l in title_text:
                vs += 10
            if v_l in tag_text:
                vs += 5
            vs += min(body_lower.count(v_l), 3)
            best = max(best, vs * w)
        raw += best
    if raw <= 0:
        return 0.0
    return raw * _page_signal_prior(fm, ptype)


def extract_snippet(body, keywords, width=160):
    """Return a one-line context snippet around the first matched keyword."""
    bl = body.lower()
    for kw in keywords:
        if kw.lower() in STOPWORDS:
            continue
        for variant in expand_keyword(kw):
            i = bl.find(variant.lower())
            if i >= 0:
                start = max(0, i - width // 2)
                seg = body[start:start + width].replace("\n", " ").strip()
                return ("…" + seg + "…") if seg else ""
    return ""


def filter_pages(pages, args):
    out = []
    for p in pages:
        fm = p["fm"]
        path = p["path"]
        ptype = detect_page_type(fm, path)
        p["_ptype"] = ptype

        if args.type:
            if not ptype.endswith(args.type) and ptype != args.type:
                continue

        if args.tag:
            all_tags = set()
            for k in ("tags", "techniques", "hardware_features", "kernel_types", "languages"):
                all_tags.update(fm.get(k) or [])
            tag_variants = {v.lower() for v in expand_keyword(args.tag)}
            if not any(str(t).lower() in tag_variants for t in all_tags):
                continue

        if args.repo:
            repo = str(fm.get("repo", "")).lower()
            if args.repo.lower() not in repo:
                continue

        if args.language:
            langs = set(fm.get("languages") or [])
            tags = set(fm.get("tags") or [])
            if args.language not in langs and args.language not in tags:
                continue

        if args.architecture:
            archs = {str(a).lower() for a in (fm.get("architectures") or [])}
            arch_variants = {v.lower() for v in expand_keyword(args.architecture)}
            if not (archs & arch_variants):
                continue

        if args.symptom:
            symptoms = set(fm.get("symptoms") or [])
            if args.symptom not in symptoms:
                continue

        if args.confidence:
            if str(fm.get("confidence", "")) != args.confidence:
                continue

        if getattr(args, "synthesis", False) and not ptype.startswith("wiki-"):
            continue

        out.append(p)
    return out


def format_result(page, compact=False):
    fm = page["fm"]
    title = fm.get("title", "Untitled")
    path = page["path"]
    pid = fm.get("id", "")
    ptype = page.get("_ptype", "?")

    snip = page.get("_snippet")
    if compact:
        line = f"  [{ptype}] {pid}: {title}  ({path})"
        if snip:
            line += f"\n        ↳ {snip}"
        return line

    lines = [f"## {title}"]
    lines.append(f"- **id**: `{pid}`")
    lines.append(f"- **type**: `{ptype}`")
    lines.append(f"- **path**: `{path}`")
    if snip:
        lines.append(f"- **match**: {snip}")
    if "architectures" in fm:
        lines.append(f"- **architectures**: {fm['architectures']}")
    for k in ("confidence", "reproducibility"):
        if k in fm:
            lines.append(f"- **{k}**: {fm[k]}")
    for k in ("tags", "techniques", "hardware_features", "kernel_types", "languages"):
        v = fm.get(k)
        if v:
            lines.append(f"- **{k}**: {v}")
    if "performance_claims" in fm and isinstance(fm["performance_claims"], list):
        for claim in fm["performance_claims"][:2]:
            lines.append(f"- **perf**: {claim.get('value')} {claim.get('metric')} on "
                         f"{claim.get('gpu')} ({claim.get('dtype')}, {claim.get('shape')})")
    if fm.get("implemented_by"):
        lines.append(f"- **implemented_by (PRs)**: {fm['implemented_by'][:6]}")
    if "sources" in fm:
        lines.append(f"- **sources**: {fm['sources'][:5]}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Query the ROCm kernel wiki")
    parser.add_argument("query", nargs="*", help="Free-text keywords")
    parser.add_argument("--type", help="Page type (kernel, technique, hardware, pattern, language, migration, pr, doc, blog, ref)")
    parser.add_argument("--tag", help="Filter by tag (tags/techniques/hardware_features/kernel_types/languages); alias-aware")
    parser.add_argument("--repo", help="Filter by source repo (partial, e.g. 'composable_kernel')")
    parser.add_argument("--language", help="Filter by language/DSL (hip, gcn-asm, composable-kernel, flydsl, triton, ...)")
    parser.add_argument("--architecture", help="Filter by architecture (gfx942, gfx950, gfx1201, ...); alias-aware")
    parser.add_argument("--symptom", help="Filter by pattern symptom (bank-conflicts, low-occupancy, ...)")
    parser.add_argument("--confidence", help="Filter by confidence (verified, source-reported, inferred, experimental)")
    parser.add_argument("--synthesis", action="store_true",
                        help="Only curated wiki synthesis pages (skip raw PR sources)")
    parser.add_argument("--limit", type=int, default=10, help="Max results (default 10)")
    parser.add_argument("--compact", action="store_true", help="Compact one-line-per-result output")
    parser.add_argument("--paths-only", action="store_true", help="Output only file paths")
    parser.add_argument("--no-cache", action="store_true", help="Bypass the JSON query index")
    args = parser.parse_args()

    all_pages = load_all_pages(use_cache=not args.no_cache)
    pages = filter_pages(all_pages, args)

    keywords = []
    for q in args.query:
        for tok in re.split(r"\s+", q.strip()):
            if tok:
                keywords.append(tok)
    if keywords:
        # IDF is computed over the full corpus so term rarity is global, not
        # relative to the post-filter subset.
        idf, _ = build_idf(all_pages)
        for p in pages:
            p["_score"] = score_keyword_match(
                p["fm"], p["body"], keywords, idf=idf, ptype=p.get("_ptype", "unknown"))
            p["_snippet"] = extract_snippet(p["body"], keywords)
        pages = [p for p in pages if p["_score"] > 0]
        pages.sort(key=lambda x: (-x["_score"], x["path"]))
    else:
        pages.sort(key=lambda x: x["path"])

    pages = pages[:args.limit]

    if args.paths_only:
        for p in pages:
            print(p["path"])
        return

    if not pages:
        print("No matching pages.")
        return

    print(f"# {len(pages)} result(s)")
    print()
    for p in pages:
        print(format_result(p, compact=args.compact))
        if not args.compact:
            print()


if __name__ == "__main__":
    configure_utf8_stdio()
    main()
