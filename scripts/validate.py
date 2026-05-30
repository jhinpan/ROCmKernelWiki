#!/usr/bin/env python3
"""Validate YAML frontmatter in all source and wiki pages against schemas.

Checks:
  - Required fields present per page type (data/schemas.yaml)
  - Controlled-vocabulary fields only use known values (data/tags.yaml)
  - source_category / status constraints
  - Reproducibility floor for technique/kernel/language pages (>= snippet)
  - wiki-kernel pages carry performance_claims with required sub-fields
  - Internal `sources:` and `related:` references resolve to real page ids
  - verified pages carry official-doc + upstream-code evidence_basis

Exit code 0 on success, 1 on any error.
"""

import re
import sys
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES_DIR = REPO_ROOT / "sources"
WIKI_DIR = REPO_ROOT / "wiki"
DATA_DIR = REPO_ROOT / "data"

REPRO_ORDER = ["concept", "pseudocode", "snippet", "runnable", "benchmarked"]
REPRO_FLOOR = {"wiki-technique", "wiki-kernel", "wiki-language"}


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_frontmatter(filepath):
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    m = re.match(r'^---\s*\r?\n(.*?)\r?\n---\s*\r?\n', content, re.DOTALL)
    if not m:
        return None
    try:
        return yaml.safe_load(m.group(1))
    except yaml.YAMLError as e:
        return {"_parse_error": str(e)}


def read_body(filepath):
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    m = re.match(r'^---\s*\r?\n.*?\r?\n---\s*\r?\n', content, re.DOTALL)
    return content[m.end():] if m else content


def detect_page_type(filepath, fm):
    rel = filepath.relative_to(REPO_ROOT)
    parts = rel.parts
    if parts[0] == "sources":
        return {
            "prs": "source-pr", "docs": "source-doc",
            "blogs": "source-blog", "refs": "source-ref",
        }.get(parts[1], "unknown")
    if parts[0] == "wiki":
        t = fm.get("type", "")
        if t:
            return f"wiki-{t}"
        return {
            "hardware": "wiki-hardware", "techniques": "wiki-technique",
            "patterns": "wiki-pattern", "kernels": "wiki-kernel",
            "languages": "wiki-language", "migration": "wiki-migration",
        }.get(parts[1] if len(parts) > 1 else "", "unknown")
    return "unknown"


def has_code_fence(body):
    return bool(re.search(r"```[a-zA-Z0-9_+-]*\s*\r?\n.*?\r?\n```", body, re.DOTALL))


def main():
    schemas = load_yaml(DATA_DIR / "schemas.yaml")
    tags = load_yaml(DATA_DIR / "tags.yaml")

    vocab = {
        "architectures": set(tags["architectures"]),
        "hardware_features": set(tags["hardware_features"]),
        "techniques": set(tags["techniques"]),
        "kernel_types": set(tags["kernel_types"]),
        "languages": set(tags["languages"]),
        "confidence": set(tags["confidence"]),
        "reproducibility": set(tags["reproducibility"]),
        "source_category": set(tags["source_categories"]),
    }
    # `tags` field draws from the union of all categorized vocab, plus the
    # diagnostic `symptoms` set and the free-form `misc_tags` set.
    all_tag_values = set()
    for v in vocab.values():
        all_tag_values |= v
    all_tag_values |= set(tags.get("symptoms", []))
    all_tag_values |= set(tags.get("misc_tags", []))

    errors = []
    warnings = []
    all_ids = {}
    referenced_ids = set()
    files = []

    for base in (SOURCES_DIR, WIKI_DIR):
        if not base.exists():
            continue
        for md in sorted(base.rglob("*.md")):
            files.append(md)

    page_records = []
    for md in files:
        rel = md.relative_to(REPO_ROOT)
        fm = extract_frontmatter(md)
        if fm is None:
            errors.append(f"{rel}: missing YAML frontmatter")
            continue
        if "_parse_error" in fm:
            errors.append(f"{rel}: YAML parse error: {fm['_parse_error']}")
            continue
        ptype = detect_page_type(md, fm)
        if ptype == "unknown":
            errors.append(f"{rel}: cannot determine page type")
            continue
        pid = fm.get("id")
        if pid:
            if pid in all_ids:
                errors.append(f"{rel}: duplicate id '{pid}' (also in {all_ids[pid]})")
            all_ids[pid] = str(rel)
        page_records.append((md, rel, fm, ptype))

    for md, rel, fm, ptype in page_records:
        schema = schemas.get(ptype)
        if not schema:
            errors.append(f"{rel}: no schema for page type '{ptype}'")
            continue

        # List-valued fields that may legitimately be empty on PR pages
        # (a PR can touch kernels without us inferring every facet). The key
        # must still be present; an empty list satisfies "required".
        EMPTY_OK = {"techniques", "hardware_features", "kernel_types",
                    "tags", "changed_paths"}
        for field in schema.get("required", []):
            present = field in fm
            empty = present and fm[field] in (None, "", [])
            if not present or (empty and field not in EMPTY_OK):
                errors.append(f"{rel}: missing required field '{field}' ({ptype})")

        # Controlled vocabulary checks
        for field in ("architectures", "hardware_features", "techniques",
                      "kernel_types", "languages"):
            for val in (fm.get(field) or []):
                if val not in vocab[field]:
                    errors.append(f"{rel}: '{val}' not in vocab '{field}'")
        for val in (fm.get("tags") or []):
            if val not in all_tag_values:
                errors.append(f"{rel}: tag '{val}' not in any vocab category")
        if "confidence" in fm and fm["confidence"] not in vocab["confidence"]:
            errors.append(f"{rel}: confidence '{fm['confidence']}' invalid")
        if "reproducibility" in fm and fm["reproducibility"] not in vocab["reproducibility"]:
            errors.append(f"{rel}: reproducibility '{fm['reproducibility']}' invalid")

        # source_category constraint
        constraints = schema.get("constraints", {})
        if "source_category" in constraints:
            allowed = constraints["source_category"]
            allowed = [allowed] if isinstance(allowed, str) else allowed
            if fm.get("source_category") not in allowed:
                errors.append(f"{rel}: source_category '{fm.get('source_category')}' "
                              f"not in {allowed}")
        if "status" in constraints and "status" in fm:
            if fm["status"] not in constraints["status"]:
                errors.append(f"{rel}: status '{fm['status']}' invalid")
        if ptype == "source-pr" and fm.get("status") == "merged" and not fm.get("merge_sha"):
            warnings.append(f"{rel}: merged PR without merge_sha")

        # Reproducibility floor + code snippet presence
        if ptype in REPRO_FLOOR:
            repro = fm.get("reproducibility")
            if repro in REPRO_ORDER and REPRO_ORDER.index(repro) < REPRO_ORDER.index("snippet"):
                errors.append(f"{rel}: reproducibility '{repro}' below floor 'snippet' for {ptype}")
            if not has_code_fence(read_body(md)):
                errors.append(f"{rel}: {ptype} page has no fenced code block (snippet required)")

        # wiki-kernel performance_claims
        if ptype == "wiki-kernel":
            pcs = fm.get("performance_claims")
            if not isinstance(pcs, list) or not pcs:
                errors.append(f"{rel}: wiki-kernel requires non-empty performance_claims")
            else:
                for i, pc in enumerate(pcs):
                    for sub in ("gpu", "dtype", "metric", "value", "source_id"):
                        if sub not in pc:
                            errors.append(f"{rel}: performance_claims[{i}] missing '{sub}'")

        # verified evidence
        if fm.get("confidence") == "verified":
            eb = fm.get("evidence_basis") or []
            types = {e.get("evidence_type") for e in eb if isinstance(e, dict)}
            if not ({"official-doc"} & types and {"upstream-code", "paper"} & types):
                errors.append(f"{rel}: confidence=verified requires evidence_basis with "
                              f"official-doc + upstream-code/paper")

        # collect references
        for ref in (fm.get("sources") or []):
            referenced_ids.add((str(rel), ref))
        for ref in (fm.get("related") or []):
            referenced_ids.add((str(rel), ref))
        for ref in (fm.get("candidate_techniques") or []):
            referenced_ids.add((str(rel), ref))

    # link integrity
    id_set = set(all_ids)
    for src, ref in sorted(referenced_ids):
        if ref not in id_set:
            errors.append(f"{src}: dangling reference '{ref}' (no page with that id)")

    n_pages = len(page_records)
    print(f"Validated {n_pages} pages, {len(all_ids)} unique ids.")
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings[:50]:
            print(f"  WARN  {w}")
        if len(warnings) > 50:
            print(f"  ... and {len(warnings) - 50} more")
    if errors:
        print(f"\n{len(errors)} ERROR(s):")
        for e in errors[:100]:
            print(f"  ERR   {e}")
        if len(errors) > 100:
            print(f"  ... and {len(errors) - 100} more")
        sys.exit(1)
    print("\nOK — 0 errors.")


if __name__ == "__main__":
    main()
