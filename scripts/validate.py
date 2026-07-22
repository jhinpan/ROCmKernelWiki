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

from _wiki_root import WIKI_ROOT
from _scope import (
    in_scope_architectures,
    is_active,
    quarantined_architectures,
    quarantined_pages,
)

REPO_ROOT = WIKI_ROOT
SOURCES_DIR = REPO_ROOT / "sources"
WIKI_DIR = REPO_ROOT / "wiki"
DATA_DIR = REPO_ROOT / "data"

REPRO_ORDER = ["concept", "pseudocode", "snippet", "runnable", "benchmarked"]
REPRO_FLOOR = {"wiki-technique", "wiki-kernel", "wiki-language"}


def load_yaml(path):
    with open(path, encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def extract_frontmatter(filepath):
    with open(filepath, encoding="utf-8-sig") as f:
        content = f.read()
    m = re.match(r'^---\s*\r?\n(.*?)\r?\n---\s*\r?\n', content, re.DOTALL)
    if not m:
        return None
    try:
        return yaml.safe_load(m.group(1))
    except yaml.YAMLError as e:
        return {"_parse_error": str(e)}


def read_body(filepath):
    with open(filepath, encoding="utf-8-sig") as f:
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


def evidence_requirements_satisfied(evidence_types, requirements):
    for requirement in requirements:
        if isinstance(requirement, str):
            if requirement not in evidence_types:
                return False
        elif isinstance(requirement, dict) and "one_of" in requirement:
            if not (set(requirement["one_of"]) & evidence_types):
                return False
        else:
            raise ValueError(f"invalid verified evidence requirement: {requirement!r}")
    return True


def body_relative_links(body):
    for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", body):
        target = target.strip().split(maxsplit=1)[0].strip("<>")
        if (
            not target
            or target.startswith(("#", "/", "http://", "https://", "mailto:"))
        ):
            continue
        path_part = target.split("#", 1)[0]
        if "/" not in path_part and not Path(path_part).suffix:
            continue
        yield target


def main():
    schemas = load_yaml(DATA_DIR / "schemas.yaml")
    tags = load_yaml(DATA_DIR / "tags.yaml")
    evidence_policy = load_yaml(DATA_DIR / "evidence-policy.yaml")
    allowed_evidence = evidence_policy["allowed_source_categories"]

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
    ids_by_path = {}
    pages_by_id = {}
    referenced_ids = set()
    active_referenced_ids = set()
    version_refs = []
    stale = []
    files = []

    # cutoff + version-claims registry for freshness/version checks
    cutoff_date = "9999-99-99"
    cutoff_path = DATA_DIR / "refresh-cutoff.yaml"
    if cutoff_path.exists():
        try:
            cutoff_date = str(load_yaml(cutoff_path).get("cutoff_date", cutoff_date))
        except Exception:
            pass
    version_claim_ids = set()
    vc_path = DATA_DIR / "version-claims.yaml"
    if vc_path.exists():
        try:
            for c in (load_yaml(vc_path).get("claims") or []):
                if isinstance(c, dict) and c.get("id"):
                    version_claim_ids.add(c["id"])
        except Exception:
            pass

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
            ids_by_path[rel.as_posix()] = pid
            pages_by_id[pid] = fm
        page_records.append((md, rel, fm, ptype))

    for md, rel, fm, ptype in page_records:
        schema = schemas.get(ptype)
        if not schema:
            errors.append(f"{rel}: no schema for page type '{ptype}'")
            continue

        # List-valued fields that may legitimately be empty on PR pages
        # (a PR can touch kernels without us inferring every facet). The key
        # must still be present; an empty list satisfies "required".
        EMPTY_OK = {"architectures", "techniques", "hardware_features", "kernel_types",
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

        if ptype.startswith("wiki-") and is_active(fm):
            outside = set(fm.get("architectures") or []) - set(
                in_scope_architectures()
            )
            if outside:
                errors.append(
                    f"{rel}: active wiki page declares out-of-scope "
                    f"architectures {sorted(outside)}"
                )

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
                    source_id = pc.get("source_id")
                    if source_id and source_id not in pages_by_id:
                        errors.append(
                            f"{rel}: performance_claims[{i}] source_id "
                            f"'{source_id}' does not resolve"
                        )

        # verified evidence
        valid_evidence_types = set()
        for index, evidence in enumerate(fm.get("evidence_basis") or []):
            if not isinstance(evidence, dict):
                errors.append(f"{rel}: evidence_basis[{index}] must be an object")
                continue
            source_id = evidence.get("source_id")
            evidence_type = evidence.get("evidence_type")
            source = pages_by_id.get(source_id)
            if source is None:
                errors.append(
                    f"{rel}: evidence_basis[{index}] source_id "
                    f"'{source_id}' does not resolve"
                )
                continue
            if evidence_type not in allowed_evidence:
                errors.append(
                    f"{rel}: unknown evidence_type '{evidence_type}' in "
                    f"evidence_basis[{index}]"
                )
                continue
            allowed = set(allowed_evidence[evidence_type])
            source_category = source.get("source_category")
            if source_category not in allowed:
                errors.append(
                    f"{rel}: evidence_type '{evidence_type}' is incompatible "
                    f"with source '{source_id}' category '{source_category}'"
                )
                continue
            valid_evidence_types.add(evidence_type)

        if fm.get("confidence") == "verified":
            requirements = evidence_policy.get("verified_requires") or []
            if not evidence_requirements_satisfied(valid_evidence_types, requirements):
                errors.append(
                    f"{rel}: confidence=verified does not satisfy "
                    f"data/evidence-policy.yaml requirements {requirements}"
                )

        # collect references
        for ref in (fm.get("sources") or []):
            referenced_ids.add((str(rel), ref))
            if ptype.startswith("wiki-") and is_active(fm):
                active_referenced_ids.add((str(rel), ref))
        for ref in (fm.get("related") or []):
            referenced_ids.add((str(rel), ref))
            if ptype.startswith("wiki-") and is_active(fm):
                active_referenced_ids.add((str(rel), ref))
        for ref in (fm.get("candidate_techniques") or []):
            referenced_ids.add((str(rel), ref))
            if ptype.startswith("wiki-") and is_active(fm):
                active_referenced_ids.add((str(rel), ref))
        for ref in (fm.get("implemented_by") or []):
            referenced_ids.add((str(rel), ref))
            if ptype.startswith("wiki-") and is_active(fm):
                active_referenced_ids.add((str(rel), ref))

        # version-sensitive claim pointers (validated against version-claims.yaml)
        vs = fm.get("version_sensitive")
        if vs:
            vs_list = vs if isinstance(vs, list) else [vs]
            for vid in vs_list:
                version_refs.append((str(rel), vid))

        # freshness: PR merge date must not exceed the declared cutoff
        if ptype == "source-pr" and fm.get("date"):
            if str(fm["date"]) > cutoff_date:
                stale.append((str(rel), str(fm["date"])))

        for target in body_relative_links(read_body(md)):
            relative_target = target.split("#", 1)[0]
            if not relative_target:
                continue
            destination = (md.parent / relative_target).resolve()
            if not destination.exists():
                errors.append(f"{rel}: dangling body link '{target}'")
                continue
            try:
                destination_relative = destination.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                continue
            destination_id = ids_by_path.get(destination_relative)
            if (
                ptype.startswith("wiki-")
                and is_active(fm)
                and destination_id in pages_by_id
                and not is_active(pages_by_id[destination_id])
            ):
                errors.append(
                    f"{rel}: active body links to inactive page "
                    f"'{destination_id}'"
                )

    # link integrity
    id_set = set(all_ids)
    for src, ref in sorted(referenced_ids):
        if ref not in id_set:
            errors.append(f"{src}: dangling reference '{ref}' (no page with that id)")
    for page_id in sorted(quarantined_pages()):
        if page_id not in id_set:
            errors.append(f"data/scope.yaml: quarantined page '{page_id}' does not exist")
    for src, ref in sorted(active_referenced_ids):
        if ref in pages_by_id and not is_active(pages_by_id[ref]):
            errors.append(f"{src}: active page references inactive page '{ref}'")

    scope_architectures = (
        set(in_scope_architectures()) | set(quarantined_architectures())
    )
    unknown_scope_architectures = scope_architectures - vocab["architectures"]
    if unknown_scope_architectures:
        errors.append(
            "data/scope.yaml: unknown architectures "
            f"{sorted(unknown_scope_architectures)}"
        )

    # version_sensitive pointers must resolve to data/version-claims.yaml
    for src, vid in sorted(version_refs):
        if vid not in version_claim_ids:
            errors.append(f"{src}: version_sensitive '{vid}' not in data/version-claims.yaml")

    # freshness: PRs merged after the declared cutoff are a warning, not an error
    # (data is still valid; it signals the cutoff file should be advanced)
    if stale:
        warnings.append(f"{len(stale)} PR page(s) have merge dates after cutoff "
                        f"{cutoff_date} — advance data/refresh-cutoff.yaml "
                        f"(e.g. {stale[0][0]} = {stale[0][1]})")

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
