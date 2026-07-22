"""Active architecture-scope policy shared by query and maintenance tools."""

import re
from hashlib import sha256

import yaml

from _wiki_root import WIKI_ROOT


_SCOPE_CACHE = None
_GFX_TOKEN = re.compile(
    r"(?<![a-z0-9])gfx\s*[0-9][0-9a-z*]*(?![a-z0-9])",
    re.IGNORECASE,
)
_MI_TOKEN = re.compile(
    r"(?<![a-z0-9])mi\s*[0-9][0-9a-z]*(?![a-z0-9])",
    re.IGNORECASE,
)
_CDNA_TOKEN = re.compile(
    r"(?<![a-z0-9])cdna[0-9]+(?![a-z0-9])",
    re.IGNORECASE,
)
_GCN_VERSION = re.compile(
    r"(?<![a-z0-9])gcn\s*[0-9]+(?![a-z0-9])",
    re.IGNORECASE,
)
_UNSUPPORTED_FAMILY = re.compile(
    r"(?<![a-z0-9])(?:rdna|navi|vega)(?![a-z])|"
    r"(?<![a-z0-9])(?:arcturus|aldebaran|fiji|hawaii|radeon|"
    r"strix\s+halo|krackan)(?![a-z0-9])",
    re.IGNORECASE,
)


def load_scope():
    global _SCOPE_CACHE
    path = WIKI_ROOT / "data" / "scope.yaml"
    signature = scope_signature()
    if _SCOPE_CACHE is not None and _SCOPE_CACHE[0] == signature:
        return _SCOPE_CACHE[1]
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    active = frozenset(str(v) for v in raw.get("in_scope_architectures", []))
    quarantined_architectures = frozenset(
        str(v) for v in raw.get("quarantined_architectures", [])
    )
    quarantined_pages = frozenset(str(v) for v in raw.get("quarantined_pages", []))
    quarantined_query_terms = frozenset(
        str(v).lower() for v in raw.get("quarantined_query_terms", [])
    )
    aliases = yaml.safe_load(
        (WIKI_ROOT / "data" / "aliases.yaml").read_text(encoding="utf-8")
    ) or {}
    active_aliases = frozenset(
        str(alias).lower()
        for architecture in active
        for alias in [architecture, *(aliases.get(architecture) or [])]
    )
    if not active:
        raise ValueError("data/scope.yaml must declare in_scope_architectures")
    if active & quarantined_architectures:
        raise ValueError("active and quarantined architectures must be disjoint")
    result = (
        active,
        quarantined_architectures,
        quarantined_pages,
        quarantined_query_terms,
        active_aliases,
    )
    _SCOPE_CACHE = (signature, result)
    return result


def in_scope_architectures():
    return load_scope()[0]


def quarantined_architectures():
    return load_scope()[1]


def quarantined_pages():
    return load_scope()[2]


def quarantined_query_terms():
    return load_scope()[3]


def active_architecture_aliases():
    return load_scope()[4]


def _upstream_scope_text(frontmatter):
    return " ".join(
        [
            str(frontmatter.get("title", "")),
            str(frontmatter.get("inclusion_reason", "")),
            *(str(path) for path in (frontmatter.get("changed_paths") or [])),
        ]
    )


def is_active(frontmatter):
    """Return whether a page belongs to the default published knowledge layer."""
    if str(frontmatter.get("id", "")) in quarantined_pages():
        return False
    if str(frontmatter.get("scope_status", "active")) != "active":
        return False
    architectures = {str(v) for v in (frontmatter.get("architectures") or [])}
    if frontmatter.get("source_category") == "upstream-code":
        # A mixed-architecture PR is retained as raw evidence but excluded from
        # active retrieval: its change cannot be assumed safe for gfx942/gfx950.
        active = in_scope_architectures()
        if architectures - active:
            return False
        scope_text = _upstream_scope_text(frontmatter)
        explicit_gfx = {
            re.sub(r"\s+", "", match.group(0).lower()).rstrip("*x")
            for match in _GFX_TOKEN.finditer(scope_text)
        }
        aliases = active_architecture_aliases()
        active_mi = {
            re.sub(r"\s+", "", alias)
            for alias in aliases
            if _MI_TOKEN.fullmatch(alias)
        }
        explicit_mi = {
            re.sub(r"\s+", "", match.group(0).lower())
            for match in _MI_TOKEN.finditer(scope_text)
        }
        explicit_mi = {
            token for token in explicit_mi if not re.search(r"x[0-9]", token)
        }
        active_cdna = {alias for alias in aliases if _CDNA_TOKEN.fullmatch(alias)}
        explicit_cdna = {
            match.group(0).lower() for match in _CDNA_TOKEN.finditer(scope_text)
        }
        if (
            explicit_gfx - active
            or explicit_mi - active_mi
            or explicit_cdna - active_cdna
            or _GCN_VERSION.search(scope_text)
            or _UNSUPPORTED_FAMILY.search(scope_text)
        ):
            return False
    return not architectures or bool(architectures & in_scope_architectures())


def scope_signature():
    digest = sha256()
    for name in ("scope.yaml", "aliases.yaml"):
        digest.update((WIKI_ROOT / "data" / name).read_bytes())
    return digest.hexdigest()[:16]
