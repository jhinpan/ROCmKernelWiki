"""Active architecture-scope policy shared by query and maintenance tools."""

from hashlib import sha256

import yaml

from _wiki_root import WIKI_ROOT


_SCOPE_CACHE = None


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
    if not active:
        raise ValueError("data/scope.yaml must declare in_scope_architectures")
    if active & quarantined_architectures:
        raise ValueError("active and quarantined architectures must be disjoint")
    result = (active, quarantined_architectures, quarantined_pages)
    _SCOPE_CACHE = (signature, result)
    return result


def in_scope_architectures():
    return load_scope()[0]


def quarantined_architectures():
    return load_scope()[1]


def quarantined_pages():
    return load_scope()[2]


def is_active(frontmatter):
    """Return whether a page belongs to the default published knowledge layer."""
    if str(frontmatter.get("id", "")) in quarantined_pages():
        return False
    architectures = {str(v) for v in (frontmatter.get("architectures") or [])}
    return not architectures or bool(architectures & in_scope_architectures())


def scope_signature():
    path = WIKI_ROOT / "data" / "scope.yaml"
    return sha256(path.read_bytes()).hexdigest()[:16]
