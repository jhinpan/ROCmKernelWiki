"""Shared persistent id index for O(1) page lookup."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import yaml

from _wiki_root import WIKI_ROOT

_FRONTMATTER = re.compile(r"^---\s*\r?\n(.*?)\r?\n---", re.DOTALL)
_MEMORY_CACHE = None


def _cache_path():
    configured = os.environ.get("ROCM_WIKI_CACHE_DIR")
    if configured:
        root = Path(configured).expanduser()
    else:
        getuid = getattr(os, "getuid", None)
        if getuid is not None:
            identity = f"uid-{getuid()}"
        else:
            identity = hashlib.sha256(str(Path.home()).encode("utf-8")).hexdigest()[:12]
        root = Path(tempfile.gettempdir()) / f"rocm-kernel-wiki-{identity}"
    root_key = hashlib.sha256(str(WIKI_ROOT).encode("utf-8")).hexdigest()[:16]
    return root / root_key / "id-index.json"


def _markdown_files():
    files = []
    for subdir in ("sources", "wiki"):
        base = WIKI_ROOT / subdir
        if base.exists():
            files.extend(base.rglob("*.md"))
    return files


def _signature(files):
    digest = hashlib.sha256()
    for path in sorted(files):
        stat = path.stat()
        digest.update(path.relative_to(WIKI_ROOT).as_posix().encode("utf-8"))
        digest.update(f":{stat.st_size}:{stat.st_mtime_ns}\n".encode("ascii"))
    return f"v1:{digest.hexdigest()}"


def _extract_id(path):
    try:
        content = path.read_text(encoding="utf-8-sig")
    except OSError:
        return None
    match = _FRONTMATTER.match(content)
    if not match:
        return None
    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None
    if not isinstance(frontmatter, dict) or not frontmatter.get("id"):
        return None
    return str(frontmatter["id"])


def id_index(use_cache=True):
    global _MEMORY_CACHE
    files = _markdown_files()
    signature = _signature(files)
    if use_cache and _MEMORY_CACHE is not None and _MEMORY_CACHE[0] == signature:
        return _MEMORY_CACHE[1]

    cache_path = _cache_path()
    if use_cache and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("signature") == signature:
                ids = cached["ids"]
                _MEMORY_CACHE = (signature, ids)
                return ids
        except (OSError, ValueError, TypeError):
            pass

    ids = {}
    for path in files:
        page_id = _extract_id(path)
        if page_id:
            if page_id in ids:
                raise ValueError(
                    f"duplicate page id {page_id!r}: {ids[page_id]} and "
                    f"{path.relative_to(WIKI_ROOT).as_posix()}"
                )
            ids[page_id] = path.relative_to(WIKI_ROOT).as_posix()
    _MEMORY_CACHE = (signature, ids)

    if use_cache:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"signature": signature, "ids": ids}
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=cache_path.parent,
                prefix=f".{cache_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                json.dump(payload, stream, sort_keys=True)
                temporary = Path(stream.name)
            temporary.replace(cache_path)
        except OSError:
            pass
    return ids
