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
        identity = f"uid-{getuid()}" if getuid else str(Path.home())
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
    latest = max((path.stat().st_mtime_ns for path in files), default=0)
    return f"v1:{len(files)}:{latest}"


def _extract_id(path):
    try:
        with path.open(encoding="utf-8") as stream:
            prefix = stream.read(16 * 1024)
    except OSError:
        return None
    match = _FRONTMATTER.match(prefix)
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
    if _MEMORY_CACHE is not None:
        return _MEMORY_CACHE

    files = _markdown_files()
    signature = _signature(files)
    cache_path = _cache_path()
    if use_cache and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("signature") == signature:
                _MEMORY_CACHE = cached["ids"]
                return _MEMORY_CACHE
        except (OSError, ValueError, TypeError):
            pass

    ids = {}
    for path in files:
        page_id = _extract_id(path)
        if page_id:
            ids[page_id] = path.relative_to(WIKI_ROOT).as_posix()
    _MEMORY_CACHE = ids

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
