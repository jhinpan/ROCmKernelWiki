"""Load and validate the allowlisted evolution source registry."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

SOURCE_KINDS = {"github-prs", "github-tree"}
TRUST_LEVELS = {"first-party", "ecosystem", "community-reviewed"}


def load_registry(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if document.get("schema_version") != 1:
        raise ValueError(f"{path}: schema_version must be 1")
    raw_sources = document.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"{path}: sources must be a non-empty list")

    defaults = document.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError(f"{path}: defaults must be an object")

    seen_ids: set[str] = set()
    sources: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: sources[{index}] must be an object")
        source = deepcopy(raw)
        for field in ("id", "kind", "repo", "short", "trust", "license"):
            if not source.get(field):
                raise ValueError(f"{path}: sources[{index}] missing {field}")
        source_id = str(source["id"])
        if source_id in seen_ids:
            raise ValueError(f"{path}: duplicate source id {source_id}")
        seen_ids.add(source_id)
        if source["kind"] not in SOURCE_KINDS:
            raise ValueError(f"{path}: {source_id} has invalid kind {source['kind']}")
        if source["trust"] not in TRUST_LEVELS:
            raise ValueError(
                f"{path}: {source_id} has invalid trust {source['trust']}"
            )
        if "/" not in str(source["repo"]):
            raise ValueError(f"{path}: {source_id} repo must be owner/name")
        for field in ("include_paths", "include_keywords", "default_languages"):
            value = source.get(field, [])
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item for item in value
            ):
                raise ValueError(f"{path}: {source_id}.{field} must be strings")
            source[field] = value
        component_paths = source.get("component_paths") or {}
        if not isinstance(component_paths, dict):
            raise ValueError(f"{path}: {source_id}.component_paths must be an object")
        for component, patterns in component_paths.items():
            if not isinstance(component, str) or not isinstance(patterns, list):
                raise ValueError(
                    f"{path}: {source_id}.component_paths is malformed"
                )
        source["component_paths"] = component_paths
        source["active"] = bool(source.get("active", True))
        source["require_positive_path_signal"] = bool(
            source.get("require_positive_path_signal", False)
        )
        source["require_rocm_marker"] = bool(
            source.get("require_rocm_marker", False)
        )
        sources.append(source)

    return {
        "schema_version": 1,
        "defaults": deepcopy(defaults),
        "sources": sources,
    }


def active_sources(registry: dict[str, Any]) -> list[dict[str, Any]]:
    return [source for source in registry["sources"] if source.get("active", True)]


def source_by_id(registry: dict[str, Any], source_id: str) -> dict[str, Any]:
    for source in registry["sources"]:
        if source["id"] == source_id:
            return source
    raise KeyError(source_id)
