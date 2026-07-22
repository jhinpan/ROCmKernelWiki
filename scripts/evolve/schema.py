"""Small schema validator for evolution candidates and proposals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_evolution_schemas(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if document.get("schema_version") != 1:
        raise ValueError(f"{path}: schema_version must be 1")
    return document


def _validate(
    value: dict[str, Any],
    schema_name: str,
    schemas_path: Path,
) -> list[str]:
    schemas = load_evolution_schemas(schemas_path)
    schema = schemas.get(schema_name)
    if not isinstance(schema, dict):
        raise ValueError(f"{schemas_path}: missing {schema_name} schema")
    errors = []
    for field in schema.get("required") or []:
        if field not in value or value[field] in (None, ""):
            errors.append(f"missing required field '{field}'")
    for field, allowed in (schema.get("enums") or {}).items():
        if field in value and value[field] not in allowed:
            errors.append(
                f"field '{field}' has invalid value {value[field]!r}; "
                f"expected one of {allowed}"
            )
    fingerprint = str(value.get("source_fingerprint") or "")
    if schema_name == "candidate" and not fingerprint.startswith("sha256:"):
        errors.append("source_fingerprint must start with sha256:")
    architectures = value.get("architectures")
    if schema_name == "candidate" and not isinstance(architectures, list):
        errors.append("architectures must be a list")
    return errors


def validate_candidate(candidate: dict[str, Any], schemas_path: Path) -> list[str]:
    return _validate(candidate, "candidate", schemas_path)


def validate_proposal(proposal: dict[str, Any], schemas_path: Path) -> list[str]:
    return _validate(proposal, "proposal", schemas_path)
