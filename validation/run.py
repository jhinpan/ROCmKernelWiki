#!/usr/bin/env python3
"""Reproducible gfx950 validation using only Python's stdlib and hipcc."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONTROLLER_PATH = Path(__file__).resolve()
VALIDATION_DIR = Path(
    os.environ.get("ROCM_WIKI_VALIDATION_PAYLOAD", CONTROLLER_PATH.parent)
).resolve()
REPOSITORY_DIR = Path(
    os.environ.get("ROCM_WIKI_REPOSITORY_DIR", VALIDATION_DIR.parent)
).resolve()
HARNESS_MANIFEST = VALIDATION_DIR / "manifest.json"
EVIDENCE_KINDS = {"hardware", "runtime", "compiler", "source-reported"}
VERDICT_STATUSES = {"pass", "fail", "recorded"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()


def parse_key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def normalize_text_output(text: str) -> str:
    """Keep textual evidence stable and Git-friendly without changing content."""
    normalized = "\n".join(line.rstrip() for line in text.splitlines())
    return normalized + ("\n" if text.endswith(("\n", "\r")) else "")


def load_and_validate_manifest() -> dict[str, Any]:
    manifest = json.loads(HARNESS_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    if manifest.get("device", {}).get("ordinal") != 0:
        raise ValueError("manifest must pin device ordinal 0")

    architectures = manifest.get("architectures", {})
    expected_architectures = {"gfx950", "gfx942"}
    if set(architectures) != expected_architectures:
        raise ValueError("manifest architectures must be gfx950 and gfx942")
    if architectures["gfx950"].get("mode") != "compile-and-run":
        raise ValueError("gfx950 must be compile-and-run")
    if architectures["gfx942"].get("mode") != "compile-only":
        raise ValueError("gfx942 must be compile-only")

    referenced_sources = [
        manifest["direct_lds"]["source"],
        manifest["direct_lds"]["runtime"]["source"],
        "probes/device_properties.hip",
    ]
    capability_ids: set[str] = set()
    for capability in manifest.get("compiler_capabilities", []):
        capability_id = capability["id"]
        if capability_id in capability_ids:
            raise ValueError(f"duplicate capability id: {capability_id}")
        capability_ids.add(capability_id)
        if set(capability["expected_accepted"]) != expected_architectures:
            raise ValueError(
                f"{capability_id}: expected_accepted must cover both architectures"
            )
        referenced_sources.append(capability["source"])

    widths = set(manifest["direct_lds"]["sizes_bytes"])
    for arch, accepted in manifest["direct_lds"]["expected_accepted"].items():
        if arch not in expected_architectures:
            raise ValueError(f"unexpected direct-LDS architecture: {arch}")
        if not set(accepted) <= widths:
            raise ValueError(f"{arch}: accepted widths must be in sizes_bytes")

    for relative in referenced_sources:
        source = (VALIDATION_DIR / relative).resolve()
        try:
            source.relative_to(VALIDATION_DIR)
        except ValueError as error:
            raise ValueError(f"probe leaves validation/: {relative}") from error
        if not source.is_file():
            raise ValueError(f"missing probe: {relative}")
    return manifest


class Harness:
    def __init__(
        self,
        specification: dict[str, Any],
        output_dir: Path,
        work_dir: Path,
        hipcc: Path,
        timeout_seconds: int,
        check_mode: bool,
    ) -> None:
        self.specification = specification
        self.output_dir = output_dir
        self.work_dir = work_dir
        self.hipcc = hipcc
        self.timeout_seconds = timeout_seconds
        self.check_mode = check_mode
        self.started_at = utc_now()
        self.finished_at = ""
        self.commands: list[dict[str, Any]] = []
        self.verdicts: list[dict[str, Any]] = []
        self.compilations: dict[tuple[str, str], dict[str, Any]] = {}
        self.rocminfo_result: dict[str, Any] | None = None
        self.git_commit: str | None = None
        self.git_branch: str | None = None
        self.toolchain: dict[str, Any] = {"hipcc_path": str(hipcc)}
        self.toolchain_label = "hipcc version unavailable"
        self.environment_overrides = {
            "ROCR_VISIBLE_DEVICES": "0",
            "HIP_VISIBLE_DEVICES": "0",
            "CUDA_VISIBLE_DEVICES": "0",
            "LC_ALL": "C",
            "LANG": "C",
        }

        output_dir.mkdir(parents=True, exist_ok=False)
        for subdirectory in ("logs", "isa", "metadata"):
            (output_dir / subdirectory).mkdir()

    def relative_output_path(self, path: Path) -> str:
        return path.relative_to(self.output_dir).as_posix()

    def run_command(
        self,
        label: str,
        argv: list[str],
        *,
        cwd: Path = VALIDATION_DIR,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        index = len(self.commands) + 1
        stem = f"{index:02d}-{safe_name(label)}"
        stdout_path = self.output_dir / "logs" / f"{stem}.stdout.txt"
        stderr_path = self.output_dir / "logs" / f"{stem}.stderr.txt"
        environment = os.environ.copy()
        environment.update(self.environment_overrides)
        started_at = utc_now()
        start = time.monotonic()
        timed_out = False

        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds or self.timeout_seconds,
                check=False,
            )
            returncode = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as error:
            timed_out = True
            returncode = 124
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            stderr += (
                f"\ncommand timed out after "
                f"{timeout_seconds or self.timeout_seconds} seconds\n"
            )
        except OSError as error:
            returncode = 127
            stdout = ""
            stderr = f"{type(error).__name__}: {error}\n"

        duration_seconds = round(time.monotonic() - start, 6)
        stdout = normalize_text_output(stdout)
        stderr = normalize_text_output(stderr)
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        record = {
            "index": index,
            "label": label,
            "argv": argv,
            "command": shlex.join(argv),
            "cwd": str(cwd),
            "environment_overrides": dict(self.environment_overrides),
            "started_at": started_at,
            "duration_seconds": duration_seconds,
            "returncode": returncode,
            "timed_out": timed_out,
            "stdout_path": self.relative_output_path(stdout_path),
            "stderr_path": self.relative_output_path(stderr_path),
        }
        self.commands.append(record)
        return {**record, "stdout": stdout, "stderr": stderr}

    @staticmethod
    def command_artifacts(result: dict[str, Any] | None) -> list[str]:
        if result is None:
            return []
        return [result["stdout_path"], result["stderr_path"]]

    def add_verdict(
        self,
        *,
        verdict_id: str,
        claim: str,
        arch: str,
        evidence_kind: str,
        status: str,
        artifact_paths: list[str],
        **details: Any,
    ) -> None:
        if evidence_kind not in EVIDENCE_KINDS:
            raise ValueError(f"invalid evidence kind: {evidence_kind}")
        if status not in VERDICT_STATUSES:
            raise ValueError(f"invalid verdict status: {status}")
        self.verdicts.append(
            {
                "id": verdict_id,
                "claim": claim,
                "arch": arch,
                "toolchain": self.toolchain_label,
                "evidence_kind": evidence_kind,
                "status": status,
                "artifact_paths": unique(artifact_paths),
                **details,
            }
        )

    def compile_assembly(
        self,
        label: str,
        source: str,
        arch: str,
        definitions: list[str] | None = None,
    ) -> dict[str, Any]:
        assembly_path = self.work_dir / f"{safe_name(label)}.s"
        argv = [
            str(self.hipcc),
            "--genco",
            "-S",
            "-std=c++17",
            "-O2",
            f"--offload-arch={arch}",
        ]
        argv.extend(f"-D{definition}" for definition in definitions or [])
        argv.extend([source, "-o", str(assembly_path)])
        command = self.run_command(f"compile-{label}", argv)
        success = command["returncode"] == 0 and assembly_path.is_file()
        isa_path: Path | None = None
        metadata_path: Path | None = None
        assembly = ""
        isa = ""
        metadata = ""

        if success:
            assembly = assembly_path.read_text(encoding="utf-8", errors="replace")
            marker = re.search(r"(?m)^[ \t]*\.amdgpu_metadata[ \t]*$", assembly)
            if marker:
                isa = assembly[: marker.start()]
                metadata = assembly[marker.start() :]
            else:
                isa = assembly
            isa_path = self.output_dir / "isa" / f"{safe_name(label)}.txt"
            metadata_path = (
                self.output_dir / "metadata" / f"{safe_name(label)}.txt"
            )
            isa_path.write_text(isa, encoding="utf-8")
            metadata_path.write_text(metadata, encoding="utf-8")

        artifacts = self.command_artifacts(command)
        if isa_path is not None:
            artifacts.append(self.relative_output_path(isa_path))
        if metadata_path is not None:
            artifacts.append(self.relative_output_path(metadata_path))
        return {
            "label": label,
            "source": source,
            "arch": arch,
            "success": success,
            "command": command,
            "assembly": assembly,
            "isa": isa,
            "metadata": metadata,
            "artifacts": artifacts,
        }

    def compile_executable(self, label: str, source: str, arch: str) -> dict[str, Any]:
        executable = self.work_dir / safe_name(label)
        command = self.run_command(
            f"build-{label}",
            [
                str(self.hipcc),
                "-std=c++17",
                "-O2",
                f"--offload-arch={arch}",
                source,
                "-o",
                str(executable),
            ],
        )
        return {
            "success": command["returncode"] == 0 and executable.is_file(),
            "executable": executable,
            "command": command,
            "artifacts": self.command_artifacts(command),
        }

    def preflight(self) -> None:
        hipcc_version = self.run_command(
            "hipcc-version", [str(self.hipcc), "--version"]
        )
        if hipcc_version["returncode"] != 0:
            raise RuntimeError("hipcc --version failed")
        version_text = (hipcc_version["stdout"] + hipcc_version["stderr"]).strip()
        version_lines = [line.strip() for line in version_text.splitlines() if line]
        self.toolchain_label = " | ".join(version_lines[:2])
        self.toolchain = {
            "hipcc_path": str(self.hipcc),
            "version": version_text,
            "artifact_paths": self.command_artifacts(hipcc_version),
        }

        rocminfo = shutil.which("rocminfo")
        if rocminfo:
            self.rocminfo_result = self.run_command(
                "rocminfo-device-0",
                [rocminfo],
                timeout_seconds=min(self.timeout_seconds, 30),
            )
            self.toolchain["rocminfo_path"] = rocminfo
            self.toolchain["rocminfo_artifact_paths"] = self.command_artifacts(
                self.rocminfo_result
            )

        git = shutil.which("git")
        if git:
            commit = self.run_command(
                "git-commit", [git, "rev-parse", "HEAD"], cwd=REPOSITORY_DIR
            )
            branch = self.run_command(
                "git-branch",
                [git, "branch", "--show-current"],
                cwd=REPOSITORY_DIR,
            )
            if commit["returncode"] == 0:
                self.git_commit = commit["stdout"].strip()
            if branch["returncode"] == 0:
                self.git_branch = branch["stdout"].strip()

    def record_source_claims(self) -> None:
        for source_claim in self.specification["source_reported_claims"]:
            self.add_verdict(
                verdict_id=source_claim["id"],
                claim=source_claim["claim"],
                arch=source_claim["arch"],
                evidence_kind="source-reported",
                status="recorded",
                artifact_paths=["manifest.json"],
                method=(
                    "Recorded from cited sources in the harness manifest; this "
                    "entry is not local hardware, runtime, or compiler evidence."
                ),
                sources=source_claim["sources"],
            )

    def validate_device_properties(self) -> dict[str, str]:
        arch = self.specification["device"]["runtime_arch"]
        build = self.compile_executable(
            "device-properties", "probes/device_properties.hip", arch
        )
        run: dict[str, Any] | None = None
        observed: dict[str, str] = {}
        if build["success"]:
            run = self.run_command(
                "run-device-properties", [str(build["executable"])]
            )
            if run["returncode"] == 0:
                observed = parse_key_values(run["stdout"])

        artifacts = list(build["artifacts"]) + self.command_artifacts(run)
        if self.rocminfo_result is not None:
            artifacts += self.command_artifacts(self.rocminfo_result)

        for property_spec in self.specification["device_properties"]:
            actual = observed.get(property_spec["field"])
            expected = property_spec["expected"]
            comparison = property_spec["comparison"]
            matched = False
            normalized_actual: Any = actual
            if actual is not None:
                if comparison == "integer":
                    try:
                        normalized_actual = int(actual)
                        matched = normalized_actual == int(expected)
                    except ValueError:
                        matched = False
                elif comparison == "contains":
                    matched = str(expected) in actual
                elif comparison == "prefix":
                    matched = actual.startswith(str(expected))
                else:
                    raise ValueError(f"unknown comparison: {comparison}")

            self.add_verdict(
                verdict_id=property_spec["id"],
                claim=property_spec["claim"],
                arch=arch,
                evidence_kind="hardware",
                status="pass" if matched else "fail",
                artifact_paths=artifacts,
                method=(
                    "hipGetDeviceProperties(0) under ROCR_VISIBLE_DEVICES=0 and "
                    "HIP_VISIBLE_DEVICES=0; waves/CU is maxThreadsPerCU / wave size."
                ),
                expected=expected,
                observed=normalized_actual,
                property=property_spec["field"],
            )
        return observed

    def validate_direct_lds_widths(self) -> None:
        direct_lds = self.specification["direct_lds"]
        expected_isa = direct_lds["expected_isa"]
        for arch in ("gfx950", "gfx942"):
            accepted_widths = set(direct_lds["expected_accepted"][arch])
            for width in direct_lds["sizes_bytes"]:
                label = f"direct-lds-{arch}-{width}b"
                compilation = self.compile_assembly(
                    label,
                    direct_lds["source"],
                    arch,
                    [f"COPY_BYTES={width}"],
                )
                expected_accepted = width in accepted_widths
                opcode = expected_isa.get(arch, {}).get(str(width))
                opcode_found = opcode is None or opcode in compilation["isa"]
                passed = (
                    compilation["success"] == expected_accepted
                    and (not expected_accepted or opcode_found)
                )
                expected_word = "accepts" if expected_accepted else "rejects"
                self.add_verdict(
                    verdict_id=f"compiler-direct-lds-{arch}-{width}b",
                    claim=(
                        f"The {arch} compiler {expected_word} {width}-byte "
                        "__builtin_amdgcn_load_to_lds transfers."
                    ),
                    arch=arch,
                    evidence_kind="compiler",
                    status="pass" if passed else "fail",
                    artifact_paths=compilation["artifacts"],
                    method=(
                        "hipcc --genco -S target compilation; gfx942 is never "
                        "executed by this harness."
                    ),
                    expected={
                        "accepted": expected_accepted,
                        "isa_contains": opcode,
                    },
                    observed={
                        "accepted": compilation["success"],
                        "isa_contains_expected": opcode_found
                        if compilation["success"]
                        else None,
                    },
                )

    def validate_compiler_capabilities(self) -> None:
        for capability in self.specification["compiler_capabilities"]:
            for arch in ("gfx950", "gfx942"):
                label = f"capability-{capability['id']}-{arch}"
                compilation = self.compile_assembly(
                    label, capability["source"], arch
                )
                self.compilations[(capability["id"], arch)] = compilation
                expected_accepted = capability["expected_accepted"][arch]
                opcode_found = (
                    capability["expected_isa"] in compilation["isa"]
                    if compilation["success"]
                    else False
                )
                passed = (
                    compilation["success"] == expected_accepted
                    and (not expected_accepted or opcode_found)
                )
                self.add_verdict(
                    verdict_id=f"compiler-{capability['id']}-{arch}",
                    claim=capability["claim"],
                    arch=arch,
                    evidence_kind="compiler",
                    status="pass" if passed else "fail",
                    artifact_paths=compilation["artifacts"],
                    method=(
                        "hipcc --genco -S target compilation followed by an "
                        "opcode check for accepted targets; gfx942 is compile-only."
                    ),
                    expected={
                        "accepted": expected_accepted,
                        "isa_contains": capability["expected_isa"]
                        if expected_accepted
                        else None,
                    },
                    observed={
                        "accepted": compilation["success"],
                        "isa_contains_expected": opcode_found
                        if compilation["success"]
                        else None,
                    },
                )

    def validate_hsa_metadata(self) -> None:
        metadata_spec = self.specification["metadata_probe"]
        key = (metadata_spec["capability_id"], metadata_spec["arch"])
        compilation = self.compilations.get(key)
        fields: dict[str, int | None] = {}
        if compilation is None:
            metadata = ""
            artifacts: list[str] = []
        else:
            metadata = compilation["metadata"]
            artifacts = compilation["artifacts"]

        for field in metadata_spec["required_fields"]:
            match = re.search(
                rf"(?m)^\s*(?:-\s*)?{re.escape(field)}:\s*(\d+)\s*$",
                metadata,
            )
            fields[field] = int(match.group(1)) if match else None
        passed = compilation is not None and compilation["success"] and all(
            value is not None for value in fields.values()
        )
        self.add_verdict(
            verdict_id="compiler-hsa-register-metadata",
            claim=metadata_spec["claim"],
            arch=metadata_spec["arch"],
            evidence_kind="compiler",
            status="pass" if passed else "fail",
            artifact_paths=artifacts,
            method=(
                "Extracted the .amdgpu_metadata block from hipcc's textual "
                "AMDGPU assembly. This checks field presence and values, not "
                "the semantic relationship between the two count namespaces."
            ),
            expected={"fields_present": metadata_spec["required_fields"]},
            observed={"fields": fields},
        )

    def validate_direct_lds_runtime(self) -> None:
        runtime_spec = self.specification["direct_lds"]["runtime"]
        arch = runtime_spec["arch"]
        assembly = self.compile_assembly(
            "direct-lds-runtime-gfx950", runtime_spec["source"], arch
        )
        build = self.compile_executable(
            "direct-lds-runtime-gfx950", runtime_spec["source"], arch
        )
        run: dict[str, Any] | None = None
        observed: dict[str, str] = {}
        if build["success"]:
            run = self.run_command(
                "run-direct-lds-runtime-gfx950", [str(build["executable"])]
            )
            observed = parse_key_values(run["stdout"])

        expected_opcode = runtime_spec["expected_isa"]
        opcode_found = expected_opcode in assembly["isa"]
        expected_values = {
            "copy_width_bytes": runtime_spec["width_bytes"],
            "wave_threads": 64,
            "payload_mismatches": 0,
            "lower_sentinel_mismatches": 0,
            "upper_sentinel_mismatches": 0,
        }
        normalized_observed: dict[str, int | None] = {}
        values_match = True
        for key, expected in expected_values.items():
            try:
                normalized_observed[key] = int(observed[key])
            except (KeyError, ValueError):
                normalized_observed[key] = None
            values_match &= normalized_observed[key] == expected

        passed = (
            assembly["success"]
            and opcode_found
            and build["success"]
            and run is not None
            and run["returncode"] == 0
            and values_match
        )
        artifacts = (
            assembly["artifacts"]
            + build["artifacts"]
            + self.command_artifacts(run)
        )
        self.add_verdict(
            verdict_id="runtime-direct-lds-pointer-contract",
            claim=(
                "A gfx950 16-byte direct-to-LDS copy with lane-varying global "
                "sources and one wave-uniform LDS base reproduces every payload "
                "word without modifying either adjacent LDS sentinel."
            ),
            arch=arch,
            evidence_kind="runtime",
            status="pass" if passed else "fail",
            artifact_paths=artifacts,
            method=(
                "One 64-lane wave executed on visible device 0; final ISA was "
                "checked for global_load_lds_dwordx4 and host code checked 256 "
                "payload dwords plus two four-dword LDS sentinels."
            ),
            expected={
                **expected_values,
                "isa_contains": expected_opcode,
            },
            observed={
                **normalized_observed,
                "isa_contains_expected": opcode_found,
                "returncode": run["returncode"] if run is not None else None,
            },
        )

    def execute(self) -> str:
        try:
            self.preflight()
            self.record_source_claims()
            self.validate_device_properties()
            self.validate_direct_lds_widths()
            self.validate_compiler_capabilities()
            self.validate_hsa_metadata()
            self.validate_direct_lds_runtime()
        except Exception:
            error_path = self.output_dir / "logs" / "harness-internal-error.txt"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            self.add_verdict(
                verdict_id="harness-internal-error",
                claim="The validation harness completes without an internal error.",
                arch="gfx950+gfx942",
                evidence_kind="runtime",
                status="fail",
                artifact_paths=[self.relative_output_path(error_path)],
                method="Python harness exception capture.",
            )
        return self.finalize()

    def write_commands(self) -> None:
        (self.output_dir / "commands.json").write_text(
            json.dumps(
                {"schema_version": 1, "commands": self.commands},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        lines: list[str] = []
        for command in self.commands:
            lines.extend(
                [
                    f"[{command['index']:02d}] {command['label']}",
                    f"cwd: {command['cwd']}",
                    "env: "
                    + " ".join(
                        f"{key}={shlex.quote(value)}"
                        for key, value in command["environment_overrides"].items()
                    ),
                    f"$ {command['command']}",
                    (
                        f"exit: {command['returncode']}  "
                        f"duration_seconds: {command['duration_seconds']}"
                    ),
                    f"stdout: {command['stdout_path']}",
                    f"stderr: {command['stderr_path']}",
                    "",
                ]
            )
        (self.output_dir / "commands.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    def source_hashes(self) -> dict[str, str]:
        paths = [Path("manifest.json")]
        paths.extend(
            path.relative_to(VALIDATION_DIR)
            for path in sorted((VALIDATION_DIR / "probes").glob("*.hip"))
        )
        readme = Path("README.md")
        if (VALIDATION_DIR / readme).is_file():
            paths.append(readme)
        hashes = {
            path.as_posix(): file_sha256(VALIDATION_DIR / path)
            for path in paths
        }
        hashes["controller/run.py"] = file_sha256(CONTROLLER_PATH)
        return hashes

    def output_validation_errors(self) -> list[str]:
        errors: list[str] = []
        required = {
            "claim",
            "arch",
            "toolchain",
            "evidence_kind",
            "status",
            "artifact_paths",
        }
        seen_ids: set[str] = set()
        for verdict in self.verdicts:
            missing = required - set(verdict)
            if missing:
                errors.append(f"{verdict.get('id')}: missing {sorted(missing)}")
            verdict_id = verdict.get("id")
            if verdict_id in seen_ids:
                errors.append(f"duplicate verdict id: {verdict_id}")
            seen_ids.add(verdict_id)
            if verdict.get("evidence_kind") not in EVIDENCE_KINDS:
                errors.append(f"{verdict_id}: invalid evidence kind")
            if verdict.get("status") not in VERDICT_STATUSES:
                errors.append(f"{verdict_id}: invalid status")
            for relative in verdict.get("artifact_paths", []):
                artifact = (self.output_dir / relative).resolve()
                try:
                    artifact.relative_to(self.output_dir.resolve())
                except ValueError:
                    errors.append(f"{verdict_id}: artifact leaves output: {relative}")
                    continue
                if not artifact.is_file():
                    errors.append(f"{verdict_id}: missing artifact: {relative}")
        return errors

    def write_run_manifest(self, overall_status: str) -> None:
        run_manifest = {
            "schema_version": 1,
            "harness_manifest_sha256": file_sha256(HARNESS_MANIFEST),
            "specification": self.specification,
            "run": {
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "overall_status": overall_status,
                "check_mode": self.check_mode,
                "output_directory": self.output_dir.name,
                "gpu_visibility": dict(self.environment_overrides),
                "runtime_device_ordinal": 0,
                "runtime_arches_executed": ["gfx950"],
                "compile_only_arches": ["gfx942"],
                "host": {
                    "platform": platform.platform(),
                    "python": sys.version.split()[0],
                },
                "repository": {
                    "path": ".",
                    "commit": self.git_commit,
                    "branch": self.git_branch,
                },
                "command_count": len(self.commands),
                "verdict_count": len(self.verdicts),
            },
            "toolchain": self.toolchain,
            "source_sha256": self.source_hashes(),
            "artifacts": {
                "commands_json": "commands.json",
                "commands_text": "commands.txt",
                "verdicts": "verdicts.json",
                "summary": "summary.txt",
            },
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def finalize(self) -> str:
        self.finished_at = utc_now()
        self.write_commands()
        provisional = (
            "fail"
            if any(verdict["status"] == "fail" for verdict in self.verdicts)
            else "pass"
        )
        self.write_run_manifest(provisional)

        validation_errors = self.output_validation_errors()
        if validation_errors:
            error_path = self.output_dir / "logs" / "output-validation.txt"
            error_path.write_text(
                "\n".join(validation_errors) + "\n", encoding="utf-8"
            )
            self.add_verdict(
                verdict_id="harness-output-contract",
                claim="Every verdict has required fields and existing artifacts.",
                arch="gfx950+gfx942",
                evidence_kind="runtime",
                status="fail",
                artifact_paths=[self.relative_output_path(error_path)],
                method="Harness self-check of verdict schema and artifact paths.",
                observed={"errors": validation_errors},
            )

        overall = (
            "fail"
            if any(verdict["status"] == "fail" for verdict in self.verdicts)
            else "pass"
        )
        counts = {
            status: sum(verdict["status"] == status for verdict in self.verdicts)
            for status in ("pass", "fail", "recorded")
        }
        verdict_document = {
            "schema_version": 1,
            "overall_status": overall,
            "generated_at": self.finished_at,
            "counts": counts,
            "verdicts": self.verdicts,
        }
        (self.output_dir / "verdicts.json").write_text(
            json.dumps(verdict_document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary_lines = [
            f"overall_status={overall}",
            f"pass={counts['pass']}",
            f"fail={counts['fail']}",
            f"recorded={counts['recorded']}",
            "runtime_arch=gfx950",
            "runtime_device=0",
            "gfx942_mode=compile-only",
        ]
        failed_ids = [
            verdict["id"]
            for verdict in self.verdicts
            if verdict["status"] == "fail"
        ]
        if failed_ids:
            summary_lines.append("failed_ids=" + ",".join(failed_ids))
        (self.output_dir / "summary.txt").write_text(
            "\n".join(summary_lines) + "\n", encoding="utf-8"
        )
        self.write_run_manifest(overall)
        return overall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile and run the compact gfx950 validation manifest; gfx942 is "
            "always compile-only."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="run the full harness in a temporary directory and keep no artifacts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="artifact directory (default: validation/results/<UTC timestamp>)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="per-command timeout in seconds (default: 120)",
    )
    args = parser.parse_args()
    if args.check and args.output is not None:
        parser.error("--check and --output are mutually exclusive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        specification = load_and_validate_manifest()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"manifest error: {error}", file=sys.stderr)
        return 2

    hipcc_value = os.environ.get("HIPCC") or shutil.which("hipcc")
    if not hipcc_value:
        print("hipcc not found; set HIPCC or add it to PATH", file=sys.stderr)
        return 2
    hipcc = Path(hipcc_value).resolve()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_context: tempfile.TemporaryDirectory[str] | None = None
    if args.check:
        output_context = tempfile.TemporaryDirectory(
            prefix="gfx950-validation-check-"
        )
        output_dir = Path(output_context.name) / "artifacts"
    elif args.output is not None:
        output_dir = args.output.resolve()
    else:
        output_dir = VALIDATION_DIR / "results" / timestamp

    try:
        with tempfile.TemporaryDirectory(prefix="gfx950-validation-work-") as work:
            harness = Harness(
                specification,
                output_dir,
                Path(work),
                hipcc,
                args.timeout,
                args.check,
            )
            overall = harness.execute()
            print((output_dir / "summary.txt").read_text(encoding="utf-8").strip())
            if args.check:
                print("artifacts=temporary (--check)")
            else:
                print(f"artifacts={output_dir}")
            return 0 if overall == "pass" else 1
    except FileExistsError:
        print(f"output directory already exists: {output_dir}", file=sys.stderr)
        return 2
    finally:
        if output_context is not None:
            output_context.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
