# gfx950 validation harness

This directory contains a compact, reproducible check of the gfx950 claims that
are sensitive to hardware or compiler version. It uses only Python's standard
library and `hipcc`; ISA and HSA metadata are split from text assembly emitted
by `hipcc --genco -S`.

The harness sets `ROCR_VISIBLE_DEVICES=0`, `HIP_VISIBLE_DEVICES=0`, and
`CUDA_VISIBLE_DEVICES=0`, and both runtime probes call `hipSetDevice(0)`.
`gfx942` is compile-only. The harness does not query or change accelerator or
memory partition modes and does not make performance claims.

## Run

From the repository root:

```bash
python3 validation/run.py
```

This writes an immutable run directory under
`validation/results/<UTC timestamp>/`. Use an explicit destination when needed:

```bash
python3 validation/run.py --output /tmp/gfx950-validation
```

For a local pass/fail check that retains no artifacts:

```bash
python3 validation/run.py --check
```

Exit status is zero only when every locally testable expectation passes.
`HIPCC=/path/to/hipcc` selects a non-default compiler.

## What is checked

- HIP device 0 reports MI355X/gfx950, wave64, 256 CUs, 32 waves/CU, and
  163840 bytes (160 KiB) of LDS per CU.
- `__builtin_amdgcn_load_to_lds` acceptance is compiled for widths
  `1,2,4,8,12,16`: gfx950 expects `1,2,4,12,16`; gfx942 expects `1,2,4`.
- A 64-lane gfx950 kernel executes a 16-byte direct-to-LDS copy using a
  lane-varying global source and one wave-uniform LDS base. It checks all 256
  payload dwords and four-dword sentinels immediately below and above the LDS
  destination interval. Its ISA must contain `global_load_lds_dwordx4`.
- Compiler capability probes check gfx950-only `v_permlane16_swap_b32` and
  scaled `f8f6f4` MFMA, plus gfx942-only native XF32 MFMA.
- The decoded `.amdgpu_metadata` block must expose `.vgpr_count` and
  `.agpr_count`. This proves extraction and namespace presence; it deliberately
  does not infer whether one count contains the other.

## Evidence model

`verdicts.json` keeps these categories separate:

- `hardware`: properties reported by HIP/ROCR for the selected device.
- `runtime`: checked kernel output from actual gfx950 execution.
- `compiler`: target acceptance/rejection and emitted ISA/HSA metadata.
- `source-reported`: cited expectations copied from `manifest.json`; these are
  recorded, never promoted to local execution evidence.

Device properties are interface reports, not independent physical
measurements. The direct-to-LDS sentinels check this probe's intended LDS write
interval; they are not a general memory-safety or malformed-pointer test.

## Artifacts

Each retained run contains:

- `manifest.json`: exact input manifest, source SHA-256 values, host, branch,
  commit, toolchain, GPU selectors, and the compile-only/runtime split.
- `commands.json` and `commands.txt`: exact commands, return codes, durations,
  environment overrides, and paths to stdout/stderr.
- `logs/`: command stdout and stderr, including expected compiler rejections.
- `isa/`: emitted AMDGPU assembly without the metadata block.
- `metadata/`: extracted textual `.amdgpu_metadata` blocks.
- `verdicts.json`: machine-readable claims with architecture, toolchain,
  evidence kind, status, expected/observed values, and artifact paths.
- `summary.txt`: overall status and failed verdict IDs, if any.
