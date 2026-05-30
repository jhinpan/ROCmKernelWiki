---
id: hw-s-waitcnt
title: "s_waitcnt — Asynchronous Memory Counters (CDNA)"
type: hardware
architectures:
- gfx942
- gfx950
tags:
- s-waitcnt
- async-copy
- async-pipeline
- lds
- global-instructions
- buffer-instructions
confidence: source-reported
related:
- hw-async-copy-lds
- hw-memory-instructions
- hw-lds
- technique-mfma-pipelining
- technique-lds-double-buffering
- migration-cuda-to-hip
sources:
- doc-cdna3-isa
- doc-cdna4-isa
- doc-llvm-amdgpu
- blog-matrix-cores-cdna
- blog-gemm-optimization
aliases:
- s_waitcnt
- waitcnt
- vmcnt
- lgkmcnt
- expcnt
---

# s_waitcnt — Asynchronous Memory Counters (CDNA)

## Overview

On CDNA there is **no scoreboard hardware that automatically stalls a wave on a
data hazard for memory**. Memory instructions (`buffer_*`, `global_*`, `flat_*`,
`ds_*`, scalar `s_load_*`) are *asynchronous*: the issuing wave continues
executing while the access is in flight. Ordering and consumption of results is
the **compiler's responsibility**, expressed with the scalar instruction
`s_waitcnt` (and on newer ISAs the helper `s_waitcnt_*` forms).

Each wave maintains a small set of saturating **event counters**. Issuing an
async op *increments* the relevant counter; the op *decrements* it when it
retires (data has landed in VGPRs/LDS, or a write has reached its coherence
point). `s_waitcnt` blocks the wave until a named counter has dropped to a
target value. This is the AMD analog of CUDA's `cp.async` commit/`mbarrier`
machinery — except the counters are **global per-wave registers managed by the
compiler**, not user-visible barrier objects in memory.

```asm
; Classic software-pipelined load -> use pattern (conceptual gfx942 asm)
    global_load_dwordx4 v[8:11],  v[0:1], off    ; VMCNT: 0 -> 1
    global_load_dwordx4 v[12:15], v[2:3], off    ; VMCNT: 1 -> 2
    ; ... independent ALU work overlaps the two outstanding loads ...
    v_add_f32 v20, v21, v22
    s_waitcnt vmcnt(1)        ; wait until <= 1 VMEM op outstanding
                              ; => v[8:11] is now valid, v[12:15] may still be in flight
    v_mfma_f32_16x16x16_f16 a[0:3], v[8:9], v[10:11], a[0:3]
    s_waitcnt vmcnt(0)        ; drain the rest before reusing v[12:15]
```

## The three counters

| Counter | Width (CDNA3) | Tracks | Decrements when |
|---|---|---|---|
| `vmcnt`   | 6-bit | Vector memory: `buffer_*`, `global_*`, `flat_*` **loads** (and store completion) | load data is written to VGPRs / store reaches L2 |
| `lgkmcnt` | 4-bit | **L**DS, **G**DS/GWS, scalar-**K** const-memory (`s_load_*`), and **M**essage ops | the LDS/scalar/message op completes |
| `expcnt`  | 3-bit | Export & GDS issue (graphics/legacy) | the export buffer is freed |

A few consequences fall directly out of the encoding:

- **`vmcnt` is split on gfx942/gfx950.** The 6-bit field is encoded across two
  sub-fields in the `s_waitcnt` immediate, so it can express up to 63 outstanding
  VMEM ops — enough for deep load pipelines.
- **`lgkmcnt` saturates at a small value** (4-bit). Because *scalar loads*, *LDS
  ops*, and *messages* all share `lgkmcnt`, a wave that interleaves `s_load_*`
  with `ds_read_*` must reason about both when picking a wait target.
- **`expcnt` is essentially unused for compute.** On CDNA3 it covers export/GDS;
  the CDNA4 ISA documents it as *Unused*. Compute kernels almost never emit an
  `expcnt` wait.

`s_waitcnt 0` (all fields zero, the assembler default when you write
`s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)`) is a **full memory drain** for the
wave.

## Ordering rules

The key semantic, stated in the ISA manuals, is about *completion order*:

1. **Same-type ops complete in issue order.** Two `global_load`s tracked by
   `vmcnt` retire in the order they were issued, so `s_waitcnt vmcnt(N)`
   deterministically means "all but the last `N` issued VMEM ops have landed."
   This is what makes the `vmcnt(1)` / `vmcnt(0)` staircase above correct.
2. **Different-type ops may complete out of order.** A `ds_read` (`lgkmcnt`) and
   a `global_load` (`vmcnt`) have *no* ordering guarantee relative to each other;
   you must wait on each counter you depend on.
3. **`flat_*` touches two counters.** Because a `flat` instruction resolves to
   global *or* LDS at runtime, it increments **both `vmcnt` and `lgkmcnt`**. In
   practice the only safe wait after a dependent `flat` op is `s_waitcnt 0` (or
   waiting both counters to zero) — a reason performance kernels prefer explicit
   `global_*`/`buffer_*` (VMCNT-only) and `ds_*` (LGKMCNT-only) over `flat`.
4. **Counters are per-wave**, not per-CU or per-workgroup. Cross-wave ordering
   within a workgroup still needs `s_barrier` (`__syncthreads`); `s_waitcnt`
   only sequences *this* wave's own outstanding accesses.

## Interaction with direct-to-LDS (async copy)

The [direct-to-LDS load](async-copy-lds.md) (`buffer_load_dword ... lds`,
`global_load_lds_*`) streams HBM→LDS while bypassing VGPRs — AMD's closest analog
to NVIDIA `cp.async`. Crucially, **its completion is still tracked by `vmcnt`**
(it is a VMEM op), *not* `lgkmcnt`, even though the destination is LDS. The
canonical double-buffered GEMM prologue therefore waits on `vmcnt` before the
consuming `ds_read`, then on a separate `s_barrier`:

```asm
    buffer_load_dwordx4 v[off], s[desc:desc+3], 0 offen lds   ; VMCNT++ ; HBM -> LDS
    ; ... issue more tile loads, overlap with MFMA on the previous tile ...
    s_waitcnt vmcnt(0)        ; all direct-to-LDS copies have written LDS
    s_barrier                 ; make this tile visible to the whole workgroup
    ds_read_b128 v[0:3], v[lds_addr]                          ; LGKMCNT++ ; LDS -> VGPR
    s_waitcnt lgkmcnt(0)      ; operands ready for the matrix core
```

Picking the *largest safe* non-zero wait targets (`vmcnt(N)` instead of
`vmcnt(0)`) is exactly how [software pipelining](../technique/mfma-pipelining.md)
keeps many loads in flight to hide HBM latency. Over-waiting (`s_waitcnt 0`
everywhere) serializes the pipeline and is one of the most common causes of a
memory-bound stall on CDNA.

## Compiler-managed, not user-managed

You rarely hand-write `s_waitcnt`. The **LLVM AMDGPU backend inserts waits
automatically** via the `SIInsertWaitcnts` pass, using its model of which
instruction produces each value. Implications for kernel authors:

- In HIP/C++ the dependency is implicit: reading a value returned by a
  `__builtin_amdgcn_global_load_*` (or a plain pointer dereference) forces the
  compiler to emit the matching `s_waitcnt` before the use.
- `__builtin_amdgcn_s_waitcnt(imm)` and `__builtin_amdgcn_sched_barrier()` let
  you nudge or pin scheduling when the heuristic over- or under-waits in a tight
  inner loop.
- Inspect the generated waits with `--save-temps` / `llvm-objdump -d` — a wall of
  `s_waitcnt vmcnt(0)` in the hot loop is a red flag that loads are not being
  hoisted far enough ahead of their uses.

## vs NVIDIA mbarrier / cp.async

| Aspect | AMD CDNA (`s_waitcnt`) | NVIDIA (Hopper/Ampere) |
|---|---|---|
| Async tracking | Global per-wave counters (`vmcnt`/`lgkmcnt`/`expcnt`) | `mbarrier` objects in shared memory; `cp.async` commit groups |
| Who manages it | Compiler (`SIInsertWaitcnts`) | Programmer (PTX `mbarrier.arrive/try_wait`, `cp.async.wait_group`) |
| Granularity | Counter value (wait "all but N") | Named barrier / phase per pipeline stage |
| HBM→shared engine | Direct-to-LDS load (tracked by `vmcnt`) | `cp.async` / TMA (`cp.async.bulk`) |
| Bulk DMA descriptor | none (no TMA equivalent on CDNA3/4) | TMA tensor descriptors |
| Cross-thread ordering | separate `s_barrier` | `mbarrier` can fuse arrival + completion |

The practical porting rule (see
[CUDA→HIP migration](../migration/cuda-to-hip.md)): a `cp.async` +
`cp.async.wait_group(N)` sequence maps to *direct-to-LDS loads* +
`s_waitcnt vmcnt(N)`, and an `mbarrier` used purely for async-copy completion has
**no object equivalent** — it collapses into a counter wait plus, where
cross-wave visibility is needed, an `s_barrier`.

## See also

- [Direct-to-LDS async copy](async-copy-lds.md)
- [Memory instructions: buffer / global / flat](memory-instructions.md)
- [MFMA software pipelining](../technique/mfma-pipelining.md)
- [LDS double buffering](../technique/lds-double-buffering.md)

## Sources

- [AMD Instinct MI300 / CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) — `s_waitcnt`, counter widths, completion ordering.
- [AMD Instinct CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) — `expcnt` deprecation, `lgkmcnt` field changes.
- [LLVM AMDGPU Backend — Memory Model & SIInsertWaitcnts](https://llvm.org/docs/AMDGPUUsage.html) — automatic waitcnt insertion.
- [Matrix Core Programming on CDNA3 and CDNA4](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html) — load/MFMA pipelining context.
- [Optimizing GEMM on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores/README.html) — pipelining loads with relaxed `vmcnt` targets.
