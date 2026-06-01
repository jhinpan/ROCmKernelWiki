---
id: hw-atomics
title: Atomics & Cross-Workgroup Sync (CDNA/RDNA)
type: hardware
architectures:
- gfx942
- gfx950
- gfx1201
tags:
- global-instructions
- buffer-instructions
- ds-instructions
- lds
- s-waitcnt
- cdna
confidence: source-reported
evidence_basis:
- source_id: doc-cdna3-isa
  evidence_type: official-doc
- source_id: doc-cdna4-isa
  evidence_type: official-doc
- source_id: ref-gcnasm
  evidence_type: upstream-code
related:
- hw-lds
- hw-memory-instructions
- technique-wave-reduce
- kernel-rmsnorm
sources:
- doc-cdna3-isa
- doc-cdna4-isa
- ref-gcnasm
aliases:
- atomics
- atomic
- cmpswap
- atomicAdd
- atomicCAS
- compare and swap
implemented_by:
- pr-composable_kernel-3098
- pr-composable_kernel-2715
- pr-aiter-3072
- pr-aiter-2394
- pr-Tensile-293
- pr-Tensile-291
- pr-Tensile-1521
- pr-Tensile-1383
---
# Atomics & Cross-Workgroup Sync (CDNA/RDNA)

## Overview

Atomic instructions read-modify-write a memory location indivisibly with respect
to other lanes, waves, and workgroups. On CDNA/RDNA they come in two distinct
hardware paths:

- **LDS atomics** — `ds_add_*`, `ds_min_*`, `ds_cmpst_*`, executed by the LDS
  atomic ALUs inside each CU. Scoped to a single workgroup (one CU).
- **Global / buffer atomics** — `global_atomic_*` (flat addressing) and
  `buffer_atomic_*` (MUBUF, with out-of-bounds semantics), executed at the
  **L2 cache** and visible device-wide.

Both forms can optionally **return** the pre-op value (the `_rtn` variant); the
non-returning form is cheaper because the wave does not have to wait for the
result to come back. HIP's `atomicAdd`, `atomicCAS`, `atomicMax`, etc. lower to
these instructions depending on the address space of the pointer.

## Where atomics execute

| Atomic family | Instruction | Executes at | Visibility scope |
|---|---|---|---|
| LDS | `ds_add_u32`, `ds_cmpst_b32` | CU LDS ALU | workgroup |
| Global (flat) | `global_atomic_add` | L2 | device (agent) |
| Buffer (MUBUF) | `buffer_atomic_add` | L2 | device (agent) |

Because global atomics resolve at L2, throughput is bounded by how many distinct
cache lines are in flight — heavy contention on a single address serializes at
the owning L2 slice. Spread counters across cache lines (padding/striping) when a
hotspot appears.

## LDS atomic ALUs

CDNA CUs contain dedicated integer atomic ALUs on the LDS path so that
`ds_add_u32`-style reductions do not consume the main VALU. CDNA4 (gfx950) widens
this path: the LDS atomic throughput was increased relative to CDNA3, with on the
order of **32 atomic ALU lanes** servicing LDS atomics per CU (vs. the narrower
CDNA3 path). This makes LDS-local histogram / reduction kernels notably faster on
gfx950. *(Exact lane counts are architecture-version sensitive; treat the figure
as source-reported and confirm against your ISA build.)*

A wave-local reduction usually wants **cross-lane** ops (DPP / `ds_swizzle`)
first, then a single LDS or global atomic per wave — not one atomic per lane.
See [wave reduce](../techniques/wave-reduce.md) and
[RMSNorm](../kernels/rmsnorm.md), which reduce within the wave before touching a
shared accumulator.

## Floating-point atomics and their caveats

`global_atomic_add_f32` and packed `global_atomic_pk_add_f16` /
`..._pk_add_bf16` (two bf16/fp16 lanes per op) exist on CDNA3/CDNA4, and the
packed forms are valuable for split-K GEMM epilogues and gradient accumulation.
Important caveats:

- **Non-determinism / non-associativity.** FP atomic add is order-dependent;
  repeated runs may differ in the low bits. This is expected, not a bug.
- **Compiler gating.** Clang only emits hardware FP atomics when it can prove the
  target is device (coarse-grained) memory and unsafe-FP atomics are allowed.
  Use `-munsafe-fp-atomics` (or `#pragma clang fp ...`) — otherwise `atomicAdd`
  on a `float*` may be lowered to a **CAS loop** (see below) or bounced to a
  slower path.
- **Fine-grained host memory.** On fine-grained (coherent) allocations the
  hardware FP atomic may not be used at all; the op falls back to a CAS loop over
  the PCIe/Infinity Fabric link.

```cpp
// Packed bf16 atomic add — two bf16 accumulated in one global atomic.
// Useful for split-K partial-sum reduction into a bf16 output tile.
__device__ void accumulate_pk_bf16(__hip_bfloat162* dst, __hip_bfloat162 v) {
    // lowers to global_atomic_pk_add_bf16 when targeting device memory
    atomicAdd(dst, v);
}
```

## atomicCAS — building your own atomics

Compare-and-swap is the universal primitive: every other atomic can be emulated
with a CAS retry loop. CDNA exposes it as `ds_cmpst_b32`/`ds_cmpst_b64` (LDS) and
`global_atomic_cmpswap`/`buffer_atomic_cmpswap` (global). The `gcnasm`
`cmpswap_atomic` example demonstrates the canonical loop; the HIP equivalent:

```cpp
// Emulate a float atomic-max with compare-and-swap (works in any address space).
__device__ float atomicMaxFloat(float* addr, float val) {
    unsigned int* uaddr = reinterpret_cast<unsigned int*>(addr);
    unsigned int  old   = *uaddr, assumed;
    do {
        assumed = old;
        float cur = __uint_as_float(assumed);
        if (cur >= val) break;                 // already larger; done
        old = atomicCAS(uaddr, assumed, __float_as_uint(val));
    } while (assumed != old);                  // retry if someone raced us
    return __uint_as_float(old);
}
```

The loop must re-read the *returned* old value each iteration; only the thread
whose `assumed` matched the memory wins, and losers retry. Keep CAS loop bodies
short — long bodies widen the race window and increase retries under contention.

## Cross-workgroup synchronization

There is **no hardware barrier across workgroups** mid-kernel — `s_barrier`
(`__syncthreads`) only synchronizes one workgroup. Cross-workgroup coordination
is built on global atomics:

**1. Spin on a global flag.** A producer writes a value and consumers poll it.
The poll must use a `volatile`/atomic load and a memory fence so the compiler
does not hoist the read out of the loop, and so writes are visible at device
scope. This is what the `gcnasm` `cross-wg-sync` example illustrates.

```cpp
// Device-scope flag spin. flag must be device (coarse-grained) memory.
__device__ void wait_for(volatile int* flag, int target) {
    while (atomicAdd((int*)flag, 0) < target) { /* spin */ }
    __threadfence();   // make subsequent reads observe producer's writes
}
__device__ void signal(int* flag) {
    __threadfence();   // publish our writes before bumping the counter
    atomicAdd(flag, 1);
}
```

**2. Grid sync / cooperative groups.** HIP cooperative groups
(`cg::this_grid().sync()` launched via `hipLaunchCooperativeKernel`) provide a
device-wide barrier, but require that **all workgroups are co-resident** — the
launch must fit in the occupancy budget of the device, so grid-sync kernels are
typically [persistent kernels](../techniques/persistent-kernel.md) sized to one
wave-group per CU. If the grid is over-subscribed, the sync deadlocks.

**3. Atomic work counters.** Stream-K and persistent GEMM schedulers use a single
`global_atomic_add` on a work-tile counter to hand out tiles dynamically,
avoiding any global barrier. This is the most scalable pattern when the work is
independent.

> **Deadlock warning.** A spin-wait across workgroups is only safe when the
> waiting and signaling workgroups are guaranteed to run concurrently. If more
> workgroups are launched than the device can hold, a waiter may occupy a CU that
> a not-yet-scheduled signaler needs → hang. Prefer cooperative launch or a
> dynamic atomic counter over hand-rolled grid barriers.

## Ordering and fences

CDNA atomics are not implicitly ordered with surrounding loads/stores. Use
`__threadfence()` (device scope, lowers to the appropriate `s_waitcnt` +
cache-flush/invalidate sequence), `__threadfence_block()` (workgroup), or
`__threadfence_system()` (system) to establish ordering. A returning atomic also
acts as a synchronization point because the wave must `s_waitcnt vmcnt(0)` for
the result — see [s_waitcnt](s-waitcnt.md).

## See also

- [LDS — Local Data Share](lds.md)
- [Memory instructions — buffer vs global vs flat](memory-instructions.md)
- [Wave reduce technique](../techniques/wave-reduce.md)
- [RMSNorm kernel](../kernels/rmsnorm.md)

## Sources

- [CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [gcnasm — GCN assembly & HIP examples (`cmpswap_atomic`, `cross-wg-sync`)](https://github.com/carlushuang/gcnasm)
- [HIP atomic functions — ROCm documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html#atomic-functions)
