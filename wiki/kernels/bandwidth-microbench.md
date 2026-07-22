---
id: kernel-bandwidth-microbench
title: HBM Bandwidth Microbenchmark (float4 non-temporal persistent read)
type: kernel
architectures:
- gfx942
- gfx950
tags:
- bandwidth-bench
- memcpy
- vectorized-loads
- nontemporal-loads
- persistent-kernel
- memory-bound
- hbm3
- buffer-instructions
confidence: source-reported
reproducibility: runnable
version_sensitive:
- vs-amdgpu-nontemporal-lowering
artifact_dir: examples/bandwidth-microbench
kernel_types:
- bandwidth-bench
- memcpy
languages:
- hip
hardware_features:
- global-instructions
- buffer-instructions
- vgpr
- wave64
techniques:
- vectorized-loads
- nontemporal-loads
- persistent-kernel
- loop-unrolling
related:
- technique-vectorized-loads
- technique-persistent-kernel
- hw-memory-instructions
- pattern-memory-bound
- kernel-vector-add-asm
sources:
- ref-gcnasm
- technique-vectorized-loads
- technique-persistent-kernel
- hw-memory-instructions
- doc-mi300x-datasheet
performance_claims:
- gpu: MI308X
  dtype: fp32
  metric: hbm-read-bandwidth
  value: 4560 GB/s
  source_id: ref-gcnasm
  utilization: ~86% of MI300-class 5.3 TB/s HBM3 peak
  confidence: source-reported
implemented_by:
- pr-Tensile-1185
- pr-aiter-3072
- pr-Tensile-311
- pr-Tensile-1406
- pr-Tensile-1288
- pr-Tensile-1184
- pr-Tensile-1179
- pr-FlyDSL-60
---
# HBM Bandwidth Microbenchmark (float4 non-temporal persistent read)

## Overview

A bandwidth microbenchmark exists to answer one question: **how close to the
HBM peak can a real kernel actually get?** It is the empirical ceiling against
which every memory-bound kernel (RMSNorm, paged-attention KV streaming,
elementwise epilogues) should be measured. If your production kernel reaches
70% of HBM peak but this microbench reaches 86%, you have ~16 points of
headroom to chase; if the microbench *also* tops out at 70%, the bottleneck is
elsewhere (occupancy, launch overhead, partition mode).

This page documents the read-only variant from
[`gcnasm`'s `bandwidth_memread`](../../sources/refs/ref-gcnasm.md): a
**persistent** kernel where each wave streams a large array with **128-bit
(`float4`) vectorized, non-temporal `global_load`s** and accumulates into a
register sink so the compiler cannot dead-code the loads away. On an **MI308X
(gfx942, CDNA3)** it sustains **~4.56 TB/s** of HBM3 read bandwidth.

## Why these three design choices

The kernel combines three orthogonal techniques, each of which independently
raises sustained bandwidth:

1. **Vectorized 128-bit loads** ([vectorized-loads](../techniques/vectorized-loads.md)).
   One `global_load_dwordx4` moves 16 bytes per lane → 1 KiB per wave64 issue.
   This minimizes the number of in-flight memory instructions needed to
   saturate the VMEM pipe and keeps the `s_waitcnt vmcnt` accounting cheap.
2. **Non-temporal hint** ([nontemporal-loads]). A streaming read touched once
   can use a target-specific cache-policy hint. On the pinned clang 20 path,
   `__builtin_nontemporal_load` lowers to a `global_load ... nt` modifier for
   gfx942/gfx950. Treat `nt` as a policy hint: it does not promise to bypass L2,
   MALL, or every cache level, and does not change coherency.
3. **Persistent grid** ([persistent-kernel](../techniques/persistent-kernel.md)).
   Launch exactly one block per CU (304 on MI300X-class, fewer active CUs on
   MI308X), then grid-stride over the whole buffer. This removes block-launch
   tail effects and keeps every CU's memory pipe continuously fed.

See [memory instructions](../hardware/memory-instructions.md) for the
`global` vs `buffer` vs `flat` distinction. This kernel uses `global_load`
(SADDR + VGPR offset, gated by **VMCNT only**), which has the lowest addressing
overhead for a flat, in-bounds streaming pattern.

## The kernel

```cpp
#include <hip/hip_runtime.h>

// Use a compiler-native vector: this clang accepts it for the nontemporal
// builtin, while HIP's float4 wrapper is rejected by that builtin.
using f32x4 = float __attribute__((ext_vector_type(4)));

// One persistent block per CU; grid-stride over the whole array.
// Each thread issues UNROLL non-temporal float4 loads per iteration.
template <int UNROLL = 8>
__global__ void bandwidth_memread(const f32x4* __restrict__ in,
                                  float* __restrict__ sink,
                                  size_t n_vec4 /* number of float4 elements */)
{
    const size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;

    // Register accumulator: forces the loads to be live (no DCE).
    f32x4 acc = {0.f, 0.f, 0.f, 0.f};

    // Grid-stride loop, manually unrolled so multiple loads are in flight
    // before the first s_waitcnt vmcnt() stalls the wave.
    size_t i = tid;
    for (; i + UNROLL * stride < n_vec4; i += UNROLL * stride) {
        f32x4 v[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u)
            // Target-specific non-temporal cache-policy hint.
            v[u] = __builtin_nontemporal_load(in + i + (size_t)u * stride);
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            acc[0] += v[u][0]; acc[1] += v[u][1];
            acc[2] += v[u][2]; acc[3] += v[u][3];
        }
    }
    // Tail.
    for (; i < n_vec4; i += stride) {
        f32x4 v = __builtin_nontemporal_load(in + i);
        acc[0] += v[0]; acc[1] += v[1]; acc[2] += v[2]; acc[3] += v[3];
    }

    // Only thread 0 of the last block writes — keeps the sink alive without
    // adding measurable store traffic to the read benchmark.
    if (acc[0] == -1.0f)
        sink[tid] = acc[0] + acc[1] + acc[2] + acc[3];
}
```

Host-side launch and timing:

```cpp
int dev = 0; hipDeviceProp_t p; hipGetDeviceProperties(&p, dev);

const size_t bytes   = (size_t)4 << 30;          // 4 GiB working set >> L2+LLC
const size_t n_vec4  = bytes / sizeof(f32x4);
const int    block   = 256;
const int    grid    = p.multiProcessorCount;     // one persistent block / CU

f32x4 *in; float *sink;
hipMalloc(&in, bytes); hipMalloc(&sink, grid * block * sizeof(float));

hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
bandwidth_memread<8><<<grid, block>>>(in, sink, n_vec4);  // warm up
hipEventRecord(a);
for (int it = 0; it < 50; ++it)
    bandwidth_memread<8><<<grid, block>>>(in, sink, n_vec4);
hipEventRecord(b); hipEventSynchronize(b);

float ms = 0.f; hipEventElapsedTime(&ms, a, b);
double gbps = (double)bytes * 50 / (ms * 1e-3) / 1e9;
printf("read bandwidth: %.0f GB/s\n", gbps);
```

Compile and run:

```bash
hipcc -O3 --offload-arch=gfx942 bandwidth_memread.hip -o bwread
./bwread        # MI308X: ~4560 GB/s
```

## What to verify in the ISA

Confirm the compiler actually emitted wide, non-temporal loads — this is the
single most common reason a bandwidth bench underperforms:

```bash
hipcc -O3 --offload-arch=gfx942 -S bandwidth_memread.hip -o - | \
    grep -E 'global_load_dwordx4.*[[:space:]]nt([[:space:]]|$)'
```

You want `global_load_dwordx4 ... nt` (128-bit with the expected policy token)
in the hot loop, *not* four separate `global_load_dword`s. The exact modifier is
compiler/target-sensitive, so inspect rather than assuming `glc`/`slc`. If the
unroll factor is too low the scheduler runs out of independent loads to hide
latency and inserts an early `s_waitcnt vmcnt(0)`; if the accumulator is
eliminated, the loop vanishes entirely. The register sink and impossible guard
exist specifically to keep the loads live without adding store traffic.

## Tuning notes

- **Unroll vs VGPRs.** Each in-flight `float4` costs 4 VGPRs. `UNROLL=8` keeps
  ~32 live load results, enough to hide HBM latency on CDNA3 while staying well
  under the [VGPR budget](../techniques/vgpr-budgeting.md) that would cap
  occupancy. Sweep 4/8/16 — past the latency-hiding point, more unroll only
  costs registers.
- **Working set must exceed the cache hierarchy.** MI300-class parts carry
  4 MB L2 per XCD and a **256 MB Infinity Cache**. A buffer that fits in LLC
  measures cache bandwidth, not HBM. Use ≥1–4 GiB.
- **Partition mode matters.** L2 is coherent **per-XCD** (an effective NUMA
  domain). Under NPS1 a persistent grid spreads naturally across all 8 stacks;
  under NPS4/CPX you measure per-partition bandwidth. Always report the mode.
- **Read vs copy vs write.** `bandwidth_memread` isolates the read path. A
  copy kernel (`memcpy`) issues loads *and* stores and typically reports a
  lower effective number because read and write traffic share the HBM bus.

## Performance

| GPU | Arch | Access | Vector width | Sustained | vs peak |
|---|---|---|---|---|---|
| MI308X | gfx942 | read-only, non-temporal | `float4` (128-bit) | **~4560 GB/s** | ~86% of 5.3 TB/s HBM3 |

The ~4.56 TB/s figure is reported by the
[`gcnasm` microbenchmark suite](../../sources/refs/ref-gcnasm.md). The 5.3 TB/s
peak is the [MI300X datasheet](../../sources/docs/doc-mi300x-datasheet.md) HBM3
figure for the MI300 family; achieving ~86% of theoretical peak is a healthy
result for a real read-streaming kernel, since DRAM refresh, row-activation
overhead, and command-bus turnaround all subtract from the ideal. Treat this as
the empirical roofline for any memory-bound gfx942 kernel — see
[memory-bound pattern](../patterns/memory-bound.md).

## Runnable example

A portable, pure-HIP version of this benchmark that **builds and runs on
gfx950** lives in
[`examples/bandwidth-microbench/`](../../examples/bandwidth-microbench/). It
keeps the same persistent grid-stride, `UNROLL=8`, and non-temporal `float4`
loads, then adds a CPU-checked sum and size sweep. The MI308X CDNA3 figure above
remains source-reported; the output below is from MI355X.

```bash
cd examples/bandwidth-microbench && ./build.sh
```

`build.sh` also greps the emitted ISA to confirm the wide non-temporal load:

```
	global_load_dwordx4 v[10:13], v[4:5], off nt
```

Expected output captured on MI355X / gfx950:

```
Device: AMD Instinct MI355X (gfx950:sramecc+:xnack-), 256 CUs, 2400 MHz

Self-check: sum=16777216 ref=16777216 rel_err=0.000e+00 -> PASS

size(MiB)    iters        GB/s
64           50           6152
256          50           6396
1024         50           6192
2048         50           6192

RESULT: PASS
```

These are values from one run, not a peak-bandwidth claim.

## See also

- [Vectorized loads](../techniques/vectorized-loads.md)
- [Persistent kernels](../techniques/persistent-kernel.md)
- [Memory instructions: buffer vs global vs flat](../hardware/memory-instructions.md)
- [Persistent ASM vector-add with async LDS](vector-add-asm.md)

## Sources

- [gcnasm — AMD GCN/CDNA assembly & microbenchmark collection](https://github.com/AMD-AGI/gcnasm)
- [AMD Instinct MI300X datasheet (HBM3 5.3 TB/s)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)
- [CDNA3 ISA Reference Guide — global_load / s_waitcnt vmcnt](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
