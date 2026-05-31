---
id: pattern-memory-bound
title: Memory-Bound Kernel (HBM-limited, idle matrix cores)
type: pattern
architectures:
- gfx942
- gfx950
tags:
- memory-bound
- low-compute-utilization
- high-hbm-bw
- hbm3
- vectorized-loads
- kernel-fusion
- data-reuse
symptoms:
- memory-bound
- low-compute-utilization
- high-hbm-bw
candidate_techniques:
- technique-vectorized-loads
- technique-lds-double-buffering
- technique-kernel-fusion
related:
- hw-memory-instructions
- hw-async-copy-lds
- hw-chiplet-xcd
- kernel-bandwidth-microbench
- pattern-mfma-underutilized
sources:
- hw-memory-instructions
- kernel-bandwidth-microbench
- blog-gemm-optimization
- doc-mi300x-datasheet
- doc-cdna3-isa
---
# Memory-Bound Kernel (HBM-limited, idle matrix cores)

## What this pattern looks like

The kernel spends most of its time waiting on global memory. The matrix cores
and VALU sit idle behind `s_waitcnt vmcnt(...)` while data trickles in from HBM.
Telltale signs:

- **High `MemUnitBusy` / HBM read+write bandwidth** approaching the device peak
  (MI300X ≈ 5.3 TB/s, MI355X up to 8 TB/s — see
  [MI300X datasheet](../../sources/docs/doc-mi300x-datasheet.md)).
- **Low `VALUBusy` and near-zero `MFMABusy`** — the compute units are starved.
- A large fraction of wave cycles attributed to **VMEM stalls** (waiting on
  VMCNT) in the profiler (`rocprof` / `rocprofv3` / Omniperf).
- Measured **arithmetic intensity** (FLOP per byte of HBM traffic) sits to the
  *left* of the roofline ridge point.

If instead the matrix cores are busy but underfed by LDS, you are looking at the
[MFMA-underutilized](mfma-underutilized.md) or [bank-conflict](bank-conflicts.md)
patterns, not this one.

## The roofline test

A kernel is memory-bound when its arithmetic intensity is below the device's
ridge point `AI_ridge = peak_FLOPs / peak_HBM_BW`. For MI300X FP16
(1307 TFLOPS dense / 5.3 TB/s ≈ **247 FLOP/byte**), anything that touches each
loaded byte only a handful of times is firmly HBM-limited.

```python
# Back-of-envelope roofline for a single kernel launch.
# peak_flops, peak_bw from the datasheet; flops/bytes from your kernel.
def bound(flops, bytes_hbm, peak_flops=1307e12, peak_bw=5.3e12):
    ai        = flops / bytes_hbm            # arithmetic intensity (FLOP/byte)
    ai_ridge  = peak_flops / peak_bw         # ~247 FLOP/byte on MI300X FP16
    t_compute = flops / peak_flops
    t_memory  = bytes_hbm / peak_bw
    return ("memory-bound" if ai < ai_ridge else "compute-bound",
            f"AI={ai:.1f}  ridge={ai_ridge:.0f}  "
            f"achievable={min(peak_flops, ai*peak_bw)/1e12:.0f} TFLOPS")

# Example: fp16 elementwise add over N elements -> 1 FLOP per 6 bytes moved
print(bound(flops=1, bytes_hbm=6))   # ('memory-bound', 'AI=0.2 ...')
```

Elementwise ops (add, scale, activations, norms), GEMV / thin GEMMs, attention
**decode** (batch-1, KV-cache streaming), and dequant kernels are intrinsically
memory-bound: there is simply not enough reuse to hide HBM latency. The goal for
these is to **hit peak HBM bandwidth**; for borderline cases (skinny GEMM) the
goal is to **raise arithmetic intensity** until the kernel crosses the ridge.

## Confirm with a bandwidth ceiling

Before optimizing, know the real achievable ceiling — not the spec number. Run a
streaming `float4` copy/triad microbenchmark and treat its sustained GB/s as
"100%". On CDNA a well-tuned copy reaches roughly 80–90% of spec HBM bandwidth;
your kernel's job is to approach *that* number, not the datasheet peak. See
[bandwidth microbench](../kernels/bandwidth-microbench.md).

## Fixes (in priority order)

### 1. Stop re-reading HBM — fuse (`technique-kernel-fusion`)

The single biggest win for memory-bound chains is to **not go back to HBM**.
Producer→consumer elementwise/normalization/activation stages each re-stream the
whole tensor; fusing them into one kernel cuts HBM traffic by the number of
stages, multiplying effective throughput. A `bias → GELU → scale` chain fused
into a GEMM epilogue moves the activation once instead of three times. See
[kernel fusion](../techniques/kernel-fusion.md) and the epilogue discussion in
the [GEMM optimization blog](../../sources/blogs/blog-gemm-optimization.md).

### 2. Vectorize and coalesce loads (`technique-vectorized-loads`)

Per-lane scalar `global_load_dword` leaves the memory system idle. Issue the
widest aligned transaction the data allows — `global_load_dwordx4` /
`buffer_load_dwordx4` (128 bits/lane) — so each wave moves 64×16 = 1024 bytes
per instruction and the loads coalesce into full cache lines. This both reduces
instruction count and keeps more requests in flight to saturate HBM.

```cpp
// Memory-bound elementwise: y = alpha*x + y, vectorized 128-bit/lane.
// One global_load_dwordx4 per lane instead of four dword loads.
using float4 = __attribute__((__vector_size__(16))) float;

__global__ void saxpy_vec4(const float4* __restrict__ x,
                           float4* __restrict__ y,
                           float alpha, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;                 // x4-packed length
    float4 xv = x[i];                    // -> global_load_dwordx4
    float4 yv = y[i];
    yv.x += alpha * xv.x;  yv.y += alpha * xv.y;
    yv.z += alpha * xv.z;  yv.w += alpha * xv.w;
    y[i] = yv;                           // -> global_store_dwordx4
}
```

Ensure base pointers are 16-byte aligned and `n` is padded/guarded so the
compiler keeps the wide opcode. For boundary tiles, prefer
**buffer** addressing: out-of-bounds reads return 0 and OOB writes are dropped,
giving a branchless guard without splitting the wide load — see
[memory instructions](../hardware/memory-instructions.md) and
[buffer OOB guards](../techniques/buffer-oob-guard.md).

### 3. Overlap streaming with compute (`technique-lds-double-buffering`)

When there *is* reuse (tiled GEMM, attention), hide the load latency entirely by
prefetching the next tile while the matrix cores consume the current one. Double
(ping-pong) buffering through LDS, ideally fed by **direct-to-LDS async copy**
(`buffer_load ... lds` / `global_load_lds_*`) so the HBM→LDS stream bypasses
VGPRs, keeps VMEM in flight across the whole K-loop. The kernel becomes limited
by `max(t_compute, t_memory)` instead of their sum. See
[LDS double buffering](../techniques/lds-double-buffering.md) and
[async copy to LDS](../hardware/async-copy-lds.md).

```cpp
// Software-pipelined tile loop: issue next load, wait on all-but-last, compute.
load_tile(buf[0], gmem + 0);                 // prime
for (int k = 0; k < K_TILES; ++k) {
    if (k + 1 < K_TILES)
        load_tile(buf[(k+1)&1], gmem + (k+1)*KTILE);  // prefetch next
    __builtin_amdgcn_s_waitcnt(/* vmcnt */ 0x0f70 /* keep 1 outstanding */);
    compute_tile(acc, buf[k & 1]);            // MFMA consumes current tile
}
```

### 4. Maximize in-flight requests / occupancy

HBM bandwidth is reached only with enough outstanding misses. If occupancy is
throttled by VGPR/LDS usage, too few waves are resident to keep the memory
pipeline full — coordinate with [occupancy tuning](../techniques/occupancy-tuning.md)
and watch [VGPR pressure](vgpr-pressure.md). More resident waves = more
concurrent VMEM transactions = higher achieved bandwidth.

### 5. Exploit the cache hierarchy / XCD locality

Memory-side **Infinity Cache** (256 MB LLC) and the per-XCD 4 MB L2 can turn HBM
traffic into cache hits if the access pattern has locality. Because L2 coherence
is **per-XCD**, cross-XCD reuse still costs LLC/HBM bandwidth; scheduling tiles
so a working set stays within one XCD reduces real HBM pressure. See
[XCD locality](xcd-locality.md).

## Anti-patterns

- **Strided / gather access** that defeats coalescing — each lane touches a
  separate cache line, so effective bandwidth collapses. Restructure layout
  (transpose via [LDS](../kernels/transpose-lds.md)) before streaming.
- **Reading then immediately re-reading** the same tensor in separate kernels
  (un-fused pipelines) — see fix #1.
- **Adding ILP/unrolling to a memory-bound loop** expecting compute speedup: it
  will not help once you are bandwidth-saturated; only traffic reduction or
  higher achieved bandwidth moves the needle.

## Verifying the fix

Re-measure achieved HBM bandwidth (`rocprof` derived `FETCH_SIZE`+`WRITE_SIZE`
over kernel time, or Omniperf's memory chart) against the
[microbenchmark ceiling](../kernels/bandwidth-microbench.md). A memory-bound
kernel is "done" when it sustains ~80–90% of that ceiling; a borderline kernel is
done when fusion/blocking has pushed its arithmetic intensity past the roofline
ridge and `MFMABusy` rises.

## Sources

- [CDNA3 ISA Reference Guide — memory instructions](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [AMD Instinct MI300X datasheet (HBM3 bandwidth, peak FLOPS)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)
- [Optimizing GEMM on AMD GPUs (ROCm blog) — fusion & tiling](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores/README.html)
- [Bandwidth microbenchmark kernel](../kernels/bandwidth-microbench.md)
- [Memory instructions: buffer vs global vs flat](../hardware/memory-instructions.md)
