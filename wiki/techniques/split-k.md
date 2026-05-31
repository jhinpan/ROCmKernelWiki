---
id: technique-split-k
title: Split-K / GlobalSplitU — Partial-Sum Reduction GEMM for Small M·N, Large K
type: technique
architectures:
- gfx942
- gfx950
- gfx1201
tags:
- split-k
- gemm
- hgemm
- fp8-gemm
- tile-scheduling
- occupancy-tuning
- low-occupancy
- reduction
confidence: source-reported
reproducibility: snippet
hardware_features:
- mfma
- agpr
- l2-cache
kernel_types:
- gemm
- hgemm
- fp8-gemm
languages:
- hip
- composable-kernel
related:
- kernel-ck-hgemm
- technique-stream-k
- technique-persistent-kernel
- technique-occupancy-tuning
- hw-mfma
- hw-chiplet-xcd
sources:
- ref-tensile
- blog-gemm-optimization
- ref-composable-kernel
- ref-hipblaslt
- doc-mi300x-datasheet
implemented_by:
- pr-composable_kernel-933
- pr-composable_kernel-2152
- pr-composable_kernel-2059
- pr-Tensile-1355
- pr-FlyDSL-370
- pr-FlyDSL-346
- pr-composable_kernel-785
- pr-composable_kernel-767
---
# Split-K / GlobalSplitU — Partial-Sum Reduction GEMM

## The problem: not enough tiles to fill the GPU

A GEMM `C[M,N] = A[M,K] · B[K,N]` is tiled along M and N. Each output tile
`(BM × BN)` is owned by one workgroup, so the launch produces
`ceil(M/BM) · ceil(N/BN)` workgroups. When **M and N are small but K is large**
— think a tall-skinny projection, an LLM decode-time GEMM with batch ≈ 1, or a
`[1, K] · [K, N]` GEMV-like shape — that product can be far smaller than the
number of CUs available.

On an MI300X with 304 CUs across 8 XCDs ([chiplet layout](../hardware/chiplet-xcd.md)),
a problem that generates only, say, 40 output tiles leaves the majority of the
machine idle no matter how well each tile is written. The kernel is
**occupancy-starved**, not compute- or memory-bound — see the
[low-occupancy pattern](../patterns/low-occupancy.md). Worse, each surviving
workgroup must stream the *entire* K dimension, so the few active CUs run long
while the rest sit idle.

**Split-K** (called **GlobalSplitU** in Tensile/rocBLAS and `GlobalSplitU`/
`k_batch` in Composable Kernel and hipBLASLt) attacks this by parallelizing the
reduction (contraction) dimension K itself.

## The idea: parallelize the K reduction

Partition K into `SplitK` (a.k.a. `GSU`) slices. Each slice computes a
**partial** product over its K-sub-range for every `(M,N)` output tile, and the
slices are summed afterward:

```
C = Σ_{s=0}^{SplitK-1}  A[:, s·Kp : (s+1)·Kp] · B[s·Kp : (s+1)·Kp, :]
```

This multiplies the grid size by `SplitK`, turning `T` output tiles into
`T · SplitK` workgroups — enough work to cover all CUs. The cost is that the
`SplitK` partial sums must be reduced into the final `C`. Two ways to do that:

- **Atomic Split-K** — every slice atomically accumulates into one `C` buffer
  (`atomicAdd` / `global_atomic_add_f32`). No extra storage, no second launch,
  but requires an atomic-capable accumulator type and `C` pre-zeroed (`beta=0`).
- **Reduction Split-K (workspace)** — each slice writes its tile to a private
  `[SplitK, M, N]` workspace; a small second kernel reduces along the `SplitK`
  axis. Deterministic and dtype-agnostic, at the price of workspace memory
  (`SplitK · M · N · sizeof(acc)`) and an extra kernel launch.

## Minimal HIP sketch (atomic Split-K)

Each workgroup owns one `(M,N)` tile **and** one K-slice `s`. The MFMA inner
loop is unchanged from a normal tiled GEMM — only the K bounds and the epilogue
differ.

```cpp
// grid.z == SplitK ; each z-slice covers K range [k0, k1)
template <int BM, int BN, int BK, int SplitK>
__global__ void splitk_gemm_f16(const half* __restrict__ A,   // [M, K] row-major
                                const half* __restrict__ B,   // [K, N] row-major
                                float* __restrict__ C,        // [M, N], pre-zeroed
                                int M, int N, int K) {
    const int tileM = blockIdx.y * BM;
    const int tileN = blockIdx.x * BN;
    const int s     = blockIdx.z;                 // K-slice id in [0, SplitK)

    // Even K split; the last slice absorbs the remainder so we never read OOB.
    const int Kp = (K + SplitK - 1) / SplitK;
    const int k0 = s * Kp;
    const int k1 = min(k0 + Kp, K);

    // Per-thread accumulator tile lives in AGPRs across the wavefront.
    acc_fragment<BM, BN> acc{};                   // see hw-mfma for layout

    for (int k = k0; k < k1; k += BK) {
        load_tiles_to_lds(A, B, tileM, tileN, k, M, N, K);   // direct-to-LDS
        __syncthreads();
        mfma_accumulate<BM, BN, BK>(acc);          // v_mfma_f32_16x16x16_f16
        __syncthreads();
    }

    // Epilogue: atomically merge this slice's partial sum into global C.
    // Only valid when C is FP32 and beta == 0 (buffer pre-zeroed by the host).
    for (auto [row, col, val] : acc.elements(tileM, tileN)) {
        if (row < M && col < N)
            atomicAdd(&C[row * N + col], val);     // -> global_atomic_add_f32
    }
}
```

Launch with `dim3 grid(ceil(N,BN), ceil(M,BM), SplitK)`. The atomic epilogue
keeps the accumulator in FP32 (`atomicAdd` on `half` is lossy/serialized; do the
reduction in FP32 and down-convert in the workspace variant instead).

For production code, **don't hand-roll this** — let the library pick `SplitK`:

```cpp
// hipBLASLt: expose Split-K as an algo/tuning knob (GSU) and let the
// heuristic choose it for small-M·N, large-K shapes.
hipblasLtMatmulPreference_t pref;
hipblasLtMatmulPreferenceCreate(&pref);
// workspace must be large enough for the reduction (split-K) variant
size_t ws_bytes = 64ull * 1024 * 1024;
hipblasLtMatmulPreferenceSetAttribute(
    pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes));

int returned = 0;
hipblasLtMatmulHeuristicResult_t algos[16];
hipblasLtMatmulAlgoGetHeuristic(handle, matmul, A_desc, B_desc, C_desc, D_desc,
                                pref, 16, algos, &returned);
// algos[i] may encode GlobalSplitU > 1; benchmark and pick the fastest.
```

## Choosing SplitK

`SplitK` is a tuning parameter, not a free win. Rules of thumb:

- **Target full occupancy, not more.** Pick the smallest `SplitK` such that
  `T · SplitK ≳ numCUs` (304 on MI300X, 256 on MI355X). Over-splitting shrinks
  each slice's K-loop until MFMA pipeline fill/drain and the reduction dominate.
- **Keep each slice's K large enough to amortize the MFMA pipeline.** If a slice
  ends up with only a few `BK` iterations, the [MFMA pipeline](mfma-pipelining.md)
  never reaches steady state and you pay startup latency repeatedly.
- **Mind the reduction cost.** Atomic Split-K adds contention on hot `C` lines
  (and atomics serialize across XCDs because [L2 is per-XCD](../hardware/chiplet-xcd.md));
  workspace Split-K adds a memory-bound reduction kernel reading
  `SplitK · M · N` floats. Both grow linearly with `SplitK`.
- **Powers of two** (`SplitK ∈ {2,4,8,16}`) keep the K split even and the
  remainder handling simple. Tensile enumerates `GlobalSplitU` candidates and
  tunes them per shape in its solution catalog.

`SplitK = 1` is just an ordinary GEMM. The technique only pays off when the base
tile count `T` is below the CU count.

## Split-K vs Stream-K

Split-K fixes a *fixed, uniform* number of K-partitions chosen at launch. It can
still leave a [tail effect](../patterns/tail-effect.md) when `T · SplitK` is not
a multiple of the CU count, and it must pick one `SplitK` for the whole problem.
[Stream-K](stream-k.md) generalizes this: it assigns a flat range of MAC-loop
iterations to each of exactly `numCUs` persistent workgroups, so the K work is
balanced across CUs *with no tail*, at the cost of a more complex
fixup/reduction. Use Split-K for its simplicity and library support; reach for
Stream-K when the residual tail from a fixed split is still costing you.

## Pitfalls

- **`C` must be zeroed before atomic Split-K** (or fold `beta·C` into exactly one
  slice). Forgetting this corrupts results silently.
- **Accumulate in FP32.** Summing `SplitK` partials in FP16/BF16 loses precision;
  the matrix core already accumulates in FP32 — keep it there until the final
  store. This matters most for FP8 GEMM where each partial is already low-range.
- **Non-determinism.** Atomic Split-K reorders the reduction, so results are not
  bit-reproducible run-to-run. Use the workspace variant when you need
  determinism.
- **Workspace sizing.** Reduction Split-K silently falls back to `SplitK=1` (or
  errors) if the provided workspace is too small — size it for the largest
  `SplitK` the heuristic may select.

## See also

- [CK FP16 GEMM kernel](../kernels/ck-hgemm.md) — where `k_batch`/GlobalSplitU is wired in
- [Stream-K scheduling](stream-k.md)
- [Persistent kernels](persistent-kernel.md)
- [Occupancy tuning](occupancy-tuning.md)
- [MFMA matrix cores](../hardware/mfma.md)

## Sources

- [Tensile — GlobalSplitU GEMM solution generator (rocBLAS backend)](https://github.com/ROCm/Tensile)
- [Optimizing GEMM on AMD GPUs (ROCm Blogs)](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores/README.html)
- [Composable Kernel — GEMM with k_batch / GlobalSplitU](https://github.com/ROCm/composable_kernel)
- [hipBLASLt — GEMM tuning and GSU knob](https://github.com/ROCm/hipBLASLt)
- [AMD Instinct MI300X datasheet (304 CU / 8 XCD)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)
