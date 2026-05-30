---
id: kernel-ck-hgemm
title: "FP16 GEMM via Composable Kernel / MFMA on MI300X"
type: kernel
architectures:
- gfx942
- gfx950
tags:
- gemm
- hgemm
- fp16
- bf16
- mfma
- composable-kernel
- lds-double-buffering
- mfma-pipelining
confidence: source-reported
reproducibility: runnable
kernel_types:
- gemm
- hgemm
languages:
- composable-kernel
- hip
related:
- hw-mfma
- technique-lds-double-buffering
- technique-mfma-pipelining
- lang-composable-kernel
- technique-split-k
- pattern-mfma-underutilized
sources:
- ref-composable-kernel
- blog-cktile-gemm
- hw-mfma
- technique-lds-double-buffering
- doc-mi300x-datasheet
- blog-gemm-optimization
performance_claims:
- gpu: MI300X
  dtype: fp16
  metric: TFLOPS
  value: "~1000-1150 TFLOPS (large compute-bound M=N=K=8192)"
  shape: "8192x8192x8192"
  utilization: "~75-88% of 1307 TFLOPS dense FP16 peak"
  source_id: blog-cktile-gemm
  baseline: hipBLASLt
- gpu: MI300X
  dtype: bf16
  metric: TFLOPS
  value: "~1000-1150 TFLOPS (compute-bound)"
  shape: "8192x8192x8192"
  utilization: "~75-88% of 1307 TFLOPS dense BF16 peak"
  source_id: doc-mi300x-datasheet
- gpu: MI300X
  dtype: fp16
  metric: peak-TFLOPS-reference
  value: "1307 TFLOPS dense FP16/BF16 (hardware ceiling)"
  source_id: doc-mi300x-datasheet
---

# FP16 GEMM via Composable Kernel / MFMA on MI300X

## Overview

A half-precision GEMM (`hgemm`) computes `C = α·op(A)·op(B) + β·C` with FP16
(or BF16) inputs and an FP32 accumulator. On CDNA3 (gfx942 / MI300X) and CDNA4
(gfx950) the multiply-accumulate is done by the [matrix cores](../hardware/mfma.md)
via `v_mfma_f32_16x16x16_f16` / `v_mfma_f32_32x32x8_f16`, each issued by a full
wavefront. [Composable Kernel (CK / ck_tile)](../languages/composable-kernel.md)
is the HIP C++ tile DSL that AMD ships to generate these kernels; it is the
backend for many ROCm GEMM paths and exposes the full tiling/pipeline policy as
template parameters.

The recurring optimization problem for an FP16 GEMM is feeding the matrix cores
fast enough: each `v_mfma_f32_16x16x16_f16` consumes 8192 FLOPs in 16 cycles, so
a CU running near peak must stream A/B tiles through LDS and overlap that
streaming with MFMA issue. CK does this with **block tiling → warp tiling →
LDS double buffering → MFMA software pipelining**.

## Tiling hierarchy

A CK GEMM decomposes the problem into nested tiles. A representative gfx942
config for a large square GEMM:

| Level | Tile (M×N×K) | Owner |
|---|---|---|
| Block (workgroup) | 256×256×64 | one CU, 4 waves |
| Warp (wavefront)  | 128×128×64 | one wave64 |
| MFMA instruction  | 32×32×8 (or 16×16×16) | matrix core |

Each workgroup loads a `256×64` slab of A and a `64×256` slab of B from HBM into
LDS, then each of its waves repeatedly issues MFMAs over the K sub-tiles,
accumulating into FP32 AGPRs. The accumulator tile per wave is large and lives in
[AGPRs](../hardware/mfma.md#accumulation-registers-agprs), so this kernel is
typically AGPR/occupancy bound — see [MFMA underutilization](../patterns/mfma-underutilized.md).

## CK-tile kernel skeleton

The host side instantiates a device GEMM with a tiling + pipeline policy. The
following compiles against the `ck_tile` headers (ROCm 6.x):

```cpp
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"

using namespace ck_tile;

// FP16 in, FP32 acc, FP16 out, row-major A / col-major B
using GemmConfig = GemmConfigBase;
using TilePartitioner = GemmTile2DPartitioner</*MPerBlock*/256,
                                              /*NPerBlock*/256,
                                              /*KPerBlock*/ 64>;

// Pipeline: GemmPipelineAgBgCrCompV3 = LDS double-buffered, MFMA-pipelined
using GemmPipeline = GemmPipelineAgBgCrCompV3<
    GemmPipelineProblem<half_t,            // A dtype  (fp16)
                        half_t,            // B dtype  (fp16)
                        float,             // acc dtype (fp32)
                        GemmShape<256,256,64>,
                        GemmTraits</*kPadM*/false,/*kPadN*/false,/*kPadK*/true,
                                   /*DoubleSmemBuffer*/true>>>;

using Kernel = GemmKernel<TilePartitioner, GemmPipeline, /*Epilogue*/CShuffleEpilogue>;

// Launch
auto kargs = Kernel::MakeKargs(p_a, p_b, p_c, M, N, K,
                               /*strideA*/K, /*strideB*/K, /*strideC*/N);
const dim3 grid  = Kernel::GridSize(M, N, /*kbatch*/1);
const dim3 block = Kernel::BlockSize();                  // 256 threads = 4 wave64
launch_kernel(stream_config{stream},
              make_kernel<Kernel>(Kernel{}, grid, block, 0, kargs));
```

The `V3` ("CompV3") pipeline name encodes the schedule: **double-buffered LDS**
plus **MFMA software pipelining**. The two are the load-bearing optimizations.

## The inner loop: double buffer + MFMA pipeline

What the generated code does each K-iteration, expressed as pseudo-HIP:

```cpp
// Two LDS buffers per operand so global loads for tile k+1 overlap
// the MFMA math on tile k. Direct-to-LDS bypasses VGPRs on the load.
__shared__ half lds_a[2][256 * 64];
__shared__ half lds_b[2][64 * 256];

int rd = 0, wr = 1;
load_tile_to_lds(lds_a[rd], A, k0);          // prologue: fill buffer 0
load_tile_to_lds(lds_b[rd], B, k0);
__syncthreads();

for (int k = k0 + KPerBlock; k < K; k += KPerBlock) {
    load_tile_to_lds(lds_a[wr], A, k);       // VMEM: stream next tile
    load_tile_to_lds(lds_b[wr], B, k);       // (no s_waitcnt yet)

    // While that DMA is in flight, run MFMAs on the buffer we already have.
    #pragma unroll
    for (int kk = 0; kk < KPerBlock; kk += 8)          // 32x32x8_f16 step
        acc = __builtin_amdgcn_mfma_f32_32x32x8f16(
                  read_a(lds_a[rd], kk), read_b(lds_b[rd], kk), acc, 0, 0, 0);

    s_waitcnt_vmcnt(0);                       // now the next tile has landed
    __syncthreads();
    rd ^= 1; wr ^= 1;                         // swap buffers
}
```

Key mechanics, all grounded in the hardware pages:

- **[LDS double buffering](../techniques/lds-double-buffering.md)** — two LDS
  buffers let the `buffer_load`/`global_load` for tile *k+1* run concurrently
  with the MFMA math on tile *k*. Gating is via `s_waitcnt vmcnt(0)`, not a
  barrier object (see [s_waitcnt](../hardware/s-waitcnt.md)). gfx942 has 64 kB
  LDS/CU; double-buffering a 256×64 + 64×256 FP16 tile pair = 2×(256·64·2 B) per
  operand = 64 kB, so realistic configs trim `KPerBlock` or `MPerBlock` to leave
  room for occupancy.
- **[MFMA pipelining](../techniques/mfma-pipelining.md)** — the `#pragma unroll`
  K-loop interleaves `ds_read` of the next MFMA's operands with the current
  MFMA, hiding LDS read latency behind matrix-core issue.
- **Direct-to-LDS** — `load_tile_to_lds` lowers to `buffer_load … lds`, copying
  HBM→LDS without staging through VGPRs ([async copy](../hardware/async-copy-lds.md)),
  which frees ArchVGPRs for addressing and keeps the AGPR-heavy accumulator tile
  resident.
- **CShuffle epilogue** — the FP32 accumulator (in AGPRs, transposed per the
  MFMA output layout) is shuffled through LDS to produce coalesced FP16 stores
  to C.

## Choosing the MFMA shape

Both `16×16×16` and `32×32×8` FP16 MFMAs are available on gfx942. `32×32×8`
amortizes instruction-issue overhead and tends to win for large compute-bound
GEMMs; `16×16×16` gives finer granularity and lower AGPR footprint, which helps
skinny / low-occupancy shapes. CK exposes this as a template parameter, and the
exact register mapping for either can be dumped with the
[Matrix Instruction Calculator](../../sources/refs/ref-matrix-calculator.md):

```bash
python3 matrix_calculator.py --architecture cdna3 \
    --instruction v_mfma_f32_32x32x8_f16 --detail-instruction
```

## Tuning checklist

- **Tile size vs occupancy** — bigger block tiles → more data reuse but more
  LDS + AGPRs → fewer waves/CU. Sweep `{256×256, 256×128, 128×128}` block tiles.
- **K-batch / [split-K](../techniques/split-k.md)** — for small M·N but large K,
  partition K across workgroups and reduce partials; otherwise few CUs are busy.
- **Padding** — set `kPadK=true` when K is not a multiple of `KPerBlock`; CK
  emits [buffer-OOB guards](../techniques/buffer-oob-guard.md) so out-of-range
  loads return 0 with no branch.
- **gfx950 notes** — CDNA4 has 160 kB LDS/CU, enabling deeper double/triple
  buffering, and `v_mfma_f32_16x16x32_f16` / `32×32×16_f16` (wider K) which halve
  the MFMA count for the same K. Retune tile sizes when porting; see the
  [gfx942→gfx950 migration](../migration/gfx942-to-gfx950.md).

## Performance

On MI300X the dense FP16/BF16 matrix-core ceiling is **1307 TFLOPS**
([datasheet](../../sources/docs/doc-mi300x-datasheet.md)). A well-tuned CK FP16
GEMM on large compute-bound shapes (e.g. M=N=K=8192) reaches roughly
**1000–1150 TFLOPS**, i.e. ~75–88% of peak, competitive with hipBLASLt
([CK-tile GEMM blog](https://rocm.blogs.amd.com/)). The absolute TFLOPS are
source-reported; the utilization band is inferred from the ratio to peak.
Achievable efficiency drops sharply for memory-bound shapes (small M or N, large
K), where the kernel becomes HBM-bandwidth bound (5.3 TB/s) rather than MFMA
bound — split-K and tile resizing are the levers there.

## Sources

- [Composable Kernel (ROCm/composable_kernel)](https://github.com/ROCm/composable_kernel)
- [CK-tile: Building Efficient GEMM Kernels on AMD GPUs](https://rocm.blogs.amd.com/)
- [MFMA — AMD Matrix Core Instructions](../hardware/mfma.md)
- [LDS double buffering technique](../techniques/lds-double-buffering.md)
- [AMD Instinct MI300X datasheet](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [Optimizing GEMM on AMD GPUs (ROCm blog)](https://rocm.blogs.amd.com/)
