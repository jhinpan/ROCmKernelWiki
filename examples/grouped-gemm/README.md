# Grouped GEMM (portable rocWMMA) — many uneven problems in one launch

A **grouped GEMM** computes a batch of *independent* matrix multiplies whose
shapes differ per group, in a **single kernel launch**:

```
for g in 0..G-1:   C_g[Mg, Ng] = A_g[Mg, Kg] · B_g[Kg, Ng]
```

This is the shape an MoE router produces (each expert `g` gets `Mg` tokens, and
`Mg` is data-dependent and uneven). This example demonstrates the *scheduling
core* of a grouped GEMM with portable rocWMMA so it actually runs on this box:

- A per-group **descriptor table** (`GroupDesc`) holds each group's logical
  `M,N,K`, padded leading dims, flat buffer offsets, its `ceil(N/16)` tile count,
  and a **prefix-sum tile base**.
- The kernel launches `total_tiles` blocks (one wave each). Each block reads its
  flat `tile_id = blockIdx.x`, finds the owning group from the prefix sum, and
  decodes `(m_tile, n_tile)`. **One launch covers all G problems** regardless of
  how lopsided the sizes are.
- The inner 16×16 tile uses rocWMMA `fragment` / `mma_sync` — **fp16 in, fp32
  accumulate** — which maps to RDNA4 **WMMA** on gfx1201 and to **MFMA** on CDNA.
- Ragged sizes (e.g. `17×33×49`) are handled by zero-padding each matrix's
  storage up to a multiple of 16, so rocWMMA's loads stay in range and the
  padding contributes 0. The CPU reference and PASS check use the **logical**
  (unpadded) sizes.

## Classification

**PORTABLE** — pure HIP + rocWMMA. Builds **and runs** on gfx1201 (RX 9070 XT,
RDNA4). The same source is portable to CDNA (gfx942/gfx950) where rocWMMA emits
MFMA. Self-checks every group against a CPU reference.

## Build & run

```bash
./build.sh
```

which runs:

```bash
hipcc --offload-arch=gfx1201 -O3 -std=c++17 -I/opt/rocm/include \
      grouped_gemm_wmma.cpp -o grouped_gemm_wmma
./grouped_gemm_wmma
```

## Expected output (captured on gfx1201, ROCm 7.2.3)

```
Build OK — running:
Device: AMD Radeon RX 9070 XT  warpSize=32
Groups: 6   total 16x16 output tiles (one launch): 71
  group 0  M= 64 N= 48 K= 80  tiles=12  max|err|=3.8147e-06  ok
  group 1  M= 32 N= 96 K= 32  tiles=12  max|err|=1.4305e-06  ok
  group 2  M=128 N= 16 K= 64  tiles= 8  max|err|=3.8147e-06  ok
  group 3  M= 16 N=128 K=112  tiles= 8  max|err|=6.6757e-06  ok
  group 4  M= 80 N= 80 K= 48  tiles=25  max|err|=2.3842e-06  ok
  group 5  M= 17 N= 33 K= 49  tiles= 6  max|err|=1.9073e-06  ok
Avg kernel time: 0.0061 ms  (340.0 GFLOP/s aggregate over all groups)
Overall max abs error: 6.6757e-06
PASS
```

(Timing is on these tiny demo shapes — it measures launch/scheduling overhead,
not steady-state GEMM throughput. The point of the example is correctness of the
single-launch flattened-tile scheduling across uneven groups.)

## Notes / how this maps to production grouped GEMM

- The flattened `tile_id → (group, m_tile, n_tile)` decode here is the same idea
  CK/AITER use, except production kernels use a **persistent kernel + atomic tile
  counter** (one workgroup per CU pulling tiles) so skewed `Mg` load-balances
  dynamically. This example uses the simpler static `grid = total_tiles` mapping,
  which is enough to demonstrate the descriptor-table + prefix-sum scheme.
- Production CDNA kernels also use `buffer_load` OOB semantics for the ragged
  `Mg` remainder tile instead of zero-padding storage; padding is the portable,
  rocWMMA-friendly choice here.
