# Runnable Kernel Examples

Each directory is a self-contained, **hipcc-compilable** example backing a
`wiki/kernels/*.md` page. Every example ships its source, a `build.sh`, and a
`README.md` with the **real captured output**.

## Two classes (see each README)

- **Portable** (pure HIP or rocWMMA): builds **and runs** on this RDNA4 / gfx1201
  box and self-checks numerics against a CPU reference. rocWMMA examples use the
  same fragment API that maps to **MFMA on CDNA** and **WMMA on RDNA**, so they
  are also valid on MI300/MI350.
- **CDNA-MFMA** (`__builtin_amdgcn_mfma_*`, gfx950 `f8f6f4`, gfx942 hand-asm):
  use the matrix-core path. Where useful, these dirs also include a portable
  rocWMMA/HIP variant that runs on gfx1201.

> **Verified on MI350X (gfx950), ROCm 7.2 (2026-06-01).** All 12 examples now
> build with `--offload-arch=gfx950` and **execute on real CDNA4 silicon** — not
> just cross-compiled. 11/12 self-check PASS on-device; the lone exception is
> `fp8-gemm`, whose `main()` only verifies that `v_mfma_scale_f32_16x16x128_f8f6f4`
> is emitted and **does not launch a numeric FP8 GEMM**. See
> [`../VERIFICATION.md`](../VERIFICATION.md).

## Build all

```bash
for d in */; do (cd "$d" && bash build.sh) || echo "FAIL $d"; done
```

## Index

| Example | Wiki page | Class | Runs on gfx1201 | Runs on MI350X (gfx950) |
|---|---|---|---|---|
| `rmsnorm/` | kernel-rmsnorm | portable HIP | ✅ PASS, ~600 GB/s | ✅ PASS (5 cases) |
| `bandwidth-microbench/` | kernel-bandwidth-microbench | portable HIP | ✅ measured GB/s | ✅ PASS, ~6.2–6.3 TB/s |
| `transpose-lds/` | kernel-transpose-lds | portable HIP + LDS | ✅ 0 mismatches | ✅ PASS, ~4.1 TB/s |
| `ck-hgemm/` | kernel-ck-hgemm | portable rocWMMA FP16 | ✅ ~5 TFLOPS, verified | ✅ PASS, max abs err 0 |
| `grouped-gemm/` | kernel-grouped-gemm | portable rocWMMA | ✅ verified | ✅ PASS, warpSize=64 |
| `flash-attention-ck/` | kernel-flash-attention-ck | portable HIP FA-2 fwd | ✅ vs CPU softmax | ✅ PASS |
| `paged-attention/` | kernel-paged-attention | portable HIP | ✅ PASS | ✅ PASS, 0.045 ms |
| `mla-decode/` | kernel-mla-decode | portable HIP | ✅ PASS | ✅ PASS |
| `fused-moe/` | kernel-fused-moe | portable HIP | ✅ PASS | ✅ PASS, 122 µs/iter |
| `vector-add-asm/` | kernel-vector-add-asm | HIP (runs) + gfx942 asm (xcompile) | ✅ HIP part | ✅ HIP part, 6.8 TB/s |
| `flydsl-preshuffle-gemm/` | kernel-flydsl-preshuffle-gemm | rocWMMA (runs) + FlyDSL .py (ref) | ✅ rocWMMA part | ✅ rocWMMA part, exact |
| `fp8-gemm/` | kernel-fp8-gemm | gfx950 FP8 MFMA + rocWMMA FP16 (runs) | ⚠️ FP8 on MI350; FP16 runs | ⚠️ builds+runs, but `main()` runs no numeric check |

> Captured on AMD Radeon RX 9070 XT (gfx1201, RDNA4), ROCm 7.2.3. Numbers are
> illustrative of this consumer part — datacenter MI300/MI350 figures differ; see
> each kernel page's `performance_claims`.
