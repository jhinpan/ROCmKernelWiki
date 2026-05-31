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
  **cross-compile-verified** for `gfx942`/`gfx950` here (object/exe builds prove
  correctness of codegen); they **execute on MI300/MI350**, not on gfx1201. Where
  useful, these dirs also include a portable rocWMMA/HIP variant that runs here.

## Build all

```bash
for d in */; do (cd "$d" && bash build.sh) || echo "FAIL $d"; done
```

## Index

| Example | Wiki page | Class | Runs on gfx1201 |
|---|---|---|---|
| `rmsnorm/` | kernel-rmsnorm | portable HIP | ✅ PASS, ~600 GB/s |
| `bandwidth-microbench/` | kernel-bandwidth-microbench | portable HIP | ✅ measured GB/s |
| `transpose-lds/` | kernel-transpose-lds | portable HIP + LDS | ✅ 0 mismatches |
| `ck-hgemm/` | kernel-ck-hgemm | portable rocWMMA FP16 | ✅ ~5 TFLOPS, verified |
| `grouped-gemm/` | kernel-grouped-gemm | portable rocWMMA | ✅ verified |
| `flash-attention-ck/` | kernel-flash-attention-ck | portable HIP FA-2 fwd | ✅ vs CPU softmax |
| `paged-attention/` | kernel-paged-attention | portable HIP | ✅ PASS |
| `mla-decode/` | kernel-mla-decode | portable HIP | ✅ PASS |
| `fused-moe/` | kernel-fused-moe | portable HIP | ✅ PASS |
| `vector-add-asm/` | kernel-vector-add-asm | HIP (runs) + gfx942 asm (xcompile) | ✅ HIP part |
| `flydsl-preshuffle-gemm/` | kernel-flydsl-preshuffle-gemm | rocWMMA (runs) + FlyDSL .py (ref) | ✅ rocWMMA part |
| `fp8-gemm/` | kernel-fp8-gemm | gfx950 FP8 MFMA (xcompile) + rocWMMA FP16 (runs) | ⚠️ FP8 on MI350; FP16 runs |

> Captured on AMD Radeon RX 9070 XT (gfx1201, RDNA4), ROCm 7.2.3. Numbers are
> illustrative of this consumer part — datacenter MI300/MI350 figures differ; see
> each kernel page's `performance_claims`.
