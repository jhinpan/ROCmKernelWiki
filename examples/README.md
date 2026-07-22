# Runnable Kernel Examples

Each directory is a self-contained, **hipcc-compilable** example backing a
`wiki/kernels/*.md` page. Every example ships its source, a `build.sh`, and a
`README.md` with the **real captured output**.

## gfx950 verification

Fresh runs on an **AMD Instinct MI355X (gfx950)** completed all 12 `build.sh`
scripts with exit code zero. Every runnable path reported `PASS`; that does not
mean every source file is a numeric test:

- `fp8-gemm/fp8_gemm_cdna.cpp` is compiler/ISA-only. Its host `main()` does not
  launch the FP8 kernels. The directory's rocWMMA FP16 fallback does run and
  self-check on gfx950.
- `flydsl-preshuffle-gemm/04_preshuffle_gemm_flydsl.py` is reference-only. The
  rocWMMA C++ demo runs and checks both kernels.
- `vector-add-asm/vadd_asm_gfx942.cpp` is cross-compiled to a gfx942 object. The
  portable HIP path is the part run on gfx950.

rocWMMA is the C++ fragment **API** used by several examples. On gfx950 those
operations are emitted as **MFMA** instructions; “rocWMMA” does not mean the
hardware executes an RDNA WMMA instruction.

## Build all

```bash
for d in */; do (cd "$d" && bash build.sh) || echo "FAIL $d"; done
```

## Index

| Example | Wiki page | MI355X / gfx950 evidence |
|---|---|---|
| `rmsnorm/` | kernel-rmsnorm | Five fp32/fp16 cases passed |
| `bandwidth-microbench/` | kernel-bandwidth-microbench | ISA check and sum check passed |
| `transpose-lds/` | kernel-transpose-lds | Exact transpose, zero mismatches |
| `ck-hgemm/` | kernel-ck-hgemm | rocWMMA API FP16 GEMM passed |
| `grouped-gemm/` | kernel-grouped-gemm | All six uneven groups passed |
| `flash-attention-ck/` | kernel-flash-attention-ck | CPU softmax comparison passed |
| `paged-attention/` | kernel-paged-attention | CPU reference comparison passed |
| `mla-decode/` | kernel-mla-decode | CPU reference comparison passed |
| `fused-moe/` | kernel-fused-moe | CPU reference comparison passed |
| `vector-add-asm/` | kernel-vector-add-asm | HIP path passed; gfx942 asm object built |
| `flydsl-preshuffle-gemm/` | kernel-flydsl-preshuffle-gemm | rocWMMA kernels passed; Python is reference-only |
| `fp8-gemm/` | kernel-fp8-gemm | FP16 fallback passed; FP8 ISA probes built |
