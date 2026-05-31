# Diff summary

- **files changed:** 28
- **lines:** +47 / -35
- **kernel-ish files:** 4

## Files (by churn)

- `op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py`  (+8/-8)
- `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_afp4wfp4.py`  (+7/-1)
- `aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_afp4wfp4_mul_add.py`  (+7/-1)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-AFP4WFP4_PRESHUFFLED-N=10240-K=8192.json`  (+1/-1)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-AFP4WFP4_PRESHUFFLED-N=106496-K=16384.json`  (+1/-1)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-AFP4WFP4_PRESHUFFLED-N=1280-K=8192.json`  (+1/-1)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-AFP4WFP4_PRESHUFFLED-N=14336-K=8192.json`  (+1/-1)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-AFP4WFP4_PRESHUFFLED-N=16384-K=16384.json`  (+1/-1)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-AFP4WFP4_PRESHUFFLED-N=16384-K=53248.json`  (+1/-1)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-AFP4WFP4_PRESHUFFLED-N=18432-K=16384.json`  (+1/-1)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-AFP4WFP4_PRESHUFFLED-N=2112-K=7168.json`  (+1/-1)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-AFP4WFP4_PRESHUFFLED-N=2560-K=8192.json`  (+1/-1)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-AFP4WFP4_PRESHUFFLED-N=28672-K=8192.json`  (+1/-1)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-AFP4WFP4_PRESHUFFLED-N=3072-K=1536.json`  (+1/-1)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-AFP4WFP4_PRESHUFFLED-N=4096-K=512.json`  (+1/-1)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_afp4wfp4.py`**
```
return get_gemm_config(
"GEMM-AFP4WFP4_PRESHUFFLED",
bounds=(4, 8, 16, 31, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192),
```

**`aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_afp4wfp4_mul_add.py`**
```
return get_gemm_config(
"GEMM-AFP4WFP4_PRESHUFFLED",
bounds=(4, 8, 16, 31, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192),
```

**`op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py`**
```
x_vals += [(v, 106496, 16384) for v in [1, 8, 16, 31, 32, 64, 128, 256]]
x_vals += [(v, 16384, 53248) for v in [1, 8, 16, 31, 32, 64, 128, 256]]
x_vals += [(v, 18432, 16384) for v in [1, 8, 16, 31, 32, 64, 128, 256]]
x_vals += [(v, 16384, 16384) for v in [1, 8, 16, 31, 32, 64, 128, 256]]
```

**`op_tests/triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_mul_add.py`**
```
x_vals = [(v, 7168, 256) for v in [1, 2, 4, 8, 16, 31, 32, 1024]]
```
