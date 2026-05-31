# Diff summary

- **files changed:** 7
- **lines:** +191 / -183
- **kernel-ish files:** 6

## Files (by churn)

- `aiter/configs/tuned_fmoe.csv`  (+160/-169)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_blockscale.cuh`  (+7/-7)
- `csrc/ck_gemm_moe_2stages_codegen/gen_instances.py`  (+10/-2)
- `op_tests/test_moe_2stage.py`  (+9/-0)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py`  (+2/-2)
- `hsa/gfx942/fmoe_2stages/tune.py`  (+2/-2)
- `hsa/gfx950/fmoe_2stages/tune.py`  (+1/-1)

## Key added lines (kernel files)

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py`**
```
1: kernelInstanceGEMM1(       256,       16,        128,       256,     1,       4,        1,),
0: kernelInstanceGEMM2(       256,       16,        128,       256,     1,       4,        1,),
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_blockscale.cuh`**
```
MXDLPerWave,    NXDLPerWave,
MXDLPerWave,    NXDLPerWave,   S<1, K0_M_A, 1, K0_A>, S<2, 1, 1, 1>,
MPerBlock,   NPerBlock,    KPerBlock,
MXDLPerWave, NXDLPerWave,
```

**`csrc/ck_gemm_moe_2stages_codegen/gen_instances.py`**
```
if (block_m == 16)
return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 16, 128, 256/
else if (block_m == 64)
if (block_m == 16)
```

**`hsa/gfx942/fmoe_2stages/tune.py`**
```
if blockM in [16, 32, 64, 128] and use_g1u1:
blockMs = [16, 32, 64, 128]
```

**`hsa/gfx950/fmoe_2stages/tune.py`**
```
blockMs = [16, 32, 64, 128]
```
