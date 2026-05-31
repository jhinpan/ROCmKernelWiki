# Diff summary

- **files changed:** 33
- **lines:** +483 / -3
- **kernel-ish files:** 33

## Files (by churn)

- `csrc/py_itfs_ck/moe_ck_2stages_kernel.cu`  (+32/-0)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertoken_f16_f8.cu`  (+29/-0)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertensor_f16_i8.cu`  (+27/-0)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertoken_b16_f8.cu`  (+27/-0)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertoken_f16_i8.cu`  (+27/-0)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertensor_b16_f8.cu`  (+26/-0)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertoken_b16_i8.cu`  (+26/-0)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertensor_f16_f8.cu`  (+25/-0)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertensor_b16_i8.cu`  (+24/-0)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm2_instance_pertoken_b16_f8.cu`  (+15/-0)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertoken_b16.cu`  (+14/-0)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertoken_f16.cu`  (+14/-0)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertoken_mulweight_b16.cu`  (+14/-0)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm2_instance_pertoken_b16_i8.cu`  (+14/-0)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm2_instance_pertoken_f16_i8.cu`  (+14/-0)

## Key added lines (kernel files)

**`csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertensor_b16.cu`**
```
CK_MOE_STAGE1_GEMM_DEFINE(96, 128/sizeof(A0DataType), 1, 4, V1, false, 1)
CK_MOE_STAGE1_GEMM_DEFINE(192, 128/sizeof(A0DataType), 1, 4, V1, false, 1)
CK_MOE_STAGE1_GEMM_DEFINE(96, 128/sizeof(A0DataType), 1, 4, V1, false, 0)
CK_MOE_STAGE1_GEMM_DEFINE(192, 128/sizeof(A0DataType), 1, 4, V1, false, 0)
```

**`csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertensor_b16_f8.cu`**
```
CK_MOE_STAGE1_GEMM_DEFINE(96, 128/sizeof(A0DataType), 1, 4, V1, true, 1)
CK_MOE_STAGE1_GEMM_DEFINE(192, 128/sizeof(A0DataType), 1, 4, V1, true, 1)
CK_MOE_STAGE1_GEMM_DEFINE(96, 128/sizeof(A0DataType), 1, 4, V1, true, 0)
CK_MOE_STAGE1_GEMM_DEFINE(192, 128/sizeof(A0DataType), 1, 4, V1, true, 0)
```

**`csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertensor_b16_i8.cu`**
```
CK_MOE_STAGE1_GEMM_DEFINE(96, 128/sizeof(A0DataType), 1, 4, V1, true, 1)
CK_MOE_STAGE1_GEMM_DEFINE(192, 128/sizeof(A0DataType), 1, 4, V1, true, 1)
CK_MOE_STAGE1_GEMM_DEFINE(96, 128/sizeof(A0DataType), 1, 4, V1, true, 0)
CK_MOE_STAGE1_GEMM_DEFINE(192, 128/sizeof(A0DataType), 1, 4, V1, true, 0)
```

**`csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertensor_f16.cu`**
```
CK_MOE_STAGE1_GEMM_DEFINE(96, 128/sizeof(A0DataType), 1, 4, V1, false, 1)
CK_MOE_STAGE1_GEMM_DEFINE(192, 128/sizeof(A0DataType), 1, 4, V1, false, 1)
CK_MOE_STAGE1_GEMM_DEFINE(96, 128/sizeof(A0DataType), 1, 4, V1, false, 0)
CK_MOE_STAGE1_GEMM_DEFINE(192, 128/sizeof(A0DataType), 1, 4, V1, false, 0)
```

**`csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertensor_f16_f8.cu`**
```
CK_MOE_STAGE1_GEMM_DEFINE(96, 128/sizeof(A0DataType), 1, 4, V1, true, 1)
CK_MOE_STAGE1_GEMM_DEFINE(192, 128/sizeof(A0DataType), 1, 4, V1, true, 1)
CK_MOE_STAGE1_GEMM_DEFINE(96, 128/sizeof(A0DataType), 1, 4, V1, true, 0)
CK_MOE_STAGE1_GEMM_DEFINE(192, 128/sizeof(A0DataType), 1, 4, V1, true, 0)
```
