# Diff summary

- **files changed:** 22
- **lines:** +1051 / -335
- **kernel-ish files:** 21

## Files (by churn)

- `op_tests/test_gemm_codegen.py`  (+469/-0)
- `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale.cu`  (+71/-73)
- `csrc/ck_gemm_a8w8_blockscale_bpreshuffle/gemm_a8w8_blockscale_bpreshuffle.cu`  (+36/-59)
- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale.cu`  (+43/-47)
- `aiter/jit/utils/chip_info.py`  (+73/-16)
- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_cktile.cu`  (+32/-50)
- `csrc/ck_gemm_a4w4_blockscale/gen_instances.py`  (+65/-13)
- `csrc/ck_gemm_a8w8_blockscale_bpreshuffle/gen_instances.py`  (+59/-15)
- `csrc/ck_gemm_a8w8_blockscale/gen_instances.py`  (+55/-13)
- `csrc/ck_gemm_a8w8_blockscale/gen_instances_cktile.py`  (+52/-13)
- `csrc/include/rocm_ops.hpp`  (+29/-24)
- `aiter/ops/gemm_op_a8w8.py`  (+26/-5)
- `aiter/ops/gemm_op_a4w4.py`  (+8/-1)
- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_instance.py`  (+7/-0)
- `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_common.py`  (+5/-0)

## Key added lines (kernel files)

**`aiter/jit/utils/chip_info.py`**
```
lookup uses the name instead of kernelId. Falls back to
kernelId if the kernelName column is absent from the CSV.
Strict on stale tuned-CSV rows: any row whose kernelName (or kernelId, in the
fallback path) is not present in the registry will raise RuntimeError listing
```

**`aiter/ops/gemm_op_a4w4.py`**
```
A.view(m, k // 2),
splitK=splitK,
kernelName=kernelName,
kernelName: str = "",
```

**`aiter/ops/gemm_op_a8w8.py`**
```
kernelName: str = "",
kernelName: str = "",
kernelName: str = "",
kernelName: str = "",
```

**`csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale.cu`**
```
using BlockwiseKernelMap = std::unordered_map<std::string_view, BlockwiseKernel>;
BlockwiseKernel blockscale_dispatch(const std::string& kernelName)
static const auto lookup = [] {
if constexpr(std::is_same_v<CDataType, F16>)
```

**`csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_common.py`**
```
kernels_by_name = {v.name: v for v in kernels_list.values()}
```
