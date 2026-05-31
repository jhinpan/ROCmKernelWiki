# Diff summary

- **files changed:** 28
- **lines:** +2030 / -1049
- **kernel-ish files:** 26

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_dpp.hpp`  (+701/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_dl_dpp8.hpp`  (+0/-370)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_dpp.hpp`  (+348/-0)
- `include/ck/tensor_operation/gpu/warp/dpp_gemm.hpp`  (+322/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_dpp.hpp`  (+271/-0)
- `include/ck/tensor_operation/gpu/thread/threadwise_contraction_dl_dpp8.hpp`  (+0/-136)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_dl_dpp8.hpp`  (+0/-133)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_dl_v1r3.hpp`  (+16/-55)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_dpp8_f16_f16_f16_km_kn_mn_instance.cpp`  (+0/-61)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_dpp8_f16_f16_f16_km_nk_mn_instance.cpp`  (+0/-61)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_dpp8_f16_f16_f16_mk_nk_mn_instance.cpp`  (+0/-61)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_dpp8_f16_f16_f16_mk_kn_mn_instance.cpp`  (+0/-60)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dpp_f16_f16_f16_km_kn_mn_instance.cpp`  (+58/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dpp_f16_f16_f16_km_nk_mn_instance.cpp`  (+58/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dpp_f16_f16_f16_mk_nk_mn_instance.cpp`  (+58/-0)

## Key added lines (kernel files)

**`example/01_gemm/gemm_dpp_fp16.cpp`**
```
using ADataType   = ck::half_t;
using BDataType   = ck::half_t;
using AccDataType = float;
using CDataType   = ck::half_t;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_dpp.hpp`**
```
namespace ck {
template <index_t BlockSize,
typename ABDataType,
typename AccDataType,
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_dl.hpp`**
```
CThreadTransferDstScalarPerVector>;
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_dpp.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename ADataType,
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_dl_v1r3.hpp`**
```
bool HasDoubleTailKBlockLoop>
index_t CThreadTransferDstScalarPerVector>
BlockwiseGemmDl_A_BK0_BM_BK1_B_BK0_BN_BK1_C_BM0_BM1_BN0_BN1_pipeline_BM0_2_BN0_2<
BlockSize,
```
