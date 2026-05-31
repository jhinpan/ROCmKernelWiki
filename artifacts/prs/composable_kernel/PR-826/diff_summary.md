# Diff summary

- **files changed:** 18
- **lines:** +1217 / -30
- **kernel-ish files:** 16

## Files (by churn)

- `include/ck/tensor_operation/gpu/block/blockwise_gemm_dl_dpp8.hpp`  (+370/-0)
- `include/ck/utility/inner_product_dpp8.hpp`  (+142/-0)
- `include/ck/tensor_operation/gpu/thread/threadwise_contraction_dl_dpp8.hpp`  (+136/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_dl_dpp8.hpp`  (+133/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_dl_v1r3.hpp`  (+58/-19)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_dpp8_f16_f16_f16_km_kn_mn_instance.cpp`  (+61/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_dpp8_f16_f16_f16_km_nk_mn_instance.cpp`  (+61/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_dpp8_f16_f16_f16_mk_nk_mn_instance.cpp`  (+61/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_dpp8_f16_f16_f16_mk_kn_mn_instance.cpp`  (+60/-0)
- `example/01_gemm/gemm_dl_dpp8_fp16.cpp`  (+37/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_dl.hpp`  (+24/-10)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm.hpp`  (+24/-0)
- `include/ck/utility/amd_gemm_dpp.hpp`  (+22/-0)
- `include/ck/tensor_operation/gpu/device/gemm_dl_algorithm.hpp`  (+18/-0)
- `library/src/tensor_operation_instance/gpu/gemm/CMakeLists.txt`  (+4/-0)

## Key added lines (kernel files)

**`example/01_gemm/gemm_dl_dpp8_fp16.cpp`**
```
using ADataType   = ck::half_t;
using BDataType   = ck::half_t;
using CDataType   = ck::half_t;
using AccDataType = float;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_dl_dpp8.hpp`**
```
namespace ck {
template <index_t BlockSize,
typename FloatA,
typename FloatB,
```

**`include/ck/tensor_operation/gpu/device/gemm_dl_algorithm.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
enum struct GemmDlAlgorithm
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_dl.hpp`**
```
GemmDlAlgorithm GemmDlAlg = GemmDlAlgorithm::Default,
CThreadTransferDstScalarPerVector,
GemmDlAlg>;
GemmDlAlg>;
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_dl_dpp8.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <
```
