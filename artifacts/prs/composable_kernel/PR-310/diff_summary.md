# Diff summary

- **files changed:** 15
- **lines:** +1652 / -6
- **kernel-ish files:** 13

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_waveletmodel_cshuffle.hpp`  (+744/-0)
- `include/ck/tensor_operation/gpu/device/device_gemm_xdl_waveletmodel_cshuffle.hpp`  (+524/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_waveletmodel.hpp`  (+157/-0)
- `test/gemm/instance/gemm_wavelet_f16_tn_instance.cpp`  (+96/-0)
- `include/ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp`  (+44/-0)
- `example/01_gemm/gemm_xdl_wavelet_fp16.cpp`  (+42/-0)
- `test/gemm/instance/gemm_wavelet_f16_tn_instance.hpp`  (+25/-0)
- `example/01_gemm/gemm_xdl_fp16.cpp`  (+4/-2)
- `include/ck/ck.hpp`  (+5/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp`  (+2/-3)
- `test/gemm/gemm_standalone_xdl_fp16.cpp`  (+5/-0)
- `example/01_gemm/CMakeLists.txt`  (+2/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp`  (+0/-1)
- `include/ck/utility/synchronization.hpp`  (+1/-0)
- `test/gemm/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/01_gemm/gemm_xdl_fp16.cpp`**
```
using F16 = ck::half_t;
using DeviceGemmInstance = DeviceGemmInstance1;
```

**`example/01_gemm/gemm_xdl_wavelet_fp16.cpp`**
```
using ADataType        = ck::half_t;
using BDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = float;
```

**`include/ck/tensor_operation/gpu/device/device_gemm_xdl_waveletmodel_cshuffle.hpp`**
```
namespace ck {
template <typename GridwiseGemm,
typename ABDataType,
typename EDataType,
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp`**
```
const auto K = arg.a_grid_desc_m_k_.GetLength(I1);
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_waveletmodel.hpp`**
```
namespace ck {
template <typename TileLoadThreadGroup, index_t NumGemmKPrefetchStage>
struct GridwiseGemmLoadWave;
template <typename TileLoadThreadGroup>
```
