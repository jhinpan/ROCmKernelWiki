# Diff summary

- **files changed:** 50
- **lines:** +2586 / -322
- **kernel-ish files:** 43

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/device_gemm_dl.hpp`  (+586/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_dl_v1r3.hpp`  (+180/-201)
- `example/01_gemm/gemm_dl_fp16.cpp`  (+211/-0)
- `example/01_gemm/gemm_dl_fp32.cpp`  (+210/-0)
- `example/01_gemm/gemm_dl_int8.cpp`  (+208/-0)
- `test/gemm/gemm_dl_fp16.cpp`  (+130/-0)
- `test/gemm/gemm_dl_fp32.cpp`  (+128/-0)
- `test/gemm/gemm_dl_int8.cpp`  (+128/-0)
- `profiler/include/profile_gemm_impl.hpp`  (+66/-14)
- `test/gemm/gemm_util.hpp`  (+38/-26)
- `include/ck/host_utility/device_prop.hpp`  (+50/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_f16_f16_f16_mk_nk_mn_instance.cpp`  (+46/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_f32_f32_f32_km_nk_mn_instance.cpp`  (+46/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_f32_f32_f32_mk_kn_mn_instance.cpp`  (+46/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_f32_f32_f32_mk_nk_mn_instance.cpp`  (+46/-0)

## Key added lines (kernel files)

**`example/01_gemm/gemm_dl_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`example/01_gemm/gemm_dl_fp32.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
```

**`example/01_gemm/gemm_dl_int8.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
```

**`include/ck/host_utility/device_prop.hpp`**
```
namespace ck {
inline std::string get_device_name()
hipDeviceProp_t props{};
int device;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_dl_v2r3.hpp`**
```
struct BlockwiseGemmDl_A_BK0_BM_BK1_B_BK0_BN_BK1_C_BM0_BM1_BN0_BN1_pipeline_BM0_2_BN0_2
__device__ BlockwiseGemmDl_A_BK0_BM_BK1_B_BK0_BN_BK1_C_BM0_BM1_BN0_BN1_pipeline_BM0_2_BN0_2()
static_assert(BM0 == 2, "wrong");
ThreadwiseContractionDl_A_TK0_TM0_TM1_TK1_B_TK0_TN0_TN1_TK1_C_TM0_TM1_TN0_TN1<
```
