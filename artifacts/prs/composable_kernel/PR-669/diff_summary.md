# Diff summary

- **files changed:** 17
- **lines:** +1329 / -40
- **kernel-ish files:** 15

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_splitk_cshuffle.hpp`  (+612/-0)
- `example/15_grouped_gemm/grouped_gemm_xdl_splitk_fp16.cpp`  (+97/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4r2.hpp`  (+78/-9)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_irregular_instance.cpp`  (+87/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_irregular_instance.cpp`  (+81/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_instance.cpp`  (+80/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_instance.cpp`  (+75/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm.hpp`  (+58/-0)
- `include/ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp`  (+48/-0)
- `profiler/src/profile_grouped_gemm.cpp`  (+26/-17)
- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_splitk.hpp`  (+39/-0)
- `profiler/include/profiler/profile_grouped_gemm_impl.hpp`  (+26/-5)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp`  (+13/-7)
- `example/15_grouped_gemm/CMakeLists.txt`  (+3/-1)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/CMakeLists.txt`  (+4/-0)

## Key added lines (kernel files)

**`example/15_grouped_gemm/grouped_gemm_xdl_splitk_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`example/15_grouped_gemm/run_grouped_gemm_example.inc`**
```
c_tensors_device[i]->SetZero();
```

**`include/ck/tensor_operation/gpu/device/device_grouped_gemm.hpp`**
```
static_assert(DsLayout::Size() == DsDataType::Size(), "wrong! inconsistent NumDTensor");
```

**`include/ck/tensor_operation/gpu/device/device_grouped_gemm_splitk.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename ALayout,
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp`**
```
using Argument              = typename GridwiseGemm::Argument;
using DefaultBlock2CTileMap = typename GridwiseGemm::DefaultBlock2CTileMap;
const auto b2c_map = DefaultBlock2CTileMap{};
std::tie(gdx, gdy, gdz) = b2c_map.CalculateGridSize(karg.M, karg.N, karg.k_batch);
```
