# Diff summary

- **files changed:** 17
- **lines:** +3096 / -19
- **kernel-ish files:** 15

## Files (by churn)

- `include/ck_tile/ops/grouped_convolution/utils/transform_conv_fwd_to_gemm.hpp`  (+1432/-0)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp`  (+800/-0)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward.cpp`  (+207/-0)
- `example/ck_tile/20_grouped_convolution/run_grouped_convolution_example.inc`  (+206/-0)
- `include/ck_tile/host/reference/reference_grouped_conv_fwd.hpp`  (+165/-0)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_utils.hpp`  (+108/-0)
- `include/ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp`  (+74/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`  (+32/-6)
- `include/ck_tile/ops/grouped_convolution/utils/convolution_specialization.hpp`  (+30/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`  (+10/-8)
- `include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`  (+11/-1)
- `include/ck_tile/ops/grouped_convolution.hpp`  (+12/-0)
- `example/ck_tile/20_grouped_convolution/CMakeLists.txt`  (+4/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp`  (+2/-2)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v5.hpp`  (+1/-2)

## Key added lines (kernel files)

**`example/ck_tile/20_grouped_convolution/grouped_convolution_forward.cpp`**
```
template <ck_tile::index_t NDimSpatial,
typename InDataType,
typename WeiDataType,
typename AccDataType,
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_utils.hpp`**
```
ck_tile::index_t fill_spatial_dimensions(std::vector<ck_tile::index_t>& filter_spatial_lengths,
std::vector<ck_tile::index_t>& image_spatial_lengths,
std::vector<ck_tile::index_t>& strides,
std::vector<ck_tile::index_t>& dilations,
```

**`example/ck_tile/20_grouped_convolution/run_grouped_convolution_example.inc`**
```
template <typename InDataType, typename WeiDataType, typename AccDataType, typename OutDataType>
auto calculate_rtol_atol(const ck_tile::index_t GemmK,
const ck_tile::index_t kbatch,
const float max_accumulated_value)
```

**`include/ck_tile/host/reference/reference_grouped_conv_fwd.hpp`**
```
namespace ck_tile {
template <ck_tile::index_t NDimSpatial,
typename InDataType,
typename WeiDataType,
```

**`include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`**
```
index_t kNumWaveGroups_ = 1,
bool FixedVectorSize_   = false,
index_t VectorSizeC_    = 1>
static constexpr bool FixedVectorSize                  = FixedVectorSize_;
```
