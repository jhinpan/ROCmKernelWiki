# Diff summary

- **files changed:** 39 (diff was byte-capped; summary is partial)
- **lines:** +5372 / -226
- **kernel-ish files:** 35

## Files (by churn)

- `include/ck_tile/ops/elementwise/unary_element_wise_operation.hpp`  (+1163/-0)
- `include/ck_tile/core/tensor/tile_window_linear.hpp`  (+1082/-0)
- `include/ck_tile/core/numeric/math.hpp`  (+929/-43)
- `example/ck_tile/09_topk_softmax/topk_softmax.cpp`  (+299/-0)
- `include/ck_tile/core/tensor/tensor_view.hpp`  (+192/-25)
- `include/ck_tile/core/arch/amd_buffer_addressing.hpp`  (+195/-18)
- `include/ck_tile/core/tensor/tile_window.hpp`  (+171/-39)
- `include/ck_tile/core/tensor/buffer_view.hpp`  (+121/-57)
- `include/ck_tile/ops/reduce/block/block_reduce.hpp`  (+170/-0)
- `include/ck_tile/ops/topk_softmax/kernel/topk_softmax_kernel.hpp`  (+166/-0)
- `include/ck_tile/host/reference/reference_topk.hpp`  (+124/-0)
- `include/ck_tile/ops/topk/block/block_topk_stream_2d.hpp`  (+113/-0)
- `example/ck_tile/09_topk_softmax/topk_softmax_api.cpp`  (+96/-0)
- `include/ck_tile/ops/softmax/block/block_softmax_2d.hpp`  (+81/-0)
- `include/ck_tile/host/reference/reference_softmax.hpp`  (+59/-21)

## Key added lines (kernel files)

**`example/ck_tile/09_topk_softmax/topk_softmax.cpp`**
```
template <typename T>
void dump_host_tensor_2d(const ck_tile::HostTensor<T>& x)
auto len = x.get_lengths();
assert(len.size() == 2);
```

**`example/ck_tile/09_topk_softmax/topk_softmax_api.cpp`**
```
constexpr ck_tile::index_t ts_experts = experts_;                                           \
using ts_problem                      = ck_tile::                                           \
TopkSoftmaxWarpPerRowProblem<ts_input_type, ts_weight_type, ts_index_type, ts_experts>; \
using ts_pipeline = ck_tile::TopkSoftmaxWarpPerRowPipeline<ts_problem>;                     \
```

**`example/ck_tile/09_topk_softmax/topk_softmax_api.hpp`**
```
struct topk_softmax_trait
std::string input_type;
std::string weight_type; // currently always float
int experts;
```

**`include/ck_tile/core/algorithm/space_filling_curve.hpp`**
```
static CK_TILE_HOST_DEVICE constexpr Index _get_index(number<AccessIdx1d>)
static CK_TILE_HOST_DEVICE constexpr auto get_index(number<AccessIdx1d>)
constexpr auto idx = _get_index(number<AccessIdx1d>{});
```

**`include/ck_tile/core/arch/amd_buffer_addressing.hpp`**
```
namespace impl {
template<index_t N, typename T> struct smem_load_trait;
template<typename T> struct smem_load_trait<16, T> { using payload_t = fp32x4_t; };
template<typename T> struct smem_load_trait<8 , T> { using payload_t = fp32x2_t; };
```
