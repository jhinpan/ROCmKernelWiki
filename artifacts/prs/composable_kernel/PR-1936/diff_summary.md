# Diff summary

- **files changed:** 11 (diff was byte-capped; summary is partial)
- **lines:** +5397 / -7
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm.hpp`  (+2144/-0)
- `include/ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v3r1_gather.hpp`  (+903/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_moe_gemm.hpp`  (+509/-0)
- `include/ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v7r3_scatter.hpp`  (+466/-0)
- `example/65_gemm_multiply_multiply/moe_gemm2_xdl_fp8.cpp`  (+448/-0)
- `example/65_gemm_multiply_multiply/moe_gemm1_xdl_fp8.cpp`  (+445/-0)
- `include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r3_scatter.hpp`  (+241/-0)
- `include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1_gather.hpp`  (+199/-0)
- `include/ck/library/utility/host_tensor.hpp`  (+26/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_v1.hpp`  (+14/-7)
- `example/65_gemm_multiply_multiply/CMakeLists.txt`  (+2/-0)

## Key added lines (kernel files)

**`example/65_gemm_multiply_multiply/moe_gemm1_xdl_fp8.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F8  = ck::f8_t;
```

**`example/65_gemm_multiply_multiply/moe_gemm2_xdl_fp8.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F8  = ck::f8_t;
```

**`include/ck/library/utility/host_tensor.hpp`**
```
void savetxt(std::string file_name, std::string dtype = "float")
std::ofstream file(file_name);
if(file.is_open())
for(auto& itm : mData)
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_v1.hpp`**
```
using Base::MWaves;
constexpr auto num_buffer_load_inst_b = HotLoopInstList::B_Buffer_Load_Inst_Num * MWaves;
constexpr auto mfma_interleave        = MPerXDL == 32 ? 1 : 2;
if constexpr(MPerBlock >= 128 && NPerBlock >= 128)
```

**`include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1_gather.hpp`**
```
namespace ck {
template <typename ThreadGroup,
typename SrcElementwiseOperation,
typename DstElementwiseOperation,
```
