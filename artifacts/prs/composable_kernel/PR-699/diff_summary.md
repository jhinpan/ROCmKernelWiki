# Diff summary

- **files changed:** 28
- **lines:** +4234 / -36
- **kernel-ish files:** 24

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_streamk.hpp`  (+1183/-0)
- `test/block_swizzle_test/block_swizzle_test.cpp`  (+406/-0)
- `include/ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp`  (+404/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_streamk.hpp`  (+357/-0)
- `profiler/include/profiler/profile_gemm_streamk_impl.hpp`  (+265/-0)
- `include/ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v6r1r2.hpp`  (+213/-0)
- `example/01_gemm/run_gemm_example.inc`  (+141/-28)
- `include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1r2.hpp`  (+164/-0)
- `test/block_swizzle_test/simple_args.h`  (+159/-0)
- `profiler/src/profile_gemm_streamk.cpp`  (+155/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_streamk.hpp`  (+121/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v3.hpp`  (+89/-0)
- `example/01_gemm/common.hpp`  (+73/-2)
- `include/ck/utility/workgroup_barrier.hpp`  (+73/-0)
- `include/ck/utility/magic_division.hpp`  (+72/-0)

## Key added lines (kernel files)

**`example/01_gemm/common.hpp`**
```
struct ProblemSizeStreamK final
ck::index_t M = 3840;
ck::index_t N = 4096;
ck::index_t K = 4096;
```

**`example/01_gemm/gemm_xdl_streamk.cpp`**
```
using ADataType        = ck::half_t;
using BDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = float;
```

**`example/01_gemm/run_gemm_example.inc`**
```
template <typename ProblemType>
bool run_gemm(const ProblemType& problem_size, const ExecutionConfig& config)
auto M       = problem_size.M;
auto N       = problem_size.N;
```

**`include/ck/host_utility/kernel_launch.hpp`**
```
template <typename... Args, typename F, typename PreProcessFunc>
float launch_and_time_kernel_with_preprocess(const StreamConfig& stream_config,
PreProcessFunc preprocess,
F kernel,
```

**`include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp`**
```
__device__ void SetSrcSliceOrigin(const SrcDesc& src_desc, const Index& src_block_slice_origin)
if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
const auto thread_cluster_idx = thread_cluster_desc_.CalculateBottomIndex(
```
