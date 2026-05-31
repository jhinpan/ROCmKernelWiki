# Diff summary

- **files changed:** 200
- **lines:** +680 / -659
- **kernel-ish files:** 200

## Files (by churn)

- `test/s_prefetch_op/s_prefetch_op_util.hpp`  (+249/-249)
- `test/permute_scale/test_permute_scale.cpp`  (+100/-100)
- `test/s_prefetch_op/s_prefetch_op.cpp`  (+66/-66)
- `test/transpose/test_transpose_xdl.cpp`  (+36/-35)
- `test/ck_tile/smoothquant/instances/smoothquant_fp16_n3072_instance.cpp`  (+2/-2)
- `test/ck_tile/smoothquant/instances/smoothquant_fp16_n4096_instance.cpp`  (+2/-2)
- `test/ck_tile/smoothquant/instances/smoothquant_fp16_n4096_tp_instance.cpp`  (+2/-2)
- `test/ck_tile/smoothquant/instances/smoothquant_fp16_n512_instance.cpp`  (+2/-2)
- `test/ck_tile/smoothquant/instances/smoothquant_fp16_n64_n128_instance.cpp`  (+2/-2)
- `test/ck_tile/smoothquant/instances/smoothquant_fp16_n768_instance.cpp`  (+2/-2)
- `test/ck_tile/smoothquant/instances/smoothquant_fwd_api.cpp`  (+2/-2)
- `test/ck_tile/smoothquant/instances/smoothquant_instance_common.hpp`  (+2/-2)
- `test/ck_tile/smoothquant/smoothquant.hpp`  (+2/-2)
- `test/ck_tile/smoothquant/test_smoothquant.cpp`  (+2/-2)
- `test/ck_tile/smoothquant/test_smoothquant_cases.inc`  (+2/-2)

## Key added lines (kernel files)

**`test/permute_scale/test_permute_scale.cpp`**
```
using F16 = ck::half_t;
using F32 = float;
using ck::index_t;
template <typename Tuple>
```

**`test/s_prefetch_op/s_prefetch_op.cpp`**
```
template <typename T, uint32_t NUM_THREADS, uint32_t NUM_SCALARS>
bool run_test(bool time_kernels)
bool pass = true;
const auto s_prefetch_kernel =
```

**`test/s_prefetch_op/s_prefetch_op_util.hpp`**
```
namespace ck {
namespace s_prefetch_op_util {
__device__ __forceinline__ void enable_scalar_prefetch()
__builtin_amdgcn_s_setreg(1 | (24 << 6), 1); // Set bit to 1
```

**`test/transpose/test_transpose_xdl.cpp`**
```
using F16 = ck::half_t;
using F32 = float;
using ck::index_t;
template <typename Tuple>
```
