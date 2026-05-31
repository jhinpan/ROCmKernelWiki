# Diff summary

- **files changed:** 14
- **lines:** +118 / -10
- **kernel-ish files:** 14

## Files (by churn)

- `include/ck/library/utility/validation_common.hpp`  (+50/-0)
- `example/01_gemm/run_gemm_example.inc`  (+13/-1)
- `profiler/include/profiler/profile_gemm_ab_scale_impl.hpp`  (+6/-1)
- `profiler/include/profiler/profile_gemm_bias_add_reduce_impl.hpp`  (+5/-1)
- `profiler/include/profiler/profile_gemm_impl.hpp`  (+5/-1)
- `profiler/include/profiler/profile_gemm_reduce_impl.hpp`  (+5/-1)
- `profiler/include/profiler/profile_gemm_splitk_impl.hpp`  (+5/-1)
- `profiler/include/profiler/profile_gemm_streamk_impl.hpp`  (+5/-1)
- `profiler/include/profiler/profile_gemm_universal_reduce_impl.hpp`  (+5/-1)
- `profiler/include/profiler/profile_gemm_universal_streamk_impl.hpp`  (+5/-1)
- `profiler/include/profiler/profile_gemm_blockscale_wp_impl.hpp`  (+5/-0)
- `profiler/include/profiler/profile_gemm_universal_impl.hpp`  (+4/-0)
- `profiler/include/profiler/profile_gemm_universal_preshuffle_impl.hpp`  (+4/-0)
- `example/01_gemm/run_gemm_example_v2.inc`  (+1/-1)

## Key added lines (kernel files)

**`example/01_gemm/run_gemm_example.inc`**
```
ck::utils::validate_gemm_strides_abc<ALayout, BLayout, CLayout>(
M, N, K, StrideA, StrideB, StrideC);
catch(const std::runtime_error& e)
std::cerr << "Error: " << e.what() << std::endl;
```

**`include/ck/library/utility/validation_common.hpp`**
```
namespace ck {
namespace utils {
template <typename Layout>
inline void
```

**`profiler/include/profiler/profile_gemm_ab_scale_impl.hpp`**
```
ck::utils::validate_gemm_stride<ALayout>(M, K, StrideA, "StrideA");
ck::utils::validate_gemm_stride<BLayout>(K, N, StrideB, "StrideB");
ck::utils::validate_gemm_stride<BLayout>(M, N, StrideE, "StrideE");
```

**`profiler/include/profiler/profile_gemm_bias_add_reduce_impl.hpp`**
```
ck::utils::validate_gemm_strides_abc<ALayout, BLayout, CLayout>(
M, N, K, StrideA, StrideB, StrideC);
```

**`profiler/include/profiler/profile_gemm_blockscale_wp_impl.hpp`**
```
ck::utils::validate_gemm_stride<ALayout>(M, K, StrideA, "StrideA");
ck::utils::validate_gemm_stride<BLayout>(K, N, StrideB, "StrideB");
ck::utils::validate_gemm_stride<BLayout>(M, N, StrideE, "StrideE");
```
