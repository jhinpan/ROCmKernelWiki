# Diff summary

- **files changed:** 9
- **lines:** +119 / -33
- **kernel-ish files:** 9

## Files (by churn)

- `profiler/src/profile_grouped_gemm.cpp`  (+31/-7)
- `test/gemm_split_k/test_gemm_splitk_util.hpp`  (+16/-3)
- `test/grouped_gemm/test_grouped_gemm_util.hpp`  (+16/-3)
- `profiler/src/profile_gemm_splitk.cpp`  (+15/-2)
- `profiler/src/profile_gemm.cpp`  (+14/-2)
- `include/ck/host_utility/kernel_launch.hpp`  (+8/-5)
- `profiler/include/profiler/profile_gemm_splitk_impl.hpp`  (+7/-4)
- `profiler/include/profiler/profile_grouped_gemm_impl.hpp`  (+7/-4)
- `profiler/include/profiler/profile_gemm_impl.hpp`  (+5/-3)

## Key added lines (kernel files)

**`include/ck/host_utility/kernel_launch.hpp`**
```
printf("Warm up %d times\n", stream_config.cold_niters_);
printf("Warm up %d times\n", stream_config.cold_niters_);
for(int i = 0; i < stream_config.cold_niters_; ++i)
kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
```

**`profiler/include/profiler/profile_gemm_impl.hpp`**
```
int StrideC,
int n_warmup,
int n_iter)
float avg_time = invoker_ptr->Run(
```

**`profiler/include/profiler/profile_gemm_splitk_impl.hpp`**
```
int KBatch,
int n_warmup,
int n_iter)
invoker_ptr->Run(argument_ptr.get(),
```

**`profiler/include/profiler/profile_grouped_gemm_impl.hpp`**
```
int kbatch   = 1,
int n_warmup = 1,
int n_iter   = 10)
invoker_ptr->Run(argument_ptr.get(),
```

**`profiler/src/profile_gemm.cpp`**
```
<< "optional:\n"
<< "arg14: number of warm-up cycles (default 1)\n"
<< "arg15: number of iterations (default 10)\n"
if(argc != 14 && argc != 16)
```
