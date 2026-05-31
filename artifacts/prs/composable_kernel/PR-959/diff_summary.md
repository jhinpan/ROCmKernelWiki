# Diff summary

- **files changed:** 7
- **lines:** +25 / -36
- **kernel-ish files:** 7

## Files (by churn)

- `client_example/22_grouped_gemm/grouped_gemm_fixed_nk_fp16.cpp`  (+4/-6)
- `client_example/21_grouped_gemm_bias/grouped_gemm_fixed_nk_bias_fp16.cpp`  (+4/-5)
- `client_example/22_grouped_gemm/grouped_gemm_fixed_nk_fp8.cpp`  (+4/-5)
- `client_example/22_grouped_gemm/grouped_gemm_fixed_nk_i8.cpp`  (+4/-5)
- `example/15_grouped_gemm/grouped_gemm_xdl_fixed_nk_fp16.cpp`  (+3/-5)
- `example/15_grouped_gemm/grouped_gemm_xdl_fixed_nk_fp8.cpp`  (+3/-5)
- `example/15_grouped_gemm/grouped_gemm_xdl_splitk_fp16.cpp`  (+3/-5)

## Key added lines (kernel files)

**`client_example/21_grouped_gemm_bias/grouped_gemm_fixed_nk_bias_fp16.cpp`**
```
const int group_count = 16;
Ms.push_back(256 + 256 * i);
Ns.push_back(128 + 128 * i);
Ks.push_back(128 + 64 * i);
```

**`client_example/22_grouped_gemm/grouped_gemm_fixed_nk_fp16.cpp`**
```
const int group_count = 16;
Ms.push_back(256 + 256 * i);
Ns.push_back(128 + 128 * i);
Ks.push_back(128 + 64 * i);
```

**`client_example/22_grouped_gemm/grouped_gemm_fixed_nk_fp8.cpp`**
```
const int group_count = 16;
Ms.push_back(256 + 256 * i);
Ns.push_back(128 + 128 * i);
Ks.push_back(128 + 64 * i);
```

**`client_example/22_grouped_gemm/grouped_gemm_fixed_nk_i8.cpp`**
```
const int group_count = 16;
Ms.push_back(256 + 256 * i);
Ns.push_back(128 + 128 * i);
Ks.push_back(128 + 64 * i);
```

**`example/15_grouped_gemm/grouped_gemm_xdl_fixed_nk_fp16.cpp`**
```
problem_size.Ms.push_back(256 + 256 * i);
problem_size.Ns.push_back(128 + 128 * i);
problem_size.Ks.push_back(128 + 64 * i);
```
