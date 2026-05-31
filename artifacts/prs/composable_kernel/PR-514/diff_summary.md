# Diff summary

- **files changed:** 82
- **lines:** +346 / -273
- **kernel-ish files:** 79

## Files (by churn)

- `profiler/src/profiler.cpp`  (+9/-141)
- `profiler/src/profiler_operation_registry.hpp`  (+79/-0)
- `profiler/CMakeLists.txt`  (+2/-61)
- `profiler/src/CMakeLists.txt`  (+63/-0)
- `profiler/src/profile_groupnorm.cpp`  (+9/-3)
- `profiler/src/profile_batched_gemm_add_relu_gemm_add.cpp`  (+8/-3)
- `profiler/src/profile_conv_fwd_bias_relu_add.cpp`  (+8/-3)
- `profiler/src/profile_batched_gemm.cpp`  (+8/-2)
- `profiler/src/profile_batched_gemm_gemm.cpp`  (+8/-2)
- `profiler/src/profile_batched_gemm_reduce.cpp`  (+8/-2)
- `profiler/src/profile_conv_bwd_data.cpp`  (+8/-2)
- `profiler/src/profile_conv_fwd.cpp`  (+8/-2)
- `profiler/src/profile_conv_fwd_bias_relu.cpp`  (+8/-2)
- `profiler/src/profile_gemm.cpp`  (+8/-2)
- `profiler/src/profile_gemm_add_add_fastgelu.cpp`  (+8/-2)

## Key added lines (kernel files)

**`profiler/src/profile_batched_gemm.cpp`**
```
printf("arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n");
REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_batched_gemm);
```

**`profiler/src/profile_batched_gemm_add_relu_gemm_add.cpp`**
```
printf("arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n");
REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_batched_gemm_add_relu_gemm_add);
```

**`profiler/src/profile_batched_gemm_gemm.cpp`**
```
printf("arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n");
REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_batched_gemm_gemm);
```

**`profiler/src/profile_batched_gemm_reduce.cpp`**
```
printf("arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n");
REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_batched_gemm_reduce);
```

**`profiler/src/profile_batchnorm_bwd.cpp`**
```
REGISTER_PROFILER_OPERATION("bnorm_bwd", "Batchnorm backward", profile_batchnorm_backward);
```
