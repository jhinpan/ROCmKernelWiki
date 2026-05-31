# Diff summary

- **files changed:** 12
- **lines:** +152 / -49
- **kernel-ish files:** 12

## Files (by churn)

- `composable_kernel/include/utility/statically_indexed_array.hpp`  (+44/-0)
- `composable_kernel/include/utility/dynamic_buffer.hpp`  (+40/-0)
- `composable_kernel/include/utility/amd_buffer_addressing.hpp`  (+17/-17)
- `composable_kernel/include/utility/amd_inline_asm.hpp`  (+14/-14)
- `composable_kernel/include/utility/config.hpp`  (+13/-1)
- `composable_kernel/include/utility/type.hpp`  (+9/-1)
- `example/1_gemm_xdl/gemm_xdl.cpp`  (+4/-5)
- `composable_kernel/include/utility/amd_xdlops.hpp`  (+4/-4)
- `composable_kernel/include/utility/data_type.hpp`  (+3/-3)
- `composable_kernel/include/utility/inner_product.hpp`  (+2/-2)
- `composable_kernel/include/utility/magic_division.hpp`  (+1/-1)
- `host/driver_offline/src/conv_fwd_driver_offline.cpp`  (+1/-1)

## Key added lines (kernel files)

**`composable_kernel/include/utility/amd_buffer_addressing.hpp`**
```
return bit_cast<double>(tmp);
return bit_cast<double2_t>(tmp);
tmp.AsType<double2_t>()(Number<0>{}) = bit_cast<double2_t>(f32_0);
tmp.AsType<double2_t>()(Number<1>{}) = bit_cast<double2_t>(f32_1);
```

**`composable_kernel/include/utility/amd_inline_asm.hpp`**
```
: "v"(bit_cast<int32_t>(a)),
"v"(bit_cast<int32_t>(b0)),
"v"(bit_cast<int32_t>(b1)),
c0 = __builtin_amdgcn_sdot4(bit_cast<int32_t>(a), bit_cast<int32_t>(b0), c0, false);
```

**`composable_kernel/include/utility/amd_xdlops.hpp`**
```
llvm_intrin_amdgcn_mfma_i32_32x32x8i8(bit_cast<int>(reg_a),
bit_cast<int>(reg_b),
llvm_intrin_amdgcn_mfma_i32_16x16x16i8(bit_cast<int>(reg_a),
bit_cast<int>(reg_b),
```

**`composable_kernel/include/utility/data_type.hpp`**
```
__host__ __device__ static constexpr half_t Min() { return bit_cast<half_t>(binary_min); }
__host__ __device__ static constexpr half_t Max() { return bit_cast<half_t>(binary_max); }
__host__ __device__ static constexpr half_t Lowest() { return bit_cast<half_t>(binary_lowest); }
```

**`composable_kernel/include/utility/dynamic_buffer.hpp`**
```
__builtin_memcpy(&tmp, &(p_data_[i]), sizeof(X));
return is_valid_element ? tmp : X{0};
__builtin_memcpy(&tmp, &(p_data_[i]), sizeof(X));
return is_valid_element ? tmp : X{invalid_element_value_};
```
