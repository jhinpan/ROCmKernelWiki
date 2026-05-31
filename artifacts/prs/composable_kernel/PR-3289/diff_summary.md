# Diff summary

- **files changed:** 13
- **lines:** +389 / -272
- **kernel-ish files:** 13

## Files (by churn)

- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp`  (+182/-136)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_weight_kernel.hpp`  (+98/-60)
- `experimental/builder/include/ck_tile/builder/reflect/instance_traits_tile_grouped_convolution_backward_data.hpp`  (+22/-19)
- `experimental/builder/include/ck_tile/builder/reflect/instance_traits_tile_grouped_convolution_backward_weight.hpp`  (+22/-19)
- `experimental/builder/include/ck_tile/builder/reflect/instance_traits_tile_grouped_convolution_forward.hpp`  (+22/-19)
- `include/ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp`  (+27/-3)
- `include/ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp`  (+4/-2)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward_large_tensor_invoker.hpp`  (+0/-5)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_data_kernel.hpp`  (+3/-2)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward_invoker.hpp`  (+0/-4)
- `experimental/builder/test/test_bwd_data_instance_traits.cpp`  (+3/-1)
- `experimental/builder/test/test_bwd_weight_instance_traits.cpp`  (+3/-1)
- `experimental/builder/test/test_fwd_instance_traits.cpp`  (+3/-1)

## Key added lines (kernel files)

**`experimental/builder/include/ck_tile/builder/reflect/instance_traits_tile_grouped_convolution_backward_data.hpp`**
```
static constexpr int kExplicitGemm = GroupedConvTraitsType_::ExplicitGemm;
oss << "," << kExplicitGemm;                                       // 12. ExplicitGemm
oss << "," << kMPerBlock;                                          // 13. MPerBlock
oss << "," << kNPerBlock;                                          // 14. NPerBlock
```

**`experimental/builder/include/ck_tile/builder/reflect/instance_traits_tile_grouped_convolution_backward_weight.hpp`**
```
static constexpr int kExplicitGemm = GroupedConvTraitsType_::ExplicitGemm;
oss << "," << kExplicitGemm;                                       // 12. ExplicitGemm
oss << "," << kMPerBlock;                                          // 13. MPerBlock
oss << "," << kNPerBlock;                                          // 14. NPerBlock
```

**`experimental/builder/include/ck_tile/builder/reflect/instance_traits_tile_grouped_convolution_forward.hpp`**
```
static constexpr int kExplicitGemm = GroupedConvTraitsType_::ExplicitGemm;
oss << "," << kExplicitGemm;                                       // 12. ExplicitGemm
oss << "," << kMPerBlock;                                          // 13. MPerBlock
oss << "," << kNPerBlock;                                          // 14. NPerBlock
```

**`experimental/builder/test/test_bwd_data_instance_traits.cpp`**
```
false /*EnableSplitImage*/,
false /*ExplicitGemm*/>;
",0"           // ExplicitGemm
```

**`experimental/builder/test/test_bwd_weight_instance_traits.cpp`**
```
false /*EnableSplitImage*/,
false /*ExplicitGemm*/>;
",0"           // ExplicitGemm
```
