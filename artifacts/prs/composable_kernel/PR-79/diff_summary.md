# Diff summary

- **files changed:** 16
- **lines:** +2098 / -13
- **kernel-ish files:** 14

## Files (by churn)

- `composable_kernel/include/tensor_operation/gridwise_batched_gemm_xdlops_v2r3.hpp`  (+708/-0)
- `device_operation/include/device_batched_gemm_xdl.hpp`  (+506/-0)
- `profiler/include/profile_batched_gemm_impl.hpp`  (+247/-0)
- `profiler/src/profile_batched_gemm.cpp`  (+155/-0)
- `reference_operation/include/reference_batched_gemm.hpp`  (+134/-0)
- `composable_kernel/include/tensor_operation/blockwise_gemm_xdlops.hpp`  (+62/-11)
- `device_operation/src/device_batched_gemm_xdl_f16_f16_f16_gmk_gkn_gmn_instance.cpp`  (+56/-0)
- `device_operation/src/device_batched_gemm_xdl_f16_f16_f16_gmk_gnk_gmn_instance.cpp`  (+56/-0)
- `device_operation/src/device_batched_gemm_xdl_f16_f16_f16_gkm_gkn_gmn_instance.cpp`  (+52/-0)
- `device_operation/src/device_batched_gemm_xdl_f16_f16_f16_gkm_gnk_gmn_instance.cpp`  (+52/-0)
- `composable_kernel/include/tensor_operation/xdlops_gemm.hpp`  (+37/-0)
- `device_operation/CMakeLists.txt`  (+12/-0)
- `profiler/src/profiler.cpp`  (+7/-2)
- `composable_kernel/include/utility/static_buffer.hpp`  (+7/-0)
- `composable_kernel/include/utility/static_buffer_of_vector_type_v2.hpp`  (+5/-0)

## Key added lines (kernel files)

**`composable_kernel/include/tensor_operation/blockwise_gemm_xdlops.hpp`**
```
StaticBufferTupleOfVector<AddressSpaceEnum_t::Vgpr,
FloatAcc,
MRepeat * NRepeat,
xdlops_gemm.GetRegSizePerXdlops(),
```

**`composable_kernel/include/tensor_operation/gridwise_batched_gemm_xdlops_v2r3.hpp`**
```
namespace ck {
template <typename GridwiseBatchedGemm,
typename FloatAB,
typename FloatC,
```

**`composable_kernel/include/tensor_operation/xdlops_gemm.hpp`**
```
template <typename CDesc_G_M0_N0_M1_N1_M2_N2>
__host__ __device__ static constexpr auto MakeCDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(
const CDesc_G_M0_N0_M1_N1_M2_N2& c_desc_g_m0_n0_m1_n1_m2_n2)
const auto G  = c_desc_g_m0_n0_m1_n1_m2_n2.GetLength(I0);
```

**`composable_kernel/include/utility/static_buffer.hpp`**
```
__host__ __device__ void Clear()
const index_t numScalars = NumOfVector * ScalarPerVector;
static_for<0, Number<numScalars>{}, 1>{}([&](auto i) { SetAsType(i, S{0}); });
```

**`composable_kernel/include/utility/static_buffer_of_vector_type_v2.hpp`**
```
__host__ __device__ void Fill(VecBaseType v)
static_for<0, GetNumElements(), 1>{}([&](auto i) { GetElement(i, true) = v; });
```
