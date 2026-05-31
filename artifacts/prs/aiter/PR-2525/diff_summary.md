# Diff summary

- **files changed:** 21
- **lines:** +2328 / -1872
- **kernel-ish files:** 21

## Files (by churn)

- `csrc/include/binary_operator.cuh`  (+1924/-1535)
- `csrc/include/attention_common.cuh`  (+62/-70)
- `csrc/include/hip_reduce.h`  (+72/-37)
- `csrc/cpp_itfs/pa/pa.cuh`  (+59/-46)
- `csrc/cpp_itfs/pa/pa_common.cuh`  (+45/-38)
- `csrc/include/aiter_opus_plus.h`  (+57/-21)
- `csrc/include/hip_compat.h`  (+0/-70)
- `csrc/kernels/cache_kernels.cu`  (+29/-29)
- `csrc/include/aiter_hip_common.h`  (+34/-0)
- `csrc/kernels/quant_kernels.cu`  (+17/-3)
- `csrc/kernels/pos_encoding_kernels.cu`  (+6/-6)
- `csrc/kernels/topk_softmax_kernels.cu`  (+9/-3)
- `csrc/kernels/moe_align_block_size_kernels.cu`  (+4/-2)
- `csrc/kernels/attention.cu`  (+2/-2)
- `csrc/kernels/custom_kernels.cu`  (+2/-2)

## Key added lines (kernel files)

**`csrc/cpp_itfs/pa/pa.cuh`**
```
constexpr int NWARPS = NUM_THREADS / WARP_SIZE;
constexpr int QK_SIZE_RATIO =
sizeof(scalar_t) / sizeof(cache_t); // 1 for 16bit types, 2 for 8bit types
if(((global_qhead_idx + gqa_ratio_loop * GQA_RATIO_PER_LOOP) < total_num_heads) &&
```

**`csrc/cpp_itfs/pa/pa_common.cuh`**
```
union vec_converter
const floatx4& inpC)
if constexpr(std::is_same<T, __hip_fp8_e4m3>::value)
return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(inpA, inpB, inpC, absz, cbid, blgp);
```

**`csrc/include/aiter_hip_common.h`**
```
static uint32_t get_warp_size_func()
static const uint32_t warp_size = []() {
hipDevice_t dev;
hipDeviceProp_t dev_prop;
```

**`csrc/include/aiter_opus_plus.h`**
```
constexpr float hi = 448.0f, lo = -448.0f;
return array<fp4_t, 1>{};
return array<fp4_t, 2>{};
return array<fp4_t, 4>{};
```

**`csrc/include/attention_common.cuh`**
```
__device__ void
_paged_attention_kernel(const int* block_table_seq,
const int64_t query_loc,
int context_len,
```
