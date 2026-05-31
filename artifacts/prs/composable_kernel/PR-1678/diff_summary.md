# Diff summary

- **files changed:** 6
- **lines:** +94 / -42
- **kernel-ish files:** 6

## Files (by churn)

- `example/ck_tile/01_fmha/fmha_fwd.cpp`  (+38/-21)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp`  (+35/-14)
- `include/ck_tile/ops/fused_moe.hpp`  (+11/-0)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+8/-2)
- `example/ck_tile/01_fmha/utils.hpp`  (+2/-2)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`  (+0/-3)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/fmha_fwd.cpp`**
```
"seqlen_k stride between 2 batches, currently used in group-mode only\n"
std::cerr << "fmha_fwd_appendkv() is not enabled. ignoring the 's_knew' option"
<< std::endl;
const bool need_append_kvcache = (0 < seqlen_knew || 0 < rotary_dim);
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
const void* cache_batch_idx; // only used if block_table_ptr is nullptr -> batch mode (kvcache)
args.block_table_ptr,
args.batch_stride_block_table,
args.page_block_size,
```

**`example/ck_tile/01_fmha/utils.hpp`**
```
bool need_append_kvcache      = false,
if(1 < batch && need_append_kvcache)
```

**`include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp`**
```
ck_tile::index_t batch_stride_k; // when using paged-kvcache, this will be stride/size for
ck_tile::index_t batch_stride_v; // when using paged-kvcache, this will be stride/size for
std::conditional_t<kDoFp8StaticQuant, Fp8StaticQuantKargs, EmptyKargs<2>>,
std::conditional_t<kIsPagedKV, PageBlockTableKargs, EmptyKargs<3>>
```
