# Diff summary

- **files changed:** 24 (diff was byte-capped; summary is partial)
- **lines:** +3038 / -740
- **kernel-ish files:** 22

## Files (by churn)

- `include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp`  (+555/-332)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp`  (+782/-0)
- `example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`  (+370/-179)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr_iglp.hpp`  (+529/-0)
- `include/ck_tile/ops/fmha/block/block_dropout.hpp`  (+377/-16)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_convert_dq.hpp`  (+141/-0)
- `example/ck_tile/01_fmha/fmha_bwd.hpp`  (+95/-11)
- `example/ck_tile/01_fmha/fmha_bwd.cpp`  (+49/-18)
- `include/ck_tile/ops/fmha/kernel/fmha_bwd_tile_partitioner.hpp`  (+0/-54)
- `include/ck_tile/core/algorithm/coordinate_transform.hpp`  (+9/-33)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_combine_kernel.hpp`  (+21/-18)
- `include/ck_tile/core/utility/philox_rand.hpp`  (+33/-0)
- `example/ck_tile/01_fmha/fmha_fwd.cpp`  (+14/-11)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp`  (+12/-13)
- `example/ck_tile/01_fmha/script/smoke_test_bwd.sh`  (+12/-10)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`**
```
DROPOUT_MAP = {
"no"                        : "ck_tile::BlockDropoutBwd<false, true,  false>",
"dropout_wg32"              : "ck_tile::BlockDropoutBwd<true,  true,  false>",
"dropout_wg32_storerandval" : "ck_tile::BlockDropoutBwd<true,  true,  true >",
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`**
```
"kr_ktr_vr_iglp" : "ck_tile::BlockFmhaBwdDQDKDVPipelineKRKTRVRIGLP",
"kr_ktr_vr"      : "ck_tile::BlockFmhaBwdDQDKDVPipelineKRKTRVR",
"kr_ktr_vr_iglp" : "ck_tile::BlockFmhaBwdPipelineEnum::KRKTRVR_IGLP",
"kr_ktr_vr"      : "ck_tile::BlockFmhaBwdPipelineEnum::KRKTRVR",
```

**`example/ck_tile/01_fmha/fmha_bwd.cpp`**
```
.insert("repeat", "20", "number of iterations to benchmark the kernel")
.insert("deterministic",
"if set to 1 will use multi-buffer reduction strategy for dq, atomic opeartion "
"will not be used");
```

**`example/ck_tile/01_fmha/fmha_bwd.hpp`**
```
void* dq_acc_ptr;
ck_tile::index_t stride_dq_acc;
ck_tile::index_t stride_dq;
ck_tile::index_t nhead_stride_dq_acc;
```

**`example/ck_tile/01_fmha/fmha_fwd.cpp`**
```
1 < num_splits
? std::array<ck_tile::index_t, 4>{num_splits, shape_batch, nhead, shape_seqlen_q}
: std::array<ck_tile::index_t, 4>{1, 1, 1, 1});
lse ? std::array<ck_tile::index_t, 3>{shape_batch, nhead, shape_seqlen_q}
```
