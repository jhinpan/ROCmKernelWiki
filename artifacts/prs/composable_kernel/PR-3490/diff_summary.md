# Diff summary

- **files changed:** 17
- **lines:** +487 / -110
- **kernel-ish files:** 15

## Files (by churn)

- `example/ck_tile/01_fmha/fmha_fwd_runner.hpp`  (+83/-9)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_trload.hpp`  (+55/-7)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp`  (+55/-6)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+38/-16)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp`  (+33/-14)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_nwarp_sshuffle_qr_ks_vs.hpp`  (+34/-7)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_async.hpp`  (+33/-6)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp`  (+32/-7)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_pagedkv_pipeline_qr_ks_vs.hpp`  (+32/-6)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_pagedkv_kernel.hpp`  (+23/-9)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+21/-8)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp`  (+15/-6)
- `include/ck_tile/ops/fmha/kernel/fmha_batch_prefill_kernel.hpp`  (+15/-5)
- `example/ck_tile/01_fmha/script/smoke_test_fwd_sink.sh`  (+7/-0)
- `test/ck_tile/fmha/test_fmha_fwd.cpp`  (+5/-2)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/example_fmha_fwd.cpp`**
```
"Comma-separated list of length 'b'. If empty, no override.")
.insert("init_sink", "0", "value to init the output tensor sink value for validation");
int init_sink_value              = arg_parser.get_int("init_sink");
init_sink_value,
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
const void* sink_ptr;
const void* sink_ptr;
const void* sink_ptr;
const void* sink_ptr;
```

**`example/ck_tile/01_fmha/fmha_fwd_runner.hpp`**
```
template <typename SMPLComputeDataType>
void copy_attention_scores_with_sink(const ck_tile::HostTensor<SMPLComputeDataType>& s_host_ref,
const ck_tile::HostTensor<SMPLComputeDataType>& sink_host,
ck_tile::HostTensor<SMPLComputeDataType>& s_with_sinks_ref,
```

**`include/ck_tile/ops/fmha/kernel/fmha_batch_prefill_kernel.hpp`**
```
const void* sink_ptr;
drop_seed_offset,
const void* sink_ptr = nullptr)
sink_ptr,
```

**`include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`**
```
const void* sink_ptr;
const void* cu_seqlen_k_ptr = nullptr,
const void* sink_ptr        = nullptr)
sink_ptr,
```
