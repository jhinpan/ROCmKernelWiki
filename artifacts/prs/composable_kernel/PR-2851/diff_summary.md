# Diff summary

- **files changed:** 13
- **lines:** +1034 / -62
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+264/-25)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_v3_kernel.hpp`  (+173/-7)
- `example/ck_tile/01_fmha/example_fmha_fwd_v3.cpp`  (+137/-11)
- `test/ck_tile/fmha/test_fmha_fwd.inc`  (+141/-0)
- `example/ck_tile/01_fmha/fmha_fwd_runner.hpp`  (+115/-12)
- `example/ck_tile/01_fmha/script/smoke_test_fwd.sh`  (+109/-0)
- `example/ck_tile/01_fmha/script/benchmark_fwd.sh`  (+33/-0)
- `example/ck_tile/01_fmha/example_fmha_fwd.cpp`  (+19/-1)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+15/-2)
- `example/ck_tile/01_fmha/script/benchmark_fwd_v3.sh`  (+17/-0)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+3/-3)
- `example/ck_tile/01_fmha/fmha_fwd_v3.hpp`  (+5/-0)
- `example/ck_tile/01_fmha/fmha_fwd_v3_impl.hpp`  (+3/-1)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
if self.skpad == 't' : return f'(a.cu_seqlen_kv_ptr != nullptr) || (a.seqlen_k == 0 || a.seqlen_k % {self.bn0} != 0)'
else :                 return f'(a.cu_seqlen_kv_ptr == nullptr) && (a.seqlen_k != 0 && a.seqlen_k % {self.bn0} == 0)'
else :                 return f'(a.cu_seqlen_kv_ptr == nullptr) && (a.seqlen_k != 0 && a.seqlen_k % {self.bn0} == 0)'
```

**`example/ck_tile/01_fmha/example_fmha_fwd.cpp`**
```
.insert("s_qpad",
"seqlen_q stride between 2 batches (group-mode optional).\n"
"Provide positive strides per-batch to simulate physical padding on Q.")
.insert("jsonfile", "fmha_fwd.json", "json file name to dump results")
```

**`example/ck_tile/01_fmha/example_fmha_fwd_v3.cpp`**
```
.insert("repeat", "30", "number of iterations to benchmark the kernel")
.insert("q_eff_lens",
"Batch-mode only: per-batch effective seqlen for Q (exclude PAD).\n"
"Comma-separated list of length 'b'. If empty, no override.")
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
const ck_tile::index_t* cu_seqlen_q_ptr  = nullptr; // [batch+1]
const ck_tile::index_t* cu_seqlen_kv_ptr = nullptr; // [batch+1]
const void* seqstart_padded_q_ptr = nullptr; // [batch+1]
const void* seqstart_padded_k_ptr = nullptr; // [batch+1]
```

**`example/ck_tile/01_fmha/fmha_fwd_runner.hpp`**
```
std::vector<ck_tile::index_t> seqlen_qpads,
std::vector<ck_tile::index_t> q_eff_lens_per_batch,
std::vector<ck_tile::index_t> kv_eff_lens_per_batch,
std::vector<int32_t> seqstart_q_with_padding_host;
```
