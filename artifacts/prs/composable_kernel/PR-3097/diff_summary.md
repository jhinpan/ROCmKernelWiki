# Diff summary

- **files changed:** 11
- **lines:** +1294 / -216
- **kernel-ish files:** 10

## Files (by churn)

- `test/ck_tile/fmha/test_fmha_bwd.cpp`  (+740/-0)
- `example/ck_tile/01_fmha/fmha_fwd_runner.hpp`  (+123/-79)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+104/-75)
- `include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp`  (+109/-15)
- `example/ck_tile/01_fmha/fmha_bwd_runner.hpp`  (+87/-22)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+48/-16)
- `example/ck_tile/01_fmha/fmha_bwd.hpp`  (+54/-3)
- `example/ck_tile/01_fmha/utils.hpp`  (+13/-3)
- `example/ck_tile/01_fmha/example_fmha_bwd.cpp`  (+12/-0)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+3/-3)
- `test/ck_tile/fmha/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
return f"(a.cu_seqlen_k_ptr != nullptr) || (a.seqlen_k == 0 || a.seqlen_k % {self.bn0} != 0)"
return f"(a.cu_seqlen_k_ptr == nullptr) && (a.seqlen_k != 0 && a.seqlen_k % {self.bn0} == 0)"
return f"(a.cu_seqlen_k_ptr == nullptr) && (a.seqlen_k != 0 && a.seqlen_k % {self.bn0} == 0)"
```

**`example/ck_tile/01_fmha/example_fmha_bwd.cpp`**
```
.insert("s_qpad",
"padded seqlen_q per batch (group mode only). "
"Use \"-s_qpad=p0,p1,...\"; -1 disables explicit padding")
.insert("s_kpad",
```

**`example/ck_tile/01_fmha/fmha_bwd.hpp`**
```
const void* seqstart_q_ptr =
nullptr; // Cumulative physical sequence length array [batch + 1]. (Used in Group mode)
const void* seqstart_k_ptr =
nullptr; // Cumulative physical sequence length array [batch + 1]. (Used in Group mode)
```

**`example/ck_tile/01_fmha/fmha_bwd_runner.hpp`**
```
std::vector<ck_tile::index_t> seqlen_qpads,
std::vector<ck_tile::index_t> seqlen_kpads,
std::tie(seqlen_qs, seqlen_ks, seqlen_qpads, seqlen_kpads) = generate_missing_seqlens(
mode, batch, seqlen_qs, seqlen_ks, seqlen_qpads, seqlen_kpads, 0, false, random_engine);
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
const void* seqstart_q_ptr =
nullptr; // Cumulative physical sequence length array [batch + 1]. (Used in Group mode)
const void* seqstart_k_ptr =
nullptr; // Cumulative physical sequence length array [batch + 1]. (Used in Group mode)
```
