# Diff summary

- **files changed:** 15
- **lines:** +3520 / -12
- **kernel-ish files:** 14

## Files (by churn)

- `include/ck_tile/ops/fmha/kernel/fmha_fwd_pagedkv_kernel.hpp`  (+1374/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_pagedkv_pipeline_qr_ks_vs.hpp`  (+751/-0)
- `example/ck_tile/01_fmha/codegen/ops/fmha_pagedkv_prefill.py`  (+585/-0)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+260/-0)
- `include/ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2r1.hpp`  (+247/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_pagedkv_pipeline_qr_ks_vs_default_policy.hpp`  (+91/-0)
- `include/ck_tile/ops/fmha/block/page_block_navigator.hpp`  (+71/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp`  (+52/-0)
- `example/ck_tile/01_fmha/fmha_fwd.cpp`  (+36/-7)
- `include/ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp`  (+28/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp`  (+12/-4)
- `example/ck_tile/01_fmha/CMakeLists.txt`  (+8/-1)
- `include/ck_tile/ops/fmha.hpp`  (+3/-0)
- `example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`  (+1/-0)
- `include/ck_tile/ops/gemm.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`**
```
"qr_pagedkv" : "ck_tile::BlockFmhaPipelineEnum::QRKSVS",
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_pagedkv_prefill.py`**
```
import copy
from dataclasses import dataclass
import fnmatch
import itertools
```

**`example/ck_tile/01_fmha/fmha_fwd.cpp`**
```
CK_TILE_FMHA_FWD_PAGEDKV_API))
else if constexpr(std::is_same_v<fmha_fwd_pagedkv_traits,
std::decay_t<decltype(traits)>>)
traits.use_pagedkv = use_kvcache;
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
struct fmha_fwd_pagedkv_args
const void* q_ptr;
const void* k_ptr;
const void* v_ptr;
```

**`include/ck_tile/ops/fmha/block/page_block_navigator.hpp`**
```
template <typename TileWindow>
CK_TILE_HOST_DEVICE index_t
move_tile_window(index_t /*block_index*/,
TileWindow& tile_window,
```
