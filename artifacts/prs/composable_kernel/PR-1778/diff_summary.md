# Diff summary

- **files changed:** 13
- **lines:** +171 / -277
- **kernel-ish files:** 12

## Files (by churn)

- `include/ck_tile/ops/fmha/kernel/fmha_fwd_tile_partitioner.hpp`  (+0/-105)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+71/-7)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_tile_partitioner.hpp`  (+0/-54)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_combine_tile_partitioner.hpp`  (+0/-48)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp`  (+30/-10)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_combine_kernel.hpp`  (+30/-9)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_appendkv_kernel.hpp`  (+20/-8)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+3/-17)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+12/-2)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`  (+2/-8)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_appendkv.py`  (+2/-4)
- `example/ck_tile/01_fmha/README.md`  (+1/-2)
- `include/ck_tile/ops/fmha.hpp`  (+0/-3)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
ck_tile::FmhaFwdKernel<fmha_pipeline_{F_idx}, fmha_epilogue_{F_idx}>;
F_pipeline      = PIPELINE_MAP[self.F_pipeline.tag])
return f"fmha_fwd_d{self.F_hdim}_{self.F_dtype}_{self.F_mode}_" + \
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd_appendkv.py`**
```
using fmha_kernel_{F_idx} = ck_tile::FmhaFwdAppendKVKernel<fmha_pipeline_{F_idx}>;
f.write(str(file_path.parent / GEN_DIR / FMHA_FWD_APPENDKV_API_FILENAME) + "\n")
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`**
```
ck_tile::FmhaFwdSplitKVKernel<fmha_pipeline, fmha_epilogue>;
ck_tile::FmhaFwdSplitKVCombineKernel<fmha_pipeline, fmha_epilogue>;
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
if constexpr(FmhaKernel::kIsGroupMode)
dim3 grids = FmhaKernel::GridSize(
args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v, args.seqlen_k_ptr != nullptr);
return ck_tile::make_tuple(kargs, grids);
```

**`include/ck_tile/ops/fmha/kernel/fmha_fwd_appendkv_kernel.hpp`**
```
template <typename FmhaPipeline_>
CK_TILE_HOST static constexpr auto GridSize(ck_tile::index_t batch_size,
ck_tile::index_t nhead,
ck_tile::index_t seqlen_q,
```
