# Diff summary

- **files changed:** 5 (diff was byte-capped; summary is partial)
- **lines:** +2584 / -7
- **kernel-ish files:** 3

## Files (by churn)

- `example/ck_tile/01_fmha/script/res_sink.log`  (+2462/-0)
- `example/ck_tile/01_fmha/script/correct_test_fwd_sink.sh`  (+74/-0)
- `example/ck_tile/01_fmha/mask.hpp`  (+36/-6)
- `example/ck_tile/01_fmha/fmha_fwd.cpp`  (+6/-1)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+6/-0)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/fmha_fwd.cpp`**
```
"'t:l,r,s', top-left attention sinks with FA style left right and sink size\n"
"'b:l,r,s', bottom-r attention sinks with FA style left right and sink size\n"
args.sink_size         = mask.sink;
mask.left, mask.right, mask.sink, real_seqlen_q, real_seqlen_k));
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
ck_tile::index_t sink_size;
ck_tile::index_t sink_size;
args.sink_size,
args.sink_size,
```

**`example/ck_tile/01_fmha/mask.hpp`**
```
ck_tile::index_t sink;
ck_tile::index_t sink_size   = 0;
left_size, right_size, sink_size, y_total, x_total, t == "xt");
tmp.type              = mask_enum::window_generic;
```
