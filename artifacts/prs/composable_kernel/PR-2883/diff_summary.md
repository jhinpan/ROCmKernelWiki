# Diff summary

- **files changed:** 13
- **lines:** +62 / -1034
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+25/-264)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_v3_kernel.hpp`  (+7/-173)
- `example/ck_tile/01_fmha/example_fmha_fwd_v3.cpp`  (+11/-137)
- `test/ck_tile/fmha/test_fmha_fwd.inc`  (+0/-141)
- `example/ck_tile/01_fmha/fmha_fwd_runner.hpp`  (+12/-115)
- `example/ck_tile/01_fmha/script/smoke_test_fwd.sh`  (+0/-109)
- `example/ck_tile/01_fmha/script/benchmark_fwd.sh`  (+0/-33)
- `example/ck_tile/01_fmha/example_fmha_fwd.cpp`  (+1/-19)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+2/-15)
- `example/ck_tile/01_fmha/script/benchmark_fwd_v3.sh`  (+0/-17)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+3/-3)
- `example/ck_tile/01_fmha/fmha_fwd_v3.hpp`  (+0/-5)
- `example/ck_tile/01_fmha/fmha_fwd_v3_impl.hpp`  (+1/-3)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
if self.skpad == 't' : return f'a.seqlen_k == 0 || a.seqlen_k % {self.bn0} != 0'
else :                 return f'a.seqlen_k != 0 && a.seqlen_k % {self.bn0} == 0'
else :                return f'a.seqlen_k % {self.bn0} == 0'
```

**`example/ck_tile/01_fmha/example_fmha_fwd.cpp`**
```
.insert("jsonfile", "fmha_fwd.json", "json file name to dump results");
```

**`example/ck_tile/01_fmha/example_fmha_fwd_v3.cpp`**
```
.insert("repeat", "30", "number of iterations to benchmark the kernel");
ck_tile::fmha_fwd_v3_args args;
host::fmha_fwd<float, DataType>(q,
problem.mask,
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
args.drop_seed_offset);
args.drop_seed_offset);
```

**`example/ck_tile/01_fmha/fmha_fwd_runner.hpp`**
```
(mode == mode_enum::batch ? seqlen_qs[0] : seqstart_q_host.back());
lse ? std::array<ck_tile::index_t, 3>{shape_batch, nhead, shape_seqlen_q}
seqstart_k.ToDevice(seqlen_kpads[0] < 0 ? seqstart_k_host.data()
: seqstart_k_with_padding_host.data());
```
