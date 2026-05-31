# Diff summary

- **files changed:** 18
- **lines:** +489 / -561
- **kernel-ish files:** 18

## Files (by churn)

- `include/ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp`  (+69/-205)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+187/-72)
- `include/ck_tile/host/reference/reference_gemm.hpp`  (+11/-151)
- `include/ck_tile/host/arg_parser.hpp`  (+44/-2)
- `test/ck_tile/batched_gemm/test_batched_gemm_util.hpp`  (+20/-22)
- `test/ck_tile/gemm/test_gemm_pipeline_util.hpp`  (+9/-31)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+32/-7)
- `example/ck_tile/16_batched_gemm/run_batched_gemm_example.inc`  (+31/-4)
- `example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`  (+21/-13)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_f16_f16_f16/device_gemm_xdl_universal_streamk_f16_f16_f16_mk_nk_mn.hpp`  (+24/-5)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`  (+13/-7)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_f16_f16_f16/device_gemm_xdl_universal_streamk_f16_f16_f16_mk_kn_mn.hpp`  (+15/-3)
- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+4/-12)
- `example/ck_tile/03_gemm/gemm_basic.hpp`  (+1/-15)
- `example/ck_tile/16_batched_gemm/batched_gemm.cpp`  (+3/-3)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
float gemm_calc(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
auto kargs = Kernel::MakeKernelArgs(args);
const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
```

**`example/ck_tile/03_gemm/gemm_basic.hpp`**
```
float gemm_calc(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s);
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
ck_tile::GemmHostArgs args;
args.a_ptr    = a_m_k_dev_buf.GetDeviceBuffer();
args.b_ptr    = b_k_n_dev_buf.GetDeviceBuffer();
args.c_ptr    = c_m_n_dev_buf.GetDeviceBuffer();
```

**`example/ck_tile/16_batched_gemm/batched_gemm.cpp`**
```
float batched_gemm(const ck_tile::BatchedGemmHostArgs& args, const ck_tile::stream_config& s)
auto kargs = Kernel::MakeKernelArgs(args);
const dim3 grids      = Kernel::GridSize(args.M, args.N, args.batch_count);
```

**`example/ck_tile/16_batched_gemm/batched_gemm.hpp`**
```
float batched_gemm(const ck_tile::BatchedGemmHostArgs& args, const ck_tile::stream_config& s);
```
