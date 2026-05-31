# Diff summary

- **files changed:** 22
- **lines:** +1284 / -207
- **kernel-ish files:** 17

## Files (by churn)

- `example/15_grouped_gemm/run_grouped_gemm_example.inc`  (+233/-0)
- `example/35_splitK_gemm/run_splitK_gemm_example.inc`  (+196/-0)
- `example/15_grouped_gemm/grouped_gemm_xdl_fp16.cpp`  (+2/-193)
- `example/24_batched_gemm/run_batched_gemm_example.inc`  (+194/-0)
- `example/15_grouped_gemm/grouped_gemm_xdl_bfp16.cpp`  (+61/-0)
- `example/15_grouped_gemm/grouped_gemm_xdl_fp32.cpp`  (+61/-0)
- `example/24_batched_gemm/batched_gemm_xdl_bfp16.cpp`  (+59/-0)
- `example/24_batched_gemm/batched_gemm_xdl_fp16.cpp`  (+59/-0)
- `example/15_grouped_gemm/grouped_gemm_xdl_int8.cpp`  (+58/-0)
- `example/24_batched_gemm/batched_gemm_xdl_fp32.cpp`  (+58/-0)
- `example/35_splitK_gemm/splitK_gemm_xdl_bfp16.cpp`  (+58/-0)
- `example/35_splitK_gemm/splitK_gemm_xdl_fp16.cpp`  (+58/-0)
- `example/35_splitK_gemm/splitK_gemm_xdl_fp32.cpp`  (+58/-0)
- `example/24_batched_gemm/batched_gemm_xdl_int8.cpp`  (+56/-0)
- `example/35_splitK_gemm/splitK_gemm_xdl_int8.cpp`  (+55/-0)

## Key added lines (kernel files)

**`example/15_grouped_gemm/grouped_gemm_xdl_bfp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using BF16 = ck::bhalf_t;
using F32  = float;
```

**`example/15_grouped_gemm/grouped_gemm_xdl_fp16.cpp`**
```
int main(int argc, char* argv[]) { return !run_grouped_gemm_example(argc, argv); }
```

**`example/15_grouped_gemm/grouped_gemm_xdl_fp32.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`example/15_grouped_gemm/grouped_gemm_xdl_int8.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
```

**`example/15_grouped_gemm/run_grouped_gemm_example.inc`**
```
struct ProblemSize final
std::vector<ck::index_t> Ms;
std::vector<ck::index_t> Ns;
std::vector<ck::index_t> Ks;
```
