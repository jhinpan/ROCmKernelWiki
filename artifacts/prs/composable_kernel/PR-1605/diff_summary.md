# Diff summary

- **files changed:** 90
- **lines:** +4667 / -121
- **kernel-ish files:** 81

## Files (by churn)

- `example/ck_tile/11_add_rmsnorm2d_rdquant/example_add_rmsnorm2d_rdquant_fwd.cpp`  (+280/-0)
- `example/ck_tile/11_add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd.cpp`  (+279/-0)
- `include/ck_tile/ops/add_rmsnorm2d_rdquant/pipeline/add_rmsnorm2d_rdquant_fwd_pipeline_three_pass.hpp`  (+266/-0)
- `include/ck_tile/ops/reduce/block/block_reduce2d.hpp`  (+260/-0)
- `include/ck_tile/ops/add_rmsnorm2d_rdquant/kernel/add_rmsnorm2d_rdquant_fwd_kernel.hpp`  (+239/-0)
- `include/ck_tile/ops/rmsnorm2d/kernel/rmsnorm2d_fwd_kernel.hpp`  (+202/-0)
- `example/ck_tile/10_rmsnorm2d/rmsnorm2d_fwd.cpp`  (+179/-0)
- `example/ck_tile/05_reduce/reduce.hpp`  (+109/-63)
- `example/ck_tile/10_rmsnorm2d/example_rmsnorm2d_fwd.cpp`  (+165/-0)
- `example/ck_tile/11_add_rmsnorm2d_rdquant/instances/add_rmsnorm2d_rdquant_fwd_api.cpp`  (+157/-0)
- `example/ck_tile/10_rmsnorm2d/instances/rmsnorm2d_fwd_api.cpp`  (+153/-0)
- `include/ck_tile/ops/add_rmsnorm2d_rdquant/pipeline/add_rmsnorm2d_rdquant_fwd_pipeline_one_pass.hpp`  (+142/-0)
- `include/ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_pipeline_two_pass.hpp`  (+131/-0)
- `example/ck_tile/11_add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd.hpp`  (+123/-0)
- `example/ck_tile/10_rmsnorm2d/rmsnorm2d_fwd.hpp`  (+117/-0)

## Key added lines (kernel files)

**`example/ck_tile/05_reduce/reduce.cpp`**
```
using XDataType       = DataType;
using ComputeDataType = float;
using YDataType       = DataType;
ck_tile::HostTensor<XDataType> x_host({m, n});
```

**`example/ck_tile/05_reduce/reduce.hpp`**
```
template <typename BlockWarps, // num warps along seq<M, N>
typename Vector>     // contiguous pixels(vector size) along seq<M, N>
struct Reduce2dShape
static constexpr index_t Vector_M = Vector::at(number<0>{});
```

**`example/ck_tile/10_rmsnorm2d/example_rmsnorm2d_fwd.cpp`**
```
auto create_args(int argc, char* argv[])
ck_tile::ArgParser arg_parser;
arg_parser.insert("m", "3328", "m dimension")
.insert("n", "4096", "n dimension")
```

**`example/ck_tile/10_rmsnorm2d/instances/rmsnorm2d_fwd_api.cpp`**
```
template <typename DataType_,
ck_tile::index_t Repeat_M_,         // each thread repeat along M
ck_tile::index_t Repeat_N_,         // each thread repeat along N
ck_tile::index_t ThreadPerBlock_M_, // num threads along M
```

**`example/ck_tile/10_rmsnorm2d/instances/rmsnorm2d_fwd_bf16_n1024_instance.cpp`**
```
template float rmsnorm2d_fwd_<trait_<ck_tile::bf16_t, 1,  2,  4,  64, 8,  true , false, false>>(const S&, A);
template float rmsnorm2d_fwd_<trait_<ck_tile::bf16_t, 1,  4,  4,  64, 4,  true , false, false>>(const S&, A);
template float rmsnorm2d_fwd_<trait_<ck_tile::bf16_t, 1,  8,  4,  64, 2,  true , false, false>>(const S&, A);
template float rmsnorm2d_fwd_<trait_<ck_tile::bf16_t, 1, 16,  4,  64, 1,  true , false, false>>(const S&, A);
```
