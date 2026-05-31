# Diff summary

- **files changed:** 25
- **lines:** +433 / -870
- **kernel-ish files:** 20

## Files (by churn)

- `example/ck_tile/37_transpose/transpose_example.cpp`  (+0/-257)
- `example/ck_tile/35_batched_transpose/batched_transpose_api.cpp`  (+156/-57)
- `example/ck_tile/37_transpose/block_transpose.hpp`  (+0/-149)
- `example/ck_tile/37_transpose/batched_transpose_kernel.hpp`  (+0/-120)
- `include/ck_tile/ops/batched_transpose/pipeline/batched_transpose_lds_problem.hpp`  (+73/-0)
- `include/ck_tile/ops/batched_transpose/pipeline/batched_transpose_lds_pipeline.hpp`  (+67/-0)
- `example/ck_tile/37_transpose/transpose_api.cpp`  (+0/-59)
- `include/ck_tile/ops/batched_transpose/pipeline/batched_transpose_lds_policy.hpp`  (+14/-44)
- `test/ck_tile/batched_transpose/batched_transpose_api.cpp`  (+20/-24)
- `example/ck_tile/35_batched_transpose/script/smoke_test.sh`  (+22/-20)
- `include/ck_tile/ops/batched_transpose/pipeline/batched_transpose_policy.hpp`  (+8/-26)
- `include/ck_tile/ops/batched_transpose/pipeline/batched_transpose_common_policy.hpp`  (+33/-0)
- `include/ck_tile/ops/batched_transpose/pipeline/batched_transpose_problem.hpp`  (+10/-21)
- `example/ck_tile/37_transpose/README.md`  (+0/-27)
- `example/ck_tile/37_transpose/transpose_example.hpp`  (+0/-27)

## Key added lines (kernel files)

**`example/ck_tile/04_img2col/image_to_column.cpp`**
```
if(config.time_kernel)
std::cout << "image_to_column: pass, No Perf generated due to config.time_kernel=0"
<< std::endl;
```

**`example/ck_tile/35_batched_transpose/batched_transpose_api.cpp`**
```
namespace {
template <int32_t pipeline_id>
struct kernel_traits;
template <>
```

**`example/ck_tile/35_batched_transpose/batched_transpose_example.cpp`**
```
.insert("kname", "0", "t to 1 will print kernel name")
.insert("pipeline", "0", "0: no LDS usage, 1: LDS-accelerated (gfx950)");
std::string pipeline   = args.get_str("pipeline");
auto trait = batched_transpose_trait{prec, layout_in, pipeline};
```

**`example/ck_tile/35_batched_transpose/batched_transpose_example.hpp`**
```
std::string pipeline;
```

**`include/ck_tile/ops/batched_transpose/kernel/batched_transpose_kernel.hpp`**
```
using Type = typename Problem::DataType;
CK_TILE_HOST static constexpr auto BlockSize() { return Problem::kBlockSize; }
```
