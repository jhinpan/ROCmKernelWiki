# Diff summary

- **files changed:** 14
- **lines:** +308 / -149
- **kernel-ish files:** 8

## Files (by churn)

- `example/ck_tile/35_batched_transpose/batched_transpose_api.cpp`  (+59/-28)
- `Jenkinsfile`  (+72/-1)
- `include/ck_tile/ops/batched_transpose/kernel/batched_transpose_kernel.hpp`  (+30/-35)
- `example/ck_tile/35_batched_transpose/batched_transpose_example.cpp`  (+23/-20)
- `include/ck_tile/ops/batched_transpose/pipeline/batched_transpose_policy.hpp`  (+23/-20)
- `example/ck_tile/35_batched_transpose/script/run_full_test.sh`  (+38/-0)
- `include/ck_tile/core/tensor/transpose_tile.hpp`  (+17/-6)
- `example/ck_tile/35_batched_transpose/script/smoke_test.sh`  (+18/-2)
- `include/ck_tile/ops/batched_transpose/pipeline/batched_transpose_pipeline.hpp`  (+6/-12)
- `include/ck_tile/core/tensor/tensor_view.hpp`  (+1/-16)
- `include/ck_tile/ops/batched_transpose/pipeline/batched_transpose_problem.hpp`  (+7/-8)
- `example/ck_tile/35_batched_transpose/script/perf_test.sh`  (+11/-0)
- `CHANGELOG.md`  (+1/-1)
- `example/ck_tile/35_batched_transpose/README.md`  (+2/-0)

## Key added lines (kernel files)

**`example/ck_tile/35_batched_transpose/batched_transpose_api.cpp`**
```
ck_tile::index_t thread_y,
bool kPadM,
bool kPadN>
uint32_t dim_stride = a.height * a.width;
```

**`example/ck_tile/35_batched_transpose/batched_transpose_example.cpp`**
```
std::cout << "Batch " << i << ":" << std::endl;
std::cout << "  Channel " << j << ":" << std::endl;
std::cout << "    Row " << k << ": ";
std::cout << static_cast<int>(x(std::vector<std::size_t>{i, j, k, v}))
```

**`include/ck_tile/core/tensor/transpose_tile.hpp`**
```
[&](auto i) {
if constexpr(vec_length_in == 1)
return 1;
return (i == y_dim_vec_in || i == y_dim_vec_out) ? y_lengths[i] : 1;
```

**`include/ck_tile/ops/batched_transpose/kernel/batched_transpose_kernel.hpp`**
```
CK_TILE_DEVICE static index_t counter = 0;
using Pipeline                        = remove_cvref_t<Pipeline_>;
using Problem                         = remove_cvref_t<typename Pipeline::Problem>;
CK_TILE_HOST static constexpr auto GridSize(const Hargs& host_args)
```

**`include/ck_tile/ops/batched_transpose/pipeline/batched_transpose_pipeline.hpp`**
```
auto input_tile = load_tile(inp_win);
auto output_tile = make_static_distributed_tensor<InputType>(
transpose_tile2d(output_tile, input_tile);
auto out_win =
```
