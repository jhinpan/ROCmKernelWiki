# Diff summary

- **files changed:** 12
- **lines:** +395 / -211
- **kernel-ish files:** 6

## Files (by churn)

- `tile_engine/ops/gemm_preshuffle/gemm_preshuffle_instance_builder.py`  (+121/-77)
- `tile_engine/ops/gemm_preshuffle/commons/validation_utils.py`  (+141/-7)
- `tile_engine/ops/gemm_preshuffle/configs/default_config.json`  (+46/-35)
- `tile_engine/ops/gemm_preshuffle/gemm_preshuffle_common.hpp`  (+24/-52)
- `tile_engine/ops/gemm_preshuffle/gemm_preshuffle_profiler.hpp`  (+19/-8)
- `tile_engine/ops/gemm_preshuffle/configs/user_provided_config.json`  (+12/-10)
- `tile_engine/ops/gemm_preshuffle/gemm_preshuffle_benchmark_single.cpp`  (+11/-4)
- `tile_engine/ops/gemm_preshuffle/CMakeLists.txt`  (+6/-7)
- `Jenkinsfile`  (+3/-7)
- `tile_engine/ops/gemm_preshuffle/gemm_preshuffle_benchmark.hpp`  (+8/-0)
- `test/ck_tile/gemm_tile_engine/configs/simple_test_config.json`  (+3/-3)
- `tile_engine/ops/gemm/CMakeLists.txt`  (+1/-1)

## Key added lines (kernel files)

**`tile_engine/ops/gemm_preshuffle/commons/validation_utils.py`**
```
if pipeline not in ["preshufflev2"]:
raise ValueError("Accepted pipeline values are: ['preshufflev2']")
if epilogue not in ["default", "cshuffle"]:
return ValueError("Accepted epilogue values are: ['default', 'cshuffle']")
```

**`tile_engine/ops/gemm_preshuffle/gemm_preshuffle_benchmark.hpp`**
```
struct KernelConfig
std::tuple<int, int, int> tile_dims;
std::tuple<int, int, int> warp_dims;
std::tuple<int, int, int> warp_tile_dims;
```

**`tile_engine/ops/gemm_preshuffle/gemm_preshuffle_benchmark_single.cpp`**
```
void benchmark_single(const ck_tile::ArgParser& arg_parser)
std::tuple<int, int, int> tile_dims =
std::make_tuple(SelectedKernel::TileM, SelectedKernel::TileN, SelectedKernel::TileK);
std::tuple<int, int, int> warp_dims = std::make_tuple(SelectedKernel::WarpPerBlock_M,
```

**`tile_engine/ops/gemm_preshuffle/gemm_preshuffle_common.hpp`**
```
template <typename T>
auto shuffle_b_permuteN(const ck_tile::HostTensor<T>& t,
ck_tile::index_t N_Warp_Tile,
ck_tile::index_t K_Warp_Tile,
```

**`tile_engine/ops/gemm_preshuffle/gemm_preshuffle_instance_builder.py`**
```
tile_config = self.config["tile_config"]
if tile_config.get("tile_m").get("values") is None:
tile_config.get("tile_m")["values"] = self._generate_values(
tile_config.get("tile_m").get("min"),
```
