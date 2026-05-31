# Diff summary

- **files changed:** 17
- **lines:** +93 / -473
- **kernel-ish files:** 17

## Files (by churn)

- `example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`  (+32/-55)
- `tile_engine/ops/gemm/gemm_common.hpp`  (+0/-59)
- `tile_engine/ops/gemm_multi_d/gemm_multi_d_common.hpp`  (+0/-59)
- `tile_engine/ops/gemm_preshuffle/gemm_preshuffle_common.hpp`  (+0/-59)
- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+0/-57)
- `example/ck_tile/40_streamk_gemm/gemm_utils.hpp`  (+0/-33)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+17/-15)
- `example/ck_tile/18_flatmm/flatmm_basic.hpp`  (+0/-32)
- `example/ck_tile/18_flatmm/moe_flatmm.hpp`  (+0/-32)
- `include/ck_tile/ops/common/utils.hpp`  (+13/-11)
- `example/ck_tile/20_grouped_convolution/conv_configs.hpp`  (+0/-21)
- `example/ck_tile/05_reduce/reduce.cpp`  (+1/-16)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+7/-6)
- `tile_engine/ops/gemm/gemm_benchmark_single.cpp`  (+6/-6)
- `tile_engine/ops/gemm_preshuffle/gemm_preshuffle_benchmark_single.cpp`  (+6/-6)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_splitk_two_stage_reduce.cpp`**
```
<< " C_Layout=" << CLayout::name
<< " A_Type=" << ck_tile::DataTypeTraits<ADataType>::name
<< " B_Type=" << ck_tile::DataTypeTraits<BDataType>::name
<< " C_Type=" << ck_tile::DataTypeTraits<CDataType>::name
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
<< " C_Layout=" << CLayout::name
<< " A_Type=" << ck_tile::DataTypeTraits<ADataType>::name
<< " B_Type=" << ck_tile::DataTypeTraits<BDataType>::name
<< " C_Type=" << ck_tile::DataTypeTraits<CDataType>::name
```

**`example/ck_tile/05_reduce/reduce.cpp`**
```
dump_reduce_json_results<DataType, ck_tile::DataTypeTraits>(
```

**`example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`**
```
auto create_args(int argc, char* argv[])
ck_tile::ArgParser arg_parser;
arg_parser.insert("m", "3840", "m dimension")
.insert("n", "4096", "n dimension")
```

**`example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`**
```
std::cout << " A_Type = " << ck_tile::DataTypeTraits<typename TypeConfig::ADataType>::name
<< " AQ_Type = " << ck_tile::DataTypeTraits<typename TypeConfig::QDataType>::name
<< " B_Type = " << ck_tile::DataTypeTraits<typename TypeConfig::BDataType>::name;
std::cout << " BQ_Type = " << ck_tile::DataTypeTraits<typename TypeConfig::QDataType>::name;
```
