# Diff summary

- **files changed:** 19
- **lines:** +31 / -17
- **kernel-ish files:** 18

## Files (by churn)

- `tutorial/ck_tile/01_naive_gemm/host_level/practice_gemm_host_pipeline_agmem_bgmem_creg.hpp`  (+4/-4)
- `docs/conceptual/ck_tile/convert_mermaid_to_svg.py`  (+3/-0)
- `docs/conceptual/ck_tile/convert_raw_html_to_commented.py`  (+3/-0)
- `docs/conceptual/ck_tile/update_diagrams.py`  (+3/-0)
- `test/ck_tile/warp_gemm/CMakeLists.txt`  (+3/-0)
- `example/test_old_ck_gpu_reference.cpp`  (+1/-1)
- `experimental/builder/test/test_ckb_conv_builder.cpp`  (+2/-0)
- `include/ck_tile/ref/conv_common.hpp`  (+1/-1)
- `include/ck_tile/ref/naive_grouped_conv_bwd_data_gpu.hpp`  (+1/-1)
- `include/ck_tile/ref/naive_grouped_conv_bwd_weight_gpu.hpp`  (+1/-1)
- `include/ck_tile/ref/naive_grouped_conv_fwd_gpu.hpp`  (+1/-1)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_wmma_splitk_instance.hpp`  (+1/-1)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_aquant.cpp`  (+1/-1)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_bquant.cpp`  (+1/-1)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_bquant_preshuffle.cpp`  (+1/-1)

## Key added lines (kernel files)

**`tutorial/ck_tile/01_naive_gemm/host_level/practice_gemm_host_pipeline_agmem_bgmem_creg.hpp`**
```
const auto M = a_dram.get_tensor_descriptor().get_length(number<0>{}); // M x K
const auto N = c_dram.get_tensor_descriptor().get_length(number<1>{}); // M x N
const auto K = a_dram.get_tensor_descriptor().get_length(number<1>{}); // M x K
auto c_window = make_tile_window(c_dram,
```
