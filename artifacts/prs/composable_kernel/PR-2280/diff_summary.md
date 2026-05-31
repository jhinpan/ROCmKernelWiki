# Diff summary

- **files changed:** 28 (diff was byte-capped; summary is partial)
- **lines:** +3364 / -830
- **kernel-ish files:** 22

## Files (by churn)

- `include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp`  (+2559/-0)
- `include/ck_tile/core/tensor/tile_window.hpp`  (+135/-407)
- `include/ck_tile/core/tensor/tile_window_linear.hpp`  (+165/-332)
- `include/ck_tile/core/tensor/tile_window_base.hpp`  (+256/-0)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+37/-25)
- `Jenkinsfile`  (+47/-1)
- `example/ck_tile/13_moe_sorting/moe_sorting.cpp`  (+19/-9)
- `example/ck_tile/11_add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd.cpp`  (+12/-9)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_multiple_d_xdl_cshuffle.hpp`  (+13/-7)
- `CMakeLists.txt`  (+11/-5)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_v2.hpp`  (+8/-8)
- `include/ck/library/utility/fill.hpp`  (+14/-0)
- `include/ck/tensor_operation/operator_transform/transform_conv_bwd_weight_to_gemm.hpp`  (+7/-7)
- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+12/-1)
- `example/ck_tile/16_batched_gemm/batched_gemm.cpp`  (+12/-1)

## Key added lines (kernel files)

**`example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_fp8_bpreshuffle.cpp`**
```
32,   32,
1,    1,   S<1, 32, 1, 8>, S<8, 8, 1>,
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
{F_occupancy},
{F_skip}>;
{F_pipeline_enum}, {F_logits}, fmha_mask_{F_idx}, {F_bias}, {F_lse}, {F_dropout}, {F_squant}, {F_spad}, {F_skpad}, {F_dp
FMHA_FWD_API_INNER_DISPATCH="""            {F_if}((t.is_group_mode == {F_mode}) && (t.is_v_rowmajor == {F_vlayout}) && (
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
ck_tile::index_t min_seqlen_q;
args.min_seqlen_q,
bool kPadDv_,
bool kSkipMinSeqlenQ_ = false>
```

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
int main(int argc, char* argv[])
return !run_gemm_example(argc, argv);
catch(const std::runtime_error& e)
std::cerr << "Runtime error: " << e.what() << '\n';
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
return !run_gemm_example(argc, argv);
```
