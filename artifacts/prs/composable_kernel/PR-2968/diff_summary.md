# Diff summary

- **files changed:** 25
- **lines:** +223 / -167
- **kernel-ish files:** 25

## Files (by churn)

- `example/22_cgemm/cgemm_xdl_fp32.cpp`  (+25/-20)
- `example/24_batched_gemm/run_batched_gemm_example_fp16int4_b_scale.inc`  (+26/-18)
- `example/22_cgemm/cgemm_xdl_int8.cpp`  (+21/-21)
- `example/24_batched_gemm/run_batched_gemm_example.inc`  (+23/-19)
- `example/44_elementwise_permute/elementwise_binary_4D_fp16.cpp`  (+14/-21)
- `example/44_elementwise_permute/elementwise_scale_permute_amax_2D_fp16_fp8.cpp`  (+15/-13)
- `example/36_sparse_embedding/sparse_embedding3_forward_layernorm.cpp`  (+13/-10)
- `example/20_grouped_conv_bwd_weight/grouped_conv_bwd_weight_xdl_bf16.cpp`  (+10/-5)
- `example/01_gemm/gemm_xdl_fp8.cpp`  (+11/-3)
- `example/01_gemm/gemm_xdl_fp8_bf8.cpp`  (+11/-3)
- `example/24_batched_gemm/batched_gemm_xdl_fp32.cpp`  (+12/-2)
- `example/24_batched_gemm/batched_gemm_xdl_bf16_v3.cpp`  (+6/-6)
- `example/24_batched_gemm/batched_gemm_xdl_fp8_rowwise_v3.cpp`  (+6/-6)
- `example/24_batched_gemm/batched_gemm_xdl_bf16.cpp`  (+4/-2)
- `example/24_batched_gemm/batched_gemm_xdl_fp16.cpp`  (+4/-2)

## Key added lines (kernel files)

**`example/01_gemm/gemm_xdl_bf16.cpp`**
```
< ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp,  
```

**`example/01_gemm/gemm_xdl_fp16_fp8.cpp`**
```
< ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp,  
```

**`example/01_gemm/gemm_xdl_fp8.cpp`**
```
< ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp,  
int main(int argc, char* argv[])
if(ck::is_gfx11_supported())
return 0;
```

**`example/01_gemm/gemm_xdl_fp8_bf8.cpp`**
```
< ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp,  
int main(int argc, char* argv[])
if(ck::is_gfx11_supported())
return 0;
```

**`example/01_gemm/gemm_xdl_int8.cpp`**
```
< ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp,  
```
