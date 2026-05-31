# Diff summary

- **files changed:** 30 (diff was byte-capped; summary is partial)
- **lines:** +3348 / -628
- **kernel-ish files:** 26

## Files (by churn)

- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3_mx.hpp`  (+1090/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3_mx_bpreshuffle.hpp`  (+1042/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1_mx.hpp`  (+300/-241)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_mx.hpp`  (+103/-164)
- `example/67_gemm_microscaling/gemm_mx_common.hpp`  (+194/-68)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_mx_pipeline_xdlops_base.hpp`  (+113/-53)
- `example/67_gemm_microscaling/gemm_mx_fp4.cpp`  (+105/-0)
- `example/67_gemm_microscaling/gemm_mx_fp4_bpreshuffle.cpp`  (+105/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_mx_bpreshuffle_selector.hpp`  (+68/-0)
- `include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_direct_load.hpp`  (+49/-14)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_mx_selector.hpp`  (+25/-30)
- `include/ck/tensor_operation/gpu/device/device_gemm_mx.hpp`  (+38/-0)
- `example/67_gemm_microscaling/CMakeLists.txt`  (+35/-2)
- `example/67_gemm_microscaling/gemm_mx_bf8.cpp`  (+12/-11)
- `example/67_gemm_microscaling/gemm_mx_fp8.cpp`  (+12/-11)

## Key added lines (kernel files)

**`example/04_gemm_add_add_fastgelu/gemm_add_add_fastgelu_xdl_lds_direct_load_fp32.cpp`**
```
< ALayout, BLayout, DsLayout, ELayout, ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AEle
```

**`example/24_batched_gemm/batched_gemm_xdl_fp8_rowwise_v3.cpp`**
```
64,             // KPerBlock
16,             // AK1
16,             // BK1
0,              // ABlockLdsExtraM
```

**`example/35_splitK_gemm/splitK_gemm_xdl_lds_direct_load_fp16.cpp`**
```
< ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout, CLayout,  AElementOp,  BElementOp,  CElementOp,    Gem
```

**`example/67_gemm_microscaling/gemm_mx_bf8.cpp`**
```
constexpr ck::index_t KPerBlock      = 256;
constexpr auto BlkGemmPVer   = ck::BlockGemmPipelineVersion::v3;
32,               // NPerBlock
2,                // NXdlPerWave
```

**`example/67_gemm_microscaling/gemm_mx_common.hpp`**
```
using Row  = ck::tensor_layout::gemm::RowMajor;
using Col  = ck::tensor_layout::gemm::ColumnMajor;
using MFMA = ck::tensor_layout::gemm::MFMA;
int warm_up         = 10;
```
