# Diff summary

- **files changed:** 40
- **lines:** +300 / -18
- **kernel-ish files:** 40

## Files (by churn)

- `include/ck_tile/host/concat.hpp`  (+122/-0)
- `include/ck_tile/ops/common/utils.hpp`  (+34/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1.hpp`  (+16/-2)
- `include/ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp`  (+15/-1)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`  (+13/-2)
- `include/ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp`  (+12/-1)
- `include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`  (+12/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp`  (+11/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp`  (+10/-0)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+8/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v2.hpp`  (+8/-0)
- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+5/-2)
- `example/ck_tile/16_batched_gemm/batched_gemm.cpp`  (+5/-2)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+2/-2)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp`  (+2/-1)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
<< "shape: " << CodegenGemmShape::GetName() << '\n'
<< "problem: " << CodegenPipelineProblem::GetName() << '\n'
<< "pipeline: " << CodegenGemmPipeline::GetName() << '\n'
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
std::cout << "The CPU verification result is:" << (pass ? "correct" : "fail") << std::endl;
std::cout << "The GPU verification result is: " << (pass ? "correct" : "fail") << std::endl;
```

**`example/ck_tile/16_batched_gemm/batched_gemm.cpp`**
```
std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
<< "shape: " << CodegenGemmShape::GetName() << '\n'
<< "problem: " << CodegenPipelineProblem::GetName() << '\n'
<< "pipeline: " << CodegenGemmPipeline::GetName() << '\n'
```

**`example/ck_tile/16_batched_gemm/run_batched_gemm_example.inc`**
```
std::cout << "The CPU verification result is:" << (pass ? "correct" : "fail") << std::endl;
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`**
```
std::cout << "Launching kernel: " << GroupedGemmKernel::GetName() << " with args:"
```
