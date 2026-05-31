# Diff summary

- **files changed:** 22
- **lines:** +897 / -1456
- **kernel-ish files:** 21

## Files (by churn)

- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+576/-263)
- `example/ck_tile/01_fmha/example_fmha_fwd_v3.cpp`  (+0/-616)
- `example/ck_tile/01_fmha/fmha_fwd_v3_impl.hpp`  (+0/-179)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_v3_kernel.hpp`  (+80/-54)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+94/-0)
- `include/ck_tile/core/algorithm/coordinate_transform.hpp`  (+82/-0)
- `example/ck_tile/01_fmha/fmha_fwd_v3.hpp`  (+0/-73)
- `example/ck_tile/01_fmha/fmha_fwd_v3.cpp`  (+0/-60)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+0/-48)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp`  (+0/-43)
- `example/ck_tile/01_fmha/CMakeLists.txt`  (+0/-34)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp`  (+26/-8)
- `example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`  (+17/-3)
- `include/ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp`  (+0/-16)
- `example/ck_tile/01_fmha/instances/fmha_fwd_v3_d128_bf16_mask.cpp`  (+0/-14)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`**
```
def get_mask_map(mask_impl: str):
if mask_impl == "generic":
elif mask_impl == "simplified":
def get_mask_impl(mask: str) -> str:
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
from typing import Callable, ClassVar, Iterable, List, Optional, Tuple
get_mask_cpp_type,
get_mask_cpp_check_expr,
FMHA_FWD_KERNEL_BODY_TEMPLATE = """
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
template <typename FmhaKernel>
auto fmha_fwd_v3_create_kargs_and_grids(fmha_fwd_args args)
int remap_opt = 2;
if(args.mask_type != static_cast<int>(mask_enum::no_mask) &&
```

**`include/ck_tile/core/algorithm/coordinate_transform.hpp`**
```
template <typename Functor, typename LowLength>
struct functor_transform : public base_transform<1, 1>
using LowerIndex = multi_index<1>;
using UpperIndex = multi_index<1>;
```

**`include/ck_tile/core/tensor/tile_window.hpp`**
```
template <typename TensorView_,
typename WindowLengths_,
typename = std::enable_if_t<is_tensor_view_v<TensorView_>>>
template <typename TensorView,
```
