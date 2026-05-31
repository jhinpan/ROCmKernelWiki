# Diff summary

- **files changed:** 8
- **lines:** +205 / -65
- **kernel-ish files:** 8

## Files (by churn)

- `example/ck_tile/02_layernorm2d/generate.py`  (+77/-55)
- `include/ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_pipeline_two_pass.hpp`  (+34/-5)
- `example/ck_tile/02_layernorm2d/layernorm2d_fwd.cpp`  (+28/-3)
- `include/ck_tile/ops/layernorm2d/kernel/layernorm2d_fwd_kernel.hpp`  (+28/-0)
- `include/ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_pipeline_one_pass.hpp`  (+18/-2)
- `include/ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_traits.hpp`  (+15/-0)
- `example/ck_tile/02_layernorm2d/layernorm2d_fwd.hpp`  (+3/-0)
- `include/ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_pipeline_problem.hpp`  (+2/-0)

## Key added lines (kernel files)

**`example/ck_tile/02_layernorm2d/generate.py`**
```
XBIAS_ENUM_STR_MAP = [
'xbias']      # pre-norm add bias
ck_tile::index_t kXbias_ = 0,
static constexpr ck_tile::index_t kXbias = kXbias_;
```

**`example/ck_tile/02_layernorm2d/layernorm2d_fwd.cpp`**
```
.insert("xbias", "0", "add bias, 0:no add, 1:add bias before fadd")
int xbias         = arg_parser.get_int("xbias");
using XBiasDataType     = typename TypeConfig::XBiasDataType;
ck_tile::HostTensor<XBiasDataType> x_bias_host({n});
```

**`example/ck_tile/02_layernorm2d/layernorm2d_fwd.hpp`**
```
using XBiasDataType   = ck_tile::half_t;
using XBiasDataType   = ck_tile::bf16_t;
int xbias;          // 0:no-bias, 1:add bias
```

**`include/ck_tile/ops/layernorm2d/kernel/layernorm2d_fwd_kernel.hpp`**
```
const void* p_x_bias;     // [1, n], bias, prec same as input
using XBiasDataType   = remove_cvref_t<typename Problem::XBiasDataType>;
static constexpr auto kXbias      = Problem::Traits::kXbias;
const void* p_x_bias;     // [1, n], bias, prec same as input
```

**`include/ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_pipeline_one_pass.hpp`**
```
using XBiasDataType   = ck_tile::remove_cvref_t<typename Problem::XBiasDataType>;
static constexpr auto kXbias             = Problem::Traits::kXbias;
typename XBiasWindow,
const XBiasWindow& x_bias_window_,
```
