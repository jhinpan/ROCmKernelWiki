# Diff summary

- **files changed:** 7
- **lines:** +49 / -54
- **kernel-ish files:** 7

## Files (by churn)

- `include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`  (+21/-25)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+12/-21)
- `include/ck/utility/amd_ck_fp8.hpp`  (+4/-4)
- `example/ck_tile/21_elementwise/elementwise_example.cpp`  (+3/-1)
- `example/ck_tile/21_elementwise/elementwise_example_add_4d.cpp`  (+3/-1)
- `example/ck_tile/21_elementwise/elementwise_example_transpose.cpp`  (+3/-1)
- `example/ck_tile/21_elementwise/elementwise_example_unary.cpp`  (+3/-1)

## Key added lines (kernel files)

**`example/ck_tile/21_elementwise/elementwise_example.cpp`**
```
bool result = true;
ck_tile::ArgParser arg_parser;
std::tie(result, arg_parser) = create_args(argc, argv);
```

**`example/ck_tile/21_elementwise/elementwise_example_add_4d.cpp`**
```
bool result = true;
ck_tile::ArgParser arg_parser;
std::tie(result, arg_parser) = create_args(argc, argv);
```

**`example/ck_tile/21_elementwise/elementwise_example_transpose.cpp`**
```
bool result = true;
ck_tile::ArgParser arg_parser;
std::tie(result, arg_parser) = create_args(argc, argv);
```

**`example/ck_tile/21_elementwise/elementwise_example_unary.cpp`**
```
bool result = true;
ck_tile::ArgParser arg_parser;
std::tie(result, arg_parser) = create_args(argc, argv);
```

**`include/ck/utility/amd_ck_fp8.hpp`**
```
using data_type  = unsigned char;
data_type m_data = data_type{};
using data_type  = unsigned char;
data_type m_data = data_type{};
```
