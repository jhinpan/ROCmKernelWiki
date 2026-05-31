# Diff summary

- **files changed:** 7
- **lines:** +109 / -47
- **kernel-ish files:** 7

## Files (by churn)

- `example/ck_tile/18_flatmm/run_flatmm_example.inc`  (+41/-33)
- `example/ck_tile/13_moe_sorting/moe_sorting.cpp`  (+19/-9)
- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+12/-1)
- `example/ck_tile/16_batched_gemm/batched_gemm.cpp`  (+12/-1)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`  (+12/-1)
- `example/ck_tile/18_flatmm/flatmm_basic.cpp`  (+12/-1)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+1/-1)

## Key added lines (kernel files)

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

**`example/ck_tile/13_moe_sorting/moe_sorting.cpp`**
```
auto [result, args] = create_args(argc, argv);
if(!result)
return -1;
std::string index_prec  = args.get_str("pr_i");
```

**`example/ck_tile/16_batched_gemm/batched_gemm.cpp`**
```
int main(int argc, char* argv[])
return !run_batched_gemm_example(argc, argv);
catch(const std::runtime_error& e)
std::cerr << "Runtime error: " << e.what() << '\n';
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`**
```
int main(int argc, char* argv[])
return !run_grouped_gemm_example<Persistent>(argc, argv);
catch(const std::runtime_error& e)
std::cerr << "Runtime error: " << e.what() << '\n';
```
