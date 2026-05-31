# Diff summary

- **files changed:** 27
- **lines:** +2428 / -864
- **kernel-ish files:** 21

## Files (by churn)

- `example/ck_tile/01_fmha/fmha_fwd_runner.hpp`  (+339/-433)
- `test/ck_tile/fmha/test_fmha_fwd.inc`  (+628/-0)
- `example/ck_tile/01_fmha/fmha_bwd_runner.hpp`  (+170/-232)
- `test/ck_tile/fmha/test_fmha_bwd.inc`  (+344/-0)
- `example/ck_tile/01_fmha/example_fmha_fwd.cpp`  (+253/-0)
- `example/ck_tile/01_fmha/example_fmha_bwd.cpp`  (+183/-0)
- `example/ck_tile/01_fmha/utils.hpp`  (+59/-81)
- `example/ck_tile/01_fmha/CMakeLists.txt`  (+78/-38)
- `example/ck_tile/01_fmha/mask.hpp`  (+35/-44)
- `include/ck_tile/host/reference/reference_batched_dropout_randval.hpp`  (+70/-0)
- `example/ck_tile/01_fmha/bias.hpp`  (+34/-20)
- `test/ck_tile/fmha/test_fmha_fwd_bf16.cpp`  (+44/-0)
- `test/ck_tile/fmha/test_fmha_fwd_fp16.cpp`  (+44/-0)
- `test/ck_tile/fmha/test_fmha_fwd_fp8.cpp`  (+43/-0)
- `test/ck_tile/fmha/CMakeLists.txt`  (+31/-0)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/bias.hpp`**
```
auto found_0 = str.find(':');
if(found_0 != std::string::npos)
std::string t = str.substr(0, found_0);
std::string v = str.substr(found_0 + 1);
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`**
```
if self.skpad == 't' : return f'true /*a.seqlen_k_ptr != nullptr || a.seqlen_k % {self.bn0} != 0*/' # TODO: order of get
else :                return f'a.seqlen_k_ptr == nullptr && a.seqlen_k % {self.bn0} == 0'
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_pagedkv_prefill.py`**
```
if self.skpad == 't' : return f'true /*a.seqlen_k_ptr != nullptr || a.seqlen_k % {self.bn0} != 0*/' # TODO: order of get
else :                return f'a.seqlen_k_ptr == nullptr && a.seqlen_k % {self.bn0} == 0'
```

**`example/ck_tile/01_fmha/example_fmha_bwd.cpp`**
```
auto create_args(int argc, char* argv[])
ck_tile::ArgParser arg_parser;
arg_parser.insert("v", "1", "whether do CPU validation or not")
.insert("mode", "0", "kernel mode. 0:batch, 1:group")
```

**`example/ck_tile/01_fmha/example_fmha_fwd.cpp`**
```
auto create_args(int argc, char* argv[])
ck_tile::ArgParser arg_parser;
arg_parser.insert("v", "1", "0:no validation, 2:cpu validation, 2:gpu validation(experimental)")
.insert("mode", "0", "kernel mode. 0:batch, 1:group")
```
