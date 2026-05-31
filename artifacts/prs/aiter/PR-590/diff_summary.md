# Diff summary

- **files changed:** 15
- **lines:** +1111 / -399
- **kernel-ish files:** 15

## Files (by churn)

- `op_tests/test_moe_ep.py`  (+147/-125)
- `op_tests/test_moe.py`  (+142/-97)
- `op_tests/test_rope.py`  (+149/-60)
- `op_tests/test_moe_2stage.py`  (+104/-15)
- `op_tests/test_pa_mtp.py`  (+71/-6)
- `op_tests/test_moe_sorting.py`  (+62/-8)
- `op_tests/test_moe_sorting_mxfp4.py`  (+62/-8)
- `op_tests/test_moe_tkw1.py`  (+64/-5)
- `op_tests/test_quant.py`  (+59/-10)
- `op_tests/test_moeTopkSoftmax.py`  (+60/-5)
- `op_tests/test_rmsnorm2dFusedAddQuant.py`  (+36/-24)
- `op_tests/test_mha.py`  (+29/-25)
- `op_tests/test_moe_blockscale.py`  (+49/-4)
- `op_tests/test_rmsnorm2d.py`  (+43/-3)
- `op_tests/test_smoothquant.py`  (+34/-4)

## Key added lines (kernel files)

**`op_tests/test_mha.py`**
```
import argparse
parser = argparse.ArgumentParser(description="config input of test")
parser.add_argument("-b", "--batch_size", type=int, default=2)
parser.add_argument("-n", "--nheads", type=int, default=5)
```

**`op_tests/test_moe.py`**
```
import argparse
parser = argparse.ArgumentParser(description="select test")
l_test = [
"test_fmoe_16_bit",
```

**`op_tests/test_moeTopkSoftmax.py`**
```
import argparse
l_dtype = ["bf16", "fp16"]
l_expert = [64, 256]
l_m = [1, 8, 16, 32, 64, 128, 256, 65536, 163840]
```

**`op_tests/test_moe_2stage.py`**
```
import argparse
import pandas as pd
l_dtype = ["bf16", "fp16"]
l_dim = [(6144, 4096)]
```

**`op_tests/test_moe_blockscale.py`**
```
import argparse
l_dtype = ["bf16"]
l_m = [1, 2, 5, 16, 32, 163840]
parser = argparse.ArgumentParser(description="config input of test")
```
