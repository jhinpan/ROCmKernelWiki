# Diff summary

- **files changed:** 33
- **lines:** +861 / -280
- **kernel-ish files:** 33

## Files (by churn)

- `op_tests/test_mha_varlen.py`  (+150/-28)
- `op_tests/test_kvcache_blockscale.py`  (+105/-70)
- `op_tests/test_mha.py`  (+105/-14)
- `op_tests/test_mla.py`  (+85/-26)
- `op_tests/test_kvcache.py`  (+55/-32)
- `op_tests/test_pa_v1.py`  (+44/-3)
- `op_tests/test_pa_ragged.py`  (+42/-3)
- `op_tests/test_rope.py`  (+27/-11)
- `op_tests/test_moe_2stage.py`  (+27/-8)
- `op_tests/test_moe_tkw1.py`  (+16/-7)
- `op_tests/test_gemm_a8w8_blockscale.py`  (+15/-5)
- `op_tests/test_pa_mtp.py`  (+14/-6)
- `op_tests/test_moe_blockscale.py`  (+14/-4)
- `op_tests/test_moe_sorting_mxfp4.py`  (+15/-3)
- `op_tests/test_activation.py`  (+10/-6)

## Key added lines (kernel files)

**`op_tests/multigpu_tests/test_communication.py`**
```
help="shape. e.g. -s 128,8192",
```

**`op_tests/multigpu_tests/test_custom_allreduce.py`**
```
help="shape. e.g. -s 128,8192",
```

**`op_tests/multigpu_tests/test_custom_allreduce_fp8.py`**
```
help="shape. e.g. -s 128,8192",
```

**`op_tests/test_activation.py`**
```
parser = argparse.ArgumentParser(
formatter_class=argparse.RawTextHelpFormatter,
description="config input of test",
help="""Data type.
```

**`op_tests/test_batch_prefill.py`**
```
parser = argparse.ArgumentParser(
formatter_class=argparse.RawTextHelpFormatter,
description="config input of test",
help="""Causal mask mode (False or True).
```
