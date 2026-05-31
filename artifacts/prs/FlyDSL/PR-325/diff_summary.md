# Diff summary

- **files changed:** 32
- **lines:** +192 / -7
- **kernel-ish files:** 27

## Files (by churn)

- `tests/README.md`  (+93/-0)
- `tests/conftest.py`  (+29/-5)
- `docs/cute_layout_algebra_guide.md`  (+3/-2)
- `tests/pytest.ini`  (+5/-0)
- `tests/unit/test_launch_overhead.py`  (+4/-0)
- `tests/kernels/test_layernorm.py`  (+3/-0)
- `tests/kernels/test_rmsnorm.py`  (+3/-0)
- `tests/kernels/test_softmax.py`  (+3/-0)
- `tests/unit/test_rocir_print.py`  (+3/-0)
- `CONTRIBUTING.md`  (+2/-0)
- `README.md`  (+2/-0)
- `tests/kernels/test_blockscale_preshuffle_gemm.py`  (+2/-0)
- `tests/kernels/test_moe_blockscale.py`  (+2/-0)
- `tests/kernels/test_moe_gemm.py`  (+2/-0)
- `tests/kernels/test_moe_reduce.py`  (+2/-0)

## Key added lines (kernel files)

**`tests/conftest.py`**
```
def pytest_addoption(parser):
"""Add FlyDSL test-session options that map to env variables."""
group = parser.getgroup("flydsl")
group.addoption(
```

**`tests/kernels/test_blockscale_preshuffle_gemm.py`**
```
pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]
```

**`tests/kernels/test_layernorm.py`**
```
pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]
```

**`tests/kernels/test_moe_blockscale.py`**
```
pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]
```

**`tests/kernels/test_moe_gemm.py`**
```
pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]
```
