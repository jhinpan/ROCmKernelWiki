# Candidate Ledgers

One YAML ledger per tracked repository, produced by
[`../scripts/harvest_prs.py`](../scripts/harvest_prs.py). Each ledger records the
include / defer / exclude decision for **every merged PR scanned** in that repo,
along with the classification reason, so the corpus is auditable and refreshable.

| File | Repo |
|---|---|
| `composable_kernel.yaml` | ROCm/composable_kernel |
| `aiter.yaml` | ROCm/aiter |
| `hipBLASLt.yaml` | ROCm/hipBLASLt |
| `Tensile.yaml` | ROCm/Tensile |
| `rocBLAS.yaml` | ROCm/rocBLAS |
| `flash-attention.yaml` | ROCm/flash-attention |
| `FlyDSL.yaml` | ROCm/FlyDSL |
| `triton.yaml` | ROCm/triton |
| `vllm.yaml` | vllm-project/vllm (ROCm-filtered) |
| `sglang.yaml` | sgl-project/sglang (ROCm-filtered) |

## Schema

```yaml
repo: ROCm/composable_kernel
scanned: 2710        # merged PRs examined
included: 2098       # PRs that produced a sources/prs/<repo>/PR-<N>.md page
deferred: ...        # kernel-adjacent but no clear signal
excluded: ...        # CI/docs/version-bump/host-only
cutoff: "2026-05-15"
prs:
  - pr: 1234
    title: "..."
    merged_at: "2026-03-12"
    decision: include            # include | defer | exclude
    reason: "kernel path '...'; keyword 'fp8'"
```

Classification policy lives in
[`../data/inclusion-policy.yaml`](../data/inclusion-policy.yaml). To refresh:

```bash
python3 scripts/harvest_prs.py --all
python3 scripts/generate-indices.py
python3 scripts/validate.py
# then bump data/refresh-cutoff.yaml
```
