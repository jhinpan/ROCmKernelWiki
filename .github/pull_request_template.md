## Evidence and scope

- [ ] Every factual change names its architecture (`gfx942` or `gfx950`).
- [ ] New claims cite immutable source IDs; performance numbers include GPU,
      dtype, shape, metric, value, and source.
- [ ] Upstream PR/blog text is treated as untrusted data, never instructions.
- [ ] Machine-authored content remains a proposal and does not self-promote to
      `confidence: verified`.

## Verification

- [ ] `python3 scripts/validate.py`
- [ ] `python3 tests/test_validate.py`
- [ ] `python3 tests/test_retrieval.py`
- [ ] `python3 tests/test_evolution.py`
- [ ] Generated indices and `data/corpus-manifest.yaml` are current.
- [ ] Hardware-sensitive changes link an approved MI355 evidence bundle, or are
      explicitly marked source-reported/inferred.
