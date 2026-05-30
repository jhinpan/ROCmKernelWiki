# Worked Query Examples

Concrete patterns showing how to translate a user question into a navigation
path and synthesize an answer from this knowledge base.

---

## Example 1: "How do I write a fast GEMM on MI300?"

**Navigation path**:
1. `queries/by-kernel-type.md` → `gemm` / `hgemm` rows
2. Read `wiki/kernels/ck-hgemm.md`
3. Optimization path: `wiki/techniques/lds-double-buffering.md`,
   `wiki/techniques/mfma-pipelining.md`, `wiki/techniques/lds-swizzling.md`
4. Hardware grounding: `wiki/hardware/mfma.md`, `wiki/hardware/lds.md`
5. Follow `sources:` to real PRs in ROCm/composable_kernel and the GEMM blog

**Command**:
```bash
python3 scripts/query.py --type kernel --tag gemm --architecture gfx942
python3 scripts/get_page.py kernel-ck-hgemm --follow-sources
```

---

## Example 2: "My kernel has LDS bank conflicts"

**Navigation path**:
1. `queries/by-problem.md` → `bank-conflicts`
2. Pattern page `wiki/patterns/bank-conflicts.md`
3. Candidate techniques: `technique-lds-swizzling`, `technique-bank-conflict-avoidance`
4. Hardware: `wiki/hardware/lds.md` (32-bank gfx942 vs 64-bank gfx950)

**Command**:
```bash
python3 scripts/query.py --symptom bank-conflicts
python3 scripts/get_page.py pattern-bank-conflicts
```

---

## Example 3: "What changed for FP8 between MI300 and MI350?"

**Navigation path**:
1. `wiki/migration/gfx942-to-gfx950.md` — the FNUZ→OCP FP8 break, f8f6f4, MX scales
2. `wiki/hardware/mxfp.md` — block-scaled FP8/FP6/FP4 details
3. `wiki/hardware/mfma.md` — the scaled MFMA instructions

**Command**:
```bash
python3 scripts/get_page.py migration-gfx942-to-gfx950
python3 scripts/query.py --tag mxfp --architecture gfx950
```

---

## Example 4: "Show me how AITER/CK implemented fused MoE on AMD"

**Navigation path**:
1. `wiki/kernels/fused-moe.md`
2. Follow `sources:` → `ref-aiter`, and real PRs
3. `python3 scripts/query.py --repo aiter --tag moe`
4. Related: `kernel-grouped-gemm`, `technique-kernel-fusion`

**Command**:
```bash
python3 scripts/get_page.py kernel-fused-moe --follow-sources
python3 scripts/query.py --repo aiter --tag fused-moe --limit 20
```

---

## Example 5: "How does AMD do async copy without cp.async?"

**Navigation path**:
1. `wiki/hardware/async-copy-lds.md` — direct-to-LDS (`buffer_load…lds`)
2. `wiki/hardware/s-waitcnt.md` — vmcnt gating
3. `wiki/techniques/lds-double-buffering.md` — using it for pipelining
4. Contrast in `wiki/migration/cuda-to-hip.md`

**Command**:
```bash
python3 scripts/grep_wiki.py "global_load_lds" "buffer_load.*lds" --any
python3 scripts/get_page.py hw-async-copy-lds
```

---

## Example 6: "Find all merged PRs touching gfx950 FP8 in Composable Kernel"

**Navigation path**:
1. `queries/by-repo.md` → ROCm/composable_kernel section
2. Filter by tag + architecture

**Command**:
```bash
python3 scripts/query.py --repo composable_kernel --tag fp8 --architecture gfx950 --limit 40
python3 scripts/grep_wiki.py "gfx950" --only sources
```

---

## Example 7: "Write a wave-level reduction on MI300"

**Navigation path**:
1. `wiki/techniques/wave-reduce.md`
2. `wiki/hardware/cross-lane.md` (DPP, ds_bpermute; permlane16 is gfx950-only)
3. Reference code in `ref-gcnasm` (`wave_reduce_dpp`)

**Command**:
```bash
python3 scripts/get_page.py technique-wave-reduce
python3 scripts/grep_wiki.py "ds_bpermute|mov_dpp" --any
```

---

## Output contract (when answering from this KB)

1. **Cite specific pages** by path and id (e.g., `wiki/kernels/fp8-gemm.md`,
   `kernel-fp8-gemm`).
2. **Follow `sources:`** to trace claims to PRs/docs/blogs/refs.
3. **Respect confidence**: `verified` > `source-reported` > `inferred` >
   `experimental`. Call out non-verified claims.
4. **Report performance claims with all fields**: gpu, dtype, shape, metric,
   value, source_id.
5. **State the architecture**: a claim true for gfx942 may not hold for gfx950
   (and vice-versa) — especially for FP8 numerics and LDS sizing.
