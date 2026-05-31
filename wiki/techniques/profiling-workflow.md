---
id: technique-profiling-workflow
title: Profiling & Roofline Workflow on ROCm (rocprof / Omniperf)
type: technique
architectures:
- gfx942
- gfx950
tags:
- occupancy-tuning
- vgpr
- lds
- bank-conflicts
- memory-bound
- mfma-underutilized
- low-compute-utilization
confidence: source-reported
reproducibility: snippet
hardware_features:
- matrix-core
- lds
- vgpr
- wave64
- l2-cache
- hbm3
languages:
- hip
related:
- hw-wavefront
- hw-lds
- technique-occupancy-tuning
- pattern-memory-bound
- pattern-bank-conflicts
- pattern-mfma-underutilized
- pattern-low-occupancy
- pattern-vgpr-pressure
sources:
- doc-rocm-hip-hw
- blog-gemm-optimization
- blog-triton-optimizations
implemented_by:
- pr-Tensile-1383
- pr-sglang-25898
- pr-composable_kernel-3137
- pr-composable_kernel-2825
- pr-composable_kernel-2319
- pr-Tensile-1529
- pr-composable_kernel-647
- pr-composable_kernel-2723
---
# Profiling & Roofline Workflow on ROCm (rocprof / Omniperf)

## The toolchain in one breath

ROCm ships a layered set of GPU profilers. Pick by the question you are asking:

| Tool | Current binary | Use it to answer |
|---|---|---|
| **rocprofv3** | `rocprofv3` | "Which kernels dominate? What are their raw hardware counters?" |
| **rocprof (legacy)** | `rocprof` | Same, older CLI/CSV format; still common in scripts |
| **Omniperf** | `rocprof-compute` | "*Why* is this one kernel slow?" — roofline, LDS/VALU/MFMA breakdown |
| **Omnitrace** | `rocprof-sys-*` | "How do host launches, copies, and kernels overlap on a timeline?" |

> **Naming churn.** As of ROCm 6.2+, *Omniperf* was upstreamed as
> **ROCm Compute Profiler** (`rocprof-compute`) and *Omnitrace* as **ROCm
> Systems Profiler** (`rocprof-sys`). The old `omniperf` / `omnitrace` entry
> points still exist as shims on many installs. Counter *names* below are stable
> across both. Treat exact CLI flags as version-sensitive.

The normal loop is **top-down**: `rocprofv3` to find the hot kernel, then
`rocprof-compute` (Omniperf) to diagnose it, then a [pattern page](../patterns/)
to fix it.

## Step 1 — find the hot kernel (rocprofv3)

Start with a wall-clock kernel summary. This needs no counter list and perturbs
timing the least:

```bash
# Per-kernel timing summary -> ./out/results_kernel_trace.csv
rocprofv3 --kernel-trace --output-format csv -d ./out -- ./my_app

# Legacy equivalent:
rocprof --stats ./my_app          # writes results.stats.csv
```

Sort the CSV by total duration. The top one or two kernels are where every later
step should focus — optimizing anything else is wasted effort.

## Step 2 — collect hardware counters

Counters are sampled by re-running the dispatch, so list only what you need.
Provide an input file (one counter per line, or `pmc:` groups):

```bash
cat > counters.txt <<'EOF'
pmc: VALUUtilization VALUBusy MFMABusy
pmc: LDSBankConflict MemUnitBusy MemUnitStalled WriteUnitStalled
pmc: FetchSize WriteSize
pmc: MeanOccupancyPerCU GRBM_GUI_ACTIVE
EOF

rocprofv3 -i counters.txt --output-format csv -d ./out -- ./my_app
```

Counters that exceed the hardware's per-pass multiplexing limit are split across
passes automatically (the kernel is replayed once per `pmc:` line), so the more
groups you request the longer the run.

## Step 3 — the counters that matter, and how to read them

These are the high-signal counters for AI/HPC kernels on CDNA. Most are
percentages of `GRBM_GUI_ACTIVE` (GPU-busy cycles):

- **`VALUUtilization`** — average fraction of the 64 lanes *active* when a VALU
  instruction issues. Low values mean **divergence or predication** (idle lanes),
  not a throughput problem.
- **`VALUBusy`** — fraction of GPU-busy cycles the vector ALU is issuing. High
  VALUBusy + low MFMA = a matrix kernel that fell back to the VALU.
- **`MFMABusy`** (matrix-core busy) — fraction of cycles the matrix unit is
  active. For a GEMM/attention kernel this is the headline number; well-tuned
  CDNA GEMMs push it high. Low MFMABusy on a "GEMM" → the matrix cores are
  starved (LDS or issue bound), see
  [MFMA underutilized](../patterns/mfma-underutilized.md).
- **`LDSBankConflict`** — fraction of LDS cycles lost to
  [bank conflicts](../patterns/bank-conflicts.md). Anything materially above zero
  on a tiled kernel means your LDS layout needs
  [swizzling/padding](lds-swizzling.md).
- **`MemUnitBusy` / `MemUnitStalled`** — the memory subsystem's busy vs stalled
  cycles. High `MemUnitStalled` is the signature of being
  [memory-bound / latency-bound](../patterns/memory-bound.md).
- **`FetchSize` / `WriteSize`** — bytes moved from/to HBM (kB). Divide by kernel
  duration to get achieved **HBM bandwidth**; compare against ~5.3 TB/s on MI300X
  to see how close to the memory roof you are.
- **`MeanOccupancyPerCU`** — achieved resident waves. Compare against the static
  ceiling you predicted (see [occupancy tuning](occupancy-tuning.md)); a gap means
  a VGPR/AGPR/LDS limit or a [scratch spill](../patterns/vgpr-pressure.md).

A counter is only a *symptom*. The job of this page is mapping it to a fix.

## Step 4 — counter → pattern page

| What you observe | Likely diagnosis | Go to |
|---|---|---|
| High `MemUnitStalled`, BW near peak | memory/latency-bound | [pattern-memory-bound](../patterns/memory-bound.md) |
| `MFMABusy` low on a GEMM/attention | matrix cores starved | [pattern-mfma-underutilized](../patterns/mfma-underutilized.md) |
| `LDSBankConflict` > 0 on tiled kernel | LDS bank conflicts | [pattern-bank-conflicts](../patterns/bank-conflicts.md) |
| `MeanOccupancyPerCU` far below ceiling | resource-limited occupancy | [pattern-low-occupancy](../patterns/low-occupancy.md) |
| Occupancy gap + non-zero scratch | register spills | [pattern-vgpr-pressure](../patterns/vgpr-pressure.md) |
| `VALUUtilization` low (lanes idle) | divergence/predication | [hw-wavefront](../hardware/wavefront.md) |

## Step 5 — roofline & deep dive (Omniperf / rocprof-compute)

For the *why*, Omniperf collects a full counter set in one shot and renders a
**roofline** — achieved (FLOP/byte, FLOP/s) plotted against empirical compute and
memory ceilings measured by on-device microbenchmarks. A point sitting on the
slanted memory roof is bandwidth-bound; a point under the flat compute roof has
arithmetic-intensity headroom you are not using.

```bash
# Collect all sections (workload db written under ./workloads/<name>/<arch>/)
rocprof-compute profile --name my_gemm -- ./my_app

# Roofline only (fast): generates a per-kernel roofline PDF
rocprof-compute profile --name my_gemm --roofline -- ./my_app

# Analyze: text report, or open the TUI/Grafana for the LDS/VALU/MFMA panels
rocprof-compute analyze --path ./workloads/my_gemm/MI300X_A1
```

The empirical roofs are architecture-specific: on **MI300X (gfx942)** the FP8
compute roof is ~2× the FP16 roof; on **MI350/MI355X (gfx950)** MXFP6/MXFP4 add
even higher roofs (the MX matrix path). Always read the roofline against the
*dtype the kernel actually issues* — a BF16 kernel measured against the FP8 roof
will look artificially far from the ceiling. The roof your point should be chasing
is the one for its MFMA instruction class.

> **Cross-check static vs dynamic.** Build with
> `hipcc -Rpass-analysis=kernel-resource-usage` to get the *predicted* VGPR/AGPR/
> LDS footprint, then confirm `MeanOccupancyPerCU` matches at runtime. A
> divergence almost always means a scratch spill the static report under-counts.

## Practical notes

- **Warm up first.** Profile a steady-state iteration, not the first launch
  (JIT/codegen and allocator warmup pollute the first dispatch).
- **Minimize multiplexing.** Fewer `pmc:` groups → fewer replays → faster, less
  jittery numbers. Collect timing and counters in *separate* runs.
- **MI300X is 8 XCDs.** Per-CU counters are aggregated across chiplets; a healthy
  average can still hide cross-XCD imbalance — watch occupancy spread, see
  [chiplet/XCD locality](../hardware/chiplet-xcd.md).
- The same workflow applies to Triton kernels — AMD's Triton guide drives exactly
  this loop (find hot kernel → counters → roofline → retune `waves_per_eu`/tile).

## See also

- [Occupancy tuning](occupancy-tuning.md)
- [Wavefront & register files](../hardware/wavefront.md)
- [LDS / shared memory](../hardware/lds.md)
- [Pattern: memory-bound](../patterns/memory-bound.md)
- [Pattern: MFMA underutilized](../patterns/mfma-underutilized.md)

## Sources

- [HIP Performance Guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html)
- [ROCProfiler / rocprofv3 documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/)
- [ROCm Compute Profiler (Omniperf) documentation](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/)
- [GEMM kernel optimization on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores/README.html)
- [Triton kernel performance optimization on AMD](https://rocm.blogs.amd.com/software-tools-optimization/triton-kernel-optimization/README.html)
