---
id: technique-hip-graphs
title: HIP Graphs — Launch-Overhead Amortization for Decode & Many-Small-Kernel Loops
type: technique
architectures:
- gfx942
- gfx950
tags:
- persistent-kernel
- kernel-fusion
- occupancy-tuning
confidence: source-reported
reproducibility: snippet
hardware_features:
- cu
kernel_types:
- decode
- paged-attention
- mla
- elementwise
languages:
- hip
related:
- kernel-paged-attention
- kernel-mla-decode
- technique-persistent-kernel
- technique-kernel-fusion
sources:
- doc-rocm-hip-hw
- ref-aiter
implemented_by:
- pr-sglang-25898
- pr-aiter-1866
- pr-composable_kernel-1789
- pr-aiter-743
- pr-aiter-729
- pr-sglang-24125
- pr-aiter-591
- pr-aiter-2921
---
# HIP Graphs — Launch-Overhead Amortization for Decode & Many-Small-Kernel Loops

## Overview

A **HIP graph** is a recorded DAG of GPU operations — kernel launches, copies,
memsets, and their dependencies — that is *captured once*, *instantiated once*
into an executable form, and then *replayed* with a single runtime call. The
replay submits the whole sequence to the hardware queue without re-walking the
host-side launch path for each node. This is the AMD/HIP analog of CUDA Graphs
and exists for the same reason: when per-kernel **launch overhead** is a
meaningful fraction of kernel runtime, replacing N `hipLaunchKernel` calls with
one `hipGraphLaunch` removes most of that host cost.

The pain point is LLM **decode**. Autoregressive decode runs one token at a time:
each step is a long chain of *tiny, fixed-shape* kernels (RMSNorm, QKV projection,
RoPE, [paged attention](../kernels/paged-attention.md) /
[MLA decode](../kernels/mla-decode.md), MoE routing, output projection). Batch
sizes are small, so each kernel runs for only a few microseconds — often *less*
than the host-side launch + dependency-resolution cost that precedes it. The CPU
becomes the bottleneck and the GPU sits idle between launches. A graph collapses
the per-step launch storm into one submission, keeping the queue fed.

## Why launch overhead dominates in decode

The default eager path pays, **per kernel**, for: argument marshalling, kernel
descriptor setup, stream dependency bookkeeping, and a doorbell write to the HW
queue. Each is sub-microsecond, but a decode step can issue dozens to hundreds of
kernels, and the GPU work per kernel is tiny — see
[low-occupancy](../patterns/low-occupancy.md) and the latency-bound regime. When
GPU kernel time ≈ host launch time, you are **launch-bound**: adding more CUs does
nothing, and the only lever is reducing per-launch cost.

A graph helps because the cost amortizes across replays:

1. **One submission, not N.** `hipGraphLaunch` enqueues the entire captured DAG;
   the host does not return to the launch path between nodes.
2. **Dependencies are pre-resolved.** Edge/ordering analysis happens at
   *instantiate* time, not on every step. Replay just executes the baked schedule.
3. **Stable, repeated shape.** Decode replays the *same* graph every token, so the
   one-time capture/instantiate cost is amortized over thousands of steps.

It does **not** make any individual kernel faster — graphs attack host overhead
and inter-kernel gaps, not kernel compute. If your kernels are already
compute-bound and back-to-back, a graph buys little.

## The three phases: capture → instantiate → launch

The most ergonomic way to build a graph is **stream capture**: run your normal
HIP code once against a stream put into capture mode; the runtime records the
operations instead of executing them.

```cpp
#include <hip/hip_runtime.h>

hipStream_t     stream;
hipGraph_t      graph;
hipGraphExec_t  exec;
hipStreamCreate(&stream);

// ---- Phase 1: capture the decode step once (records, does not run) ----
hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal);

rmsnorm<<<g1, b1, 0, stream>>>(/* ... */);            // fixed shapes
qkv_proj<<<g2, b2, 0, stream>>>(/* ... */);
paged_attention<<<g3, b3, 0, stream>>>(/* ... */);    // KV addresses must be stable
out_proj<<<g4, b4, 0, stream>>>(/* ... */);

hipStreamEndCapture(stream, &graph);

// ---- Phase 2: instantiate (resolve deps, bake an executable graph) ----
hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

// ---- Phase 3: replay every decode token with ONE launch ----
for (int step = 0; step < num_tokens; ++step) {
    hipGraphLaunch(exec, stream);                     // one host call, whole DAG
    hipStreamSynchronize(stream);                     // or pipeline across steps
}

hipGraphExecDestroy(exec);
hipGraphDestroy(graph);
hipStreamDestroy(stream);
```

You can also build a graph **explicitly** with `hipGraphAddKernelNode` /
`hipGraphAddDependencies` when you want full control of the DAG, but for an
existing eager code path stream capture is the lowest-friction route.

## When it helps

- **Decode / autoregressive loops** — many tiny kernels per step, replayed for
  every token. The canonical win.
- **Fixed shapes across iterations.** The graph topology and launch dimensions are
  baked at instantiate time; identical replays are what amortize the setup.
- **Deep static chains.** Norm → proj → attention → MoE → proj pipelines where the
  inter-kernel host gap rivals the kernel runtime.
- **As a complement to [kernel fusion](kernel-fusion.md) and
  [persistent kernels](persistent-kernel.md).** Fuse what you can to cut node
  count; a graph then amortizes the launches that remain. ROCm's
  [AITER](../../sources/refs/ref-aiter.md) inference kernels are designed to drop
  into exactly this graph-replayed decode loop.

## Pitfalls

- **Dynamic shapes break replay.** A captured graph hard-codes grid/block dims and
  buffer sizes. If sequence length, batch, or token count changes per step, the
  graph is stale. Either pad to a fixed shape (capture for the max and mask), or
  keep a small set of graphs keyed by shape and pick one per step.
- **Memory addresses are baked in.** Captured kernel-node arguments embed the
  *pointer values* present at capture time. If you reallocate activation buffers,
  grow a paged-attention **KV cache**, or change a workspace base, the graph
  replays against stale addresses → wrong results or faults. Use a persistent
  arena allocated once before capture, or `hipGraphExecKernelNodeSetParams` to
  patch node arguments without re-instantiating.
- **Host-side control flow vanishes.** Branches, CPU-side `if`/early-exit, and
  data-dependent loop bounds evaluated on the host are *not* recorded — only the
  GPU ops issued during the capture pass are. Sampling/stopping logic stays on the
  host outside the graph.
- **Capture hygiene.** Unsupported operations (some blocking syncs, certain
  library calls) can invalidate a capture; check the returned status and confirm
  the library you call is capture-safe. Use a non-default stream for capture so you
  do not record unrelated work.
- **Re-instantiate cost.** `hipGraphInstantiate` is not free; doing it per step
  defeats the purpose. Instantiate once, then prefer
  `hipGraphExecUpdate` / `hipGraphExec*NodeSetParams` to refresh arguments.

## Performance notes

- The win scales with **node count × per-launch overhead ÷ GPU work**. Profile the
  eager loop first: if there are visible host-side gaps between kernels on the
  timeline (e.g. in `rocprofv3` traces), graphs will close them; if the timeline is
  already dense with compute, they will not.
- Reducing node count *before* graphing (via [fusion](kernel-fusion.md)) compounds
  with graphs — fewer nodes means less capture/instantiate cost and a tighter
  replay.
- Treat graph capture as a correctness contract on shapes and pointers: pin the
  allocator, fix the shapes, and validate replay output against the eager path once
  before trusting it in the decode loop.

## See also

- [Persistent kernels](persistent-kernel.md) — the in-kernel counterpart to
  amortizing launch cost.
- [Kernel fusion](kernel-fusion.md) — cut node count before graphing.
- [Paged attention](../kernels/paged-attention.md) and
  [MLA decode](../kernels/mla-decode.md) — the decode kernels that live inside the
  graph.

## Sources

- [ROCm HIP — Hardware Implementation](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html) — see `doc-rocm-hip-hw`
- [HIP Runtime API — Graph Management](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___graph.html)
- [HIP graph usage (how-to)](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hipgraph.html)
- [AITER — AI Tensor Engine for ROCm](https://github.com/ROCm/aiter) — see `ref-aiter`
