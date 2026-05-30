#!/usr/bin/env python3
"""Generate source anchor pages (docs, blogs, reference-repo studies) from
the structured catalog below. These are the citable anchors that wiki
synthesis pages reference via their `sources:` field.

Run: python3 scripts/gen_source_anchors.py
"""
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT  # noqa: E402

CUTOFF = "2026-05-15"

# ---- DOCS (official-doc / paper) -------------------------------------------
DOCS = [
    dict(id="doc-cdna3-isa", title='AMD Instinct MI300 (CDNA3) Instruction Set Architecture Reference Guide',
         url="https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf",
         source_category="official-doc", architectures=["gfx942"],
         tags=["mfma", "lds", "buffer-instructions", "s-waitcnt", "wave64", "cdna"],
         body="""The authoritative ISA reference for CDNA3 (gfx942 / MI300 family), revision
5-August-2025. Documents the VOP3P-MAI MFMA encoding, the LDS (64 kB/CU, 32
banks of 512 Dwords, 32-bit wide), MUBUF/MTBUF buffer instructions with their
128-bit resource descriptor (V#) and out-of-bounds semantics, FLAT/GLOBAL/SCRATCH
addressing, the `s_waitcnt` counters (VMCNT 6-bit, LGKMCNT 4-bit, EXPCNT 3-bit),
the direct-to-LDS load path, and cross-lane primitives (`ds_swizzle`,
`ds_permute`/`ds_bpermute`, DPP modifiers).

Key facts used across this wiki:

- LDS: *"64 kB memory per compute unit, segmented into 32 banks of 512 Dwords,
  each bank being 32 bits wide."* Bank conflicts serialize; an access can take
  2–64 cycles depending on conflicts.
- Buffer OOB: *"Reads that go out-of-range return zero ... Writes that are
  out-of-range do not write anything."* — the basis for branchless boundary
  guards on AMD.
- FP8 on CDNA3 is the **FNUZ** (OCP-incompatible) encoding — numerically
  distinct from the OCP FP8 introduced on CDNA4."""),

    dict(id="doc-cdna4-isa", title='CDNA4 Instruction Set Architecture Reference Guide',
         url="https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf",
         source_category="official-doc", architectures=["gfx950"],
         tags=["mfma", "fp8", "fp6", "fp4", "mxfp", "lds", "block-scale", "cdna"],
         body="""The authoritative ISA reference for CDNA4 (gfx950 / MI350-MI355X),
revision 5-August-2025. Adds the unified low-precision matrix instructions
`v_mfma_f32_16x16x128_f8f6f4` and `v_mfma_f32_32x32x64_f8f6f4`, plus their
microscaling (MX) `v_mfma_scale_*` variants.

Key facts used across this wiki:

- The `f8f6f4` ops **repurpose CBSZ/BLGP** as per-matrix element-format selectors:
  `000`=E4M3 (FP8), `001`=E5M2 (BF8), `010`=E2M3 (FP6), `011`=E3M2 (BF6),
  `100`=E2M1 (FP4). Mixed A/B formats are allowed.
- MX scale format is **E8M0**; `ABID[0]=1` enables scaling (else all scales 1.0).
  The hardware folds scales into the exponent sum.
- LDS grows to **160 kB/CU, 64 banks of 640 Dwords**.
- Direct-to-LDS widens to 12/16-byte copies (`GLOBAL_LOAD_LDS_DWORDX3/X4`).
- `v_permlane16_*` cross-lane ops are added (absent on gfx942).
- EXPCNT is "Unused"; TF32/XF32 native path dropped in favor of MX formats."""),

    dict(id="doc-cdna3-whitepaper", title='AMD CDNA 3 Architecture White Paper',
         url="https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf",
         source_category="official-doc", architectures=["gfx942"],
         tags=["xcd", "l2-cache", "infinity-cache", "hbm3", "cu", "cdna"],
         body="""Architecture white paper for CDNA3 / MI300. Describes the chiplet
design: each XCD (Accelerator Complex Die) has 40 physical CUs (38 active),
sharing a 4 MB L2; up to 8 XCDs → 304 CUs on MI300X. A 256 MB Infinity Cache
(memory-side LLC, 16-way) sits on the IO dies; MI300X ships 192 GB HBM3 at
5.3 TB/s. L2 coherence is per-XCD, making the XCD an effective NUMA domain."""),

    dict(id="doc-cdna4-whitepaper", title='AMD CDNA 4 Architecture White Paper',
         url="https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf",
         source_category="official-doc", architectures=["gfx950"],
         tags=["xcd", "fp4", "fp6", "mxfp", "infinity-cache", "hbm3", "cdna"],
         body="""Architecture white paper for CDNA4 / MI350-MI355X. Each XCD has 36
physical CUs (32 active) on TSMC N3P, 4 MB L2; up to 8 XCDs → 256 CUs. MI355X
ships 288 GB HBM3E at up to 8 TB/s. Table 1 peak matrix throughput: FP16/BF16
2.5 PF, OCP-FP8 5.0 PF, INT8 5.0 POPS, MXFP6/MXFP4 10 PF (dense). FP8 is the
OCP encoding (labeled "OCP-FP8"), distinct from CDNA3's FNUZ FP8."""),

    dict(id="doc-mi300x-datasheet", title='AMD Instinct MI300X Data Sheet',
         url="https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf",
         source_category="official-doc", architectures=["gfx942"],
         tags=["hbm3", "fp8", "bf16", "fp16", "int8", "matrix-core"],
         body="""Official MI300X peak-throughput figures (dense | with-sparsity):
TF32 653.7 | 1307.4 TFLOPS; FP16 1307.4 | 2614.9 TFLOPS; BF16 1307.4 | 2614.9
TFLOPS; FP8 2614.9 | 5229.8 TFLOPS; INT8 2614.9 | 5229.8 TOPS; FP64/FP32 matrix
163.4 TFLOPS. Memory: 192 GB HBM3, 5.3 TB/s."""),

    dict(id="doc-rocm-hip-hw", title='ROCm HIP — Hardware Implementation',
         url="https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html",
         source_category="official-doc", architectures=["gfx942", "gfx950"],
         tags=["wave64", "sgpr", "vgpr", "agpr", "cu", "cdna"],
         body="""ROCm HIP hardware-implementation reference. Documents the CU register
files: ~12.5 KiB SGPR storage per CU; 256–512 KiB VGPR storage split across the
four SIMD16 units; up to 512 total VGPRs per wave (256 Arch + 256 Acc). Up to
40 waves/CU occupancy (four pools × 10 waves), typically limited by register and
LDS usage. CDNA is wave64-only."""),

    dict(id="doc-llvm-amdgpu", title='LLVM — User Guide for the AMDGPU Backend',
         url="https://llvm.org/docs/AMDGPUUsage.html",
         source_category="official-doc", architectures=["gfx942", "gfx950", "gfx1201"],
         tags=["mfma", "rocdl", "async-copy", "block-scale"],
         body="""The LLVM AMDGPU backend user guide: target names (gfx942, gfx950,
gfx1201), the `llvm.amdgcn.mfma.*` and `llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4`
intrinsics, and `llvm.amdgcn.load.to.lds` (lowers to `global_load_lds` /
`buffer_load_*_lds`; gfx950 allows 12/16-byte copies). Authoritative reference for
the intrinsics ROCm/HIP/Triton ultimately emit."""),

    dict(id="doc-flash-attention-2", title='FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning',
         url="https://arxiv.org/abs/2307.08691",
         source_category="paper", architectures=["gfx942", "gfx950"],
         tags=["flash-attention", "attention", "software-pipelining"],
         body="""The FlashAttention-2 paper (Tri Dao, 2023). The tiling/online-softmax
algorithm that ROCm's CK-tile and Triton FMHA kernels implement on CDNA. Used here
as the algorithmic reference for the attention kernel pages; the AMD-specific work
(MFMA layout, LDS double-buffering, direct-to-LDS) is layered on top."""),
]

# ---- BLOGS (benchmark-blog / community-note) -------------------------------
BLOGS = [
    dict(id="blog-amd-matrix-cores", title='AMD Matrix Cores', author="AMD ROCm",
         url="https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html",
         source_category="benchmark-blog", architectures=["gfx942"],
         tags=["mfma", "matrix-core", "bf16", "fp16"],
         body="""The foundational ROCm blog on AMD Matrix Cores. Introduces the
`__builtin_amdgcn_mfma_*` intrinsics, the per-wavefront register-fragment layout
for A/B/C/D operands, and the `v_mfma_f32_16x16x16f16` / `v_mfma_f32_32x32x8f16`
shapes. The starting point for understanding MFMA programming."""),

    dict(id="blog-matrix-cores-cdna", title='Matrix Core Programming on AMD CDNA3 and CDNA4 Architecture', author="AMD ROCm",
         url="https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html",
         source_category="benchmark-blog", architectures=["gfx942", "gfx950"],
         tags=["mfma", "fp8", "fp6", "fp4", "mxfp", "block-scale"],
         body="""The definitive intrinsic-level reference for MFMA on gfx942 and
gfx950. Covers the `__builtin_amdgcn_mfma_f32_*` builtins, the CDNA4 `f8f6f4`
unified low-precision ops, scaled MFMA with E8M0 block scales, and the FNUZ-vs-OCP
FP8 distinction between the two architectures."""),

    dict(id="blog-gemm-optimization", title='GEMM Kernel Optimization for AMD GPUs', author="AMD ROCm",
         url="https://rocm.blogs.amd.com/artificial-intelligence/gemm_blog/README.html",
         source_category="benchmark-blog", architectures=["gfx942"],
         tags=["gemm", "mfma", "lds-double-buffering", "swizzle", "tile-scheduling"],
         body="""End-to-end walkthrough of GEMM optimization on MI300: macro-tiling,
LDS staging with double buffering, MFMA scheduling, bank-conflict avoidance via
swizzled LDS layouts, and work-group → L2 tile mapping for locality."""),

    dict(id="blog-fp8-gemm-cdna4", title='FP8 GEMM Optimization on AMD CDNA4 Architecture', author="AMD ROCm",
         url="https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html",
         source_category="benchmark-blog", architectures=["gfx950"],
         tags=["fp8-gemm", "fp8", "block-scale", "mfma", "async-copy"],
         body="""FP8 GEMM tuning specifically for gfx950: using the OCP-FP8 `f8f6f4`
MFMA path, E8M0 block scaling, direct-to-LDS async copy (now 16-byte wide), and
the larger 160 kB LDS to deepen the software pipeline."""),

    dict(id="blog-4wave-fp8-gemm", title='Deep Dive Into 4-Wave Interleave FP8 GEMM', author="AMD ROCm",
         url="https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html",
         source_category="benchmark-blog", architectures=["gfx950"],
         tags=["fp8-gemm", "mfma-pipelining", "occupancy-tuning", "wave-specialization"],
         body="""A detailed case study of a 4-wave interleaved FP8 GEMM on CDNA4,
overlapping MFMA issue across waves to hide latency and saturate the matrix cores."""),

    dict(id="blog-cktile-gemm", title='Hands-On with CK-Tile: Develop and Run Optimized GEMM on AMD GPUs', author="AMD ROCm",
         url="https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html",
         source_category="benchmark-blog", architectures=["gfx942"],
         tags=["gemm", "mfma", "lds-double-buffering"],
         body="""Tutorial on building a GEMM with the CK-tile DSL: tile descriptors,
the warp/block/pipeline/kernel operator tiers, distributed tensors, and
`load_tile`/`store_tile`/`shuffle_tile` tile-level APIs."""),

    dict(id="blog-cktile-flash", title='From Theory to Kernel: Implement FlashAttention-v2 with CK-Tile', author="AMD ROCm",
         url="https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html",
         source_category="benchmark-blog", architectures=["gfx942"],
         tags=["flash-attention", "attention", "mfma", "lds-double-buffering"],
         body="""Implements FlashAttention-2 with CK-tile: online-softmax tiling,
back-to-back MFMA GEMMs (QK^T then PV), LDS staging, and the block/pipeline
operator structure for a fused attention kernel on MI300."""),

    dict(id="blog-triton-amd", title='Developing Triton Kernels on AMD GPUs', author="Clint Greene (AMD ROCm)",
         url="https://rocm.blogs.amd.com/artificial-intelligence/triton/README.html",
         source_category="benchmark-blog", architectures=["gfx942"],
         tags=["gemm", "attention"],
         body="""Intro to writing Triton kernels for ROCm: how `tl.dot` maps onto
MFMA, wave size 64 on CDNA, and getting started with the AMD Triton backend."""),

    dict(id="blog-triton-optimizations", title='Unlock Peak Performance on AMD GPUs with Triton Kernel Optimizations', author="AMD ROCm",
         url="https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html",
         source_category="benchmark-blog", architectures=["gfx942", "gfx950"],
         tags=["mfma-pipelining", "occupancy-tuning", "async-copy"],
         body="""The key Triton-on-AMD tuning knobs: `matrix_instr_nonkdim` (MFMA
size selection), `waves_per_eu` (occupancy), `kpack` (K-packing; deprecated on
gfx950), `num_stages` (pipeline depth), and async-copy / buffer-ops passes."""),

    dict(id="blog-gluon-gemm", title='From Naive to Near-Peak: Building High-Performance GEMM Kernels with Gluon', author="AMD ROCm",
         url="https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html",
         source_category="benchmark-blog", architectures=["gfx950"],
         tags=["gemm", "mfma-pipelining", "async-copy", "lds-double-buffering"],
         body="""A step-by-step GEMM optimization in Gluon (Triton's lower-level
layer), progressing from a naive tiling to a near-peak kernel by adding LDS
buffering, async copy, and MFMA scheduling on CDNA4."""),

    dict(id="blog-flash-attention-amd", title='Accelerating Large Language Models with Flash Attention on AMD GPUs', author="AMD ROCm",
         url="https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html",
         source_category="benchmark-blog", architectures=["gfx942"],
         tags=["flash-attention", "attention", "kv-cache"],
         body="""Overview of FlashAttention on AMD GPUs: the algorithm, the ROCm
flash-attention (CK) and Triton paths, and how to use them from PyTorch."""),

    dict(id="blog-hipblaslt-tuning", title='hipBLASLt Offline Tuning', author="AMD ROCm",
         url="https://rocm.blogs.amd.com/software-tools-optimization/hipblaslt-offline-tuning-part1/README.html",
         source_category="benchmark-blog", architectures=["gfx942", "gfx950"],
         tags=["gemm", "fp8-gemm", "tile-scheduling"],
         body="""How hipBLASLt selects and tunes GEMM solutions offline, picking
among TensileLite-generated assembly kernels for a given problem size and dtype."""),
]

# ---- REFERENCE REPOS (reference-repo) --------------------------------------
REFS = [
    dict(id="ref-flydsl", title='FlyDSL — Flexible Layout DSL for AMD GPUs', repo="ROCm/FlyDSL",
         url="https://github.com/ROCm/FlyDSL", author="ROCm",
         source_category="reference-repo", architectures=["gfx942", "gfx950", "gfx1201"],
         tags=["flydsl", "mfma", "gemm", "preshuffle-layout"], languages=["flydsl", "mlir", "python"],
         body="""FlyDSL is a Python DSL plus an MLIR-native compiler stack for
authoring AMD GPU kernels with explicit layouts and tiling. The `fly` dialect is
a layout IR (`!fly.int_tuple`, `!fly.layout`, `!fly.coord_tensor`, `!fly.memref`)
with CuTe-like layout algebra: a Layout is a (Shape, Stride) pair mapping a
coordinate to a linear index. Kernels use `@flyc.kernel` / `@flyc.jit`; the JIT
traces Python to MLIR and lowers Fly → ROCDL → LLVM → fatbin. Tiling is explicit
across block/warp/thread/instruction scopes with MFMA atoms. Verified targets:
MI300X/MI308X (gfx942), MI350/MI355X (gfx950), MI450 (gfx1250), Radeon AI PRO
R9700 (gfx1201). Apache-2.0. Examples: `01-vectorAdd.py`, `02-tiledCopy.py`,
`03-tiledMma.py`, `04-preshuffle_gemm.py`. Note: experimental, not part of the
official ROCm distribution."""),

    dict(id="ref-gcnasm", title='gcnasm — GCN Assembly & HIP Programming Examples', repo="carlushuang/gcnasm",
         url="https://github.com/carlushuang/gcnasm", author="carlushuang",
         source_category="reference-repo", architectures=["gfx942", "gfx950"],
         tags=["gcn-asm", "mfma", "bandwidth-bench", "dpp", "async-copy"], languages=["gcn-asm", "hip"],
         body="""A collection of AMD GPU programming examples (CDNA/RDNA, primarily
gfx942/MI300) covering hand-written GCN assembly kernels, HIP device code, and
PyTorch/Triton extensions. Standout examples: `bandwidth_memread` (float4
non-temporal persistent bandwidth microbench, ~4.56 TB/s on MI308X);
`vector_add_asm` (persistent kernel, `buffer_load_dword ... offen lds` async load
to LDS, double LDS buffering, OOB-based control flow, `vmcnt(3)` pipelining);
`matrix_core` / `matrix_core_gfx950` (MFMA demos); `hgemm` (128×128 MFMA asm);
`wave_reduce_dpp`, `ds_permute`, `transpose-lds`. An excellent low-level
MFMA/assembly reference."""),

    dict(id="ref-composable-kernel", title='Composable Kernel (CK / CK-tile)', repo="ROCm/composable_kernel",
         url="https://github.com/ROCm/composable_kernel", author="ROCm",
         source_category="reference-repo", architectures=["gfx942", "gfx950"],
         tags=["composable-kernel", "gemm", "flash-attention", "fused-moe"], languages=["composable-kernel", "hip", "cpp"],
         body="""Composable Kernel is a HIP C++ tile-based programming model for
performance-critical ML kernels, built on tensor coordinate-transformation. The
newer `ck_tile` DSL is self-contained (single-header components like
`ck_tile/core.hpp`, `ck_tile/ops/fmha.hpp`) and organizes operators by execution
level: warp → block → pipeline → kernel. Core abstractions: tensor descriptors,
distributed tensors (storage + thread distribution), and tile APIs `load_tile`,
`store_tile`, `shuffle_tile`, `slice_tile`. CK is the primary kernel backend for
many ROCm ML ops. MIT. (Active development has moved to ROCm/rocm-libraries.)"""),

    dict(id="ref-rocwmma", title='rocWMMA — C++ WMMA-style MMA library', repo="ROCm/rocWMMA",
         url="https://github.com/ROCm/rocWMMA", author="ROCm",
         source_category="reference-repo", architectures=["gfx942", "gfx950", "gfx1201"],
         tags=["matrix-core", "mfma", "wmma", "gemm"], languages=["cpp", "hip"],
         body="""rocWMMA is a header-only C++ library for mixed-precision MMA. It
exposes a CUDA-`wmma`-like fragment API (load → mma_sync → store) that compiles
directly into `v_mfma_*` (CDNA) or `v_wmma_*` (RDNA) instructions, handling the
wavefront register-fragment layout for you. Supports gfx908/90a/942/950 (CDNA)
and gfx1100/1201 (RDNA). The recommended path over raw `__builtin_amdgcn_mfma_*`
when you want portable MMA. MIT."""),

    dict(id="ref-aiter", title='AITER — AI Tensor Engine for ROCm', repo="ROCm/aiter",
         url="https://github.com/ROCm/aiter", author="ROCm",
         source_category="reference-repo", architectures=["gfx942", "gfx950"],
         tags=["fused-moe", "attention", "mla", "paged-attention", "quantization"], languages=["hip", "cpp", "triton"],
         body="""AITER is AMD's high-performance AI operator library — the default
kernel backend for LLM inference on AMD GPUs (e.g. vLLM's default attention
backend). It offers C++ and Python APIs and dispatches across multiple kernel
backends: Triton, Composable Kernel, and hand-tuned assembly (with optional
FlyDSL kernels for mixed-precision MoE, falling back to CK when absent). Coverage:
attention (MHA, MLA, Paged Attention), fused MoE, GEMM, normalization,
quantization, and fused GEMM+communication."""),

    dict(id="ref-hipblaslt", title='hipBLASLt — GEMM with epilogue fusion', repo="ROCm/hipBLASLt",
         url="https://github.com/ROCm/hipBLASLt", author="ROCm",
         source_category="reference-repo", architectures=["gfx942", "gfx950"],
         tags=["gemm", "fp8-gemm", "epilogue-fusion"], languages=["hip", "cpp", "gcn-asm"],
         body="""hipBLASLt is AMD's lightweight GEMM library (cuBLASLt-style API,
`hipblasLtMatmul`) computing D = Activation(alpha·op(A)·op(B) + beta·op(C) + bias)
with GELU/ReLU/Swish epilogues and bias fusion. FP8 support distinguishes FNUZ
(gfx942) from OCP (gfx950) types. Its kernel generator backend is TensileLite,
which emits AMDGPU assembly GEMM kernels selected per problem size/dtype."""),

    dict(id="ref-tensile", title='Tensile — assembly GEMM kernel generator', repo="ROCm/Tensile",
         url="https://github.com/ROCm/Tensile", author="ROCm",
         source_category="reference-repo", architectures=["gfx942", "gfx950"],
         tags=["gemm", "gcn-asm", "tile-scheduling", "split-k"], languages=["gcn-asm", "python"],
         body="""Tensile is a Python tool that generates benchmark-driven GEMM (and
tensor-contraction) backend libraries, mainly for rocBLAS. Its `KernelLanguage`
parameter chooses HIP or assembly output; the `MatrixInstruction` parameter
encodes the MFMA shape and wave-tiling ([M,N,K,B, WaveTileM/N, WaveGroupM/N]).
Solution selection runs a four-level catalog (hardware → operation → problem →
exact solution) with performance-ranked kernels."""),

    dict(id="ref-matrix-calculator", title='AMD Matrix Instruction Calculator', repo="ROCm/amd_matrix_instruction_calculator",
         url="https://github.com/ROCm/amd_matrix_instruction_calculator", author="ROCm",
         source_category="reference-repo", architectures=["gfx942", "gfx950"],
         tags=["mfma", "matrix-core", "agpr"], languages=["python"],
         body="""An official tool that, given an MFMA/WMMA instruction, reports its
shape, supported dtypes, register usage (Arch vs Acc VGPRs), operand
element→register mapping, FLOP counts, and per-CU throughput. Indispensable for
deriving exact MFMA operand layouts instead of hand-computing them. Supports
`--list-instructions`, `--detail-instruction`, `--get-register`, `--matrix-entry`."""),
]


def write_page(subdir, fm, body):
    out = WIKI_ROOT / "sources" / subdir
    out.mkdir(parents=True, exist_ok=True)
    yaml_fm = yaml.dump(fm, sort_keys=False, allow_unicode=True, default_flow_style=False)
    title = fm["title"]
    md = f"---\n{yaml_fm}---\n\n# {title}\n\n{body}\n\n## Reference\n\n- Upstream: <{fm['url']}>\n"
    (out / f"{fm['id']}.md").write_text(md, encoding="utf-8")


def main():
    n = 0
    for d in DOCS:
        body = d.pop("body")
        fm = dict(id=d["id"], title=d["title"], url=d["url"],
                  source_category=d["source_category"],
                  architectures=d["architectures"], tags=d["tags"],
                  retrieved_at=CUTOFF)
        if "author" in d:
            fm["author"] = d["author"]
        write_page("docs", fm, body); n += 1
    for d in BLOGS:
        body = d.pop("body")
        fm = dict(id=d["id"], title=d["title"], author=d["author"], url=d["url"],
                  source_category=d["source_category"],
                  architectures=d["architectures"], tags=d["tags"],
                  retrieved_at=CUTOFF)
        write_page("blogs", fm, body); n += 1
    for d in REFS:
        body = d.pop("body")
        fm = dict(id=d["id"], title=d["title"], repo=d["repo"], url=d["url"],
                  author=d["author"], source_category=d["source_category"],
                  architectures=d["architectures"], tags=d["tags"],
                  languages=d.get("languages", []), retrieved_at=CUTOFF)
        write_page("refs", fm, body); n += 1
    print(f"wrote {n} source anchor pages")


if __name__ == "__main__":
    main()
