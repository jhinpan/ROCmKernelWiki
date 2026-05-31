#!/usr/bin/env python3
# 04_preshuffle_gemm_flydsl.py
#
# FAITHFUL FlyDSL REFERENCE SNIPPET -- DOES NOT RUN HERE.
# FlyDSL is NOT installed on this gfx1201 box (no `import flydsl`), and FlyDSL
# targets CDNA MFMA (gfx942/gfx950). This file is illustrative of the *pattern*
# only; the runnable demonstration of the same preshuffle idea is the portable
# rocWMMA program preshuffle_gemm_rocwmma.cpp in this directory.
#
# FlyDSL is an Apache-2.0 research layout DSL whose `fly` dialect/API is still
# moving -- check https://github.com/ROCm/FlyDSL for the current spelling.

import flydsl as fly                 # not installed here -> ImportError if run
from flydsl import flyc
from flydsl.lang import mfma, copy, alloc_lds, cta_id, sync

# MFMA atom for bf16 -> f32 on CDNA3: 16x16x16 (M, N, K)
MFMA_M, MFMA_N, MFMA_K = 16, 16, 16


# ---------------------------------------------------------------------------
# The layout transform: B (K, N) row-major -> fragment-contiguous order.
# Expressed as an algebraic layout composition rather than index arithmetic.
# ---------------------------------------------------------------------------
def preshuffle_B(shape_KN):
    K, N = shape_KN
    src = fly.Layout(shape=(K, N), stride=(N, 1))      # row-major B
    tiled = src.tile((MFMA_K, MFMA_N))                 # ((Kt,Nt),(MFMA_K,MFMA_N))
    shuffled = tiled.permute((0, 1, 3, 2))             # fragment-major inner layout
    return shuffled.coalesce()                         # contiguous 16x16 fragments


# ---------------------------------------------------------------------------
# Steady-state kernel: B is already preshuffled, so its stage is a flat copy.
# ---------------------------------------------------------------------------
BM, BN, BK = 128, 128, 32           # CTA macro-tile
WM, WN = 64, 64                     # per-wave tile (wave64)


@flyc.kernel(arch="gfx942", waves_per_block=4)
def preshuffle_gemm(A: fly.Tensor("bf16"),
                    Bsh: fly.Tensor("bf16"),    # already preshuffled
                    C: fly.Tensor("f32"),
                    M: int, N: int, K: int):
    bm, bn = cta_id(0), cta_id(1)

    sA = alloc_lds("bf16", (BM, BK), buffers=2)
    sB = alloc_lds("bf16", (BK, BN), buffers=2)

    acc = fly.frag("f32", (WM, WN), init=0.0)

    for k0 in fly.pipelined(range(0, K, BK), stages=2):
        # A still needs a (small) swizzled stage; B is a flat copy because it
        # was preshuffled into fragment order on the host.
        copy(A.tile(bm, k0, (BM, BK)),   sA.next(), swizzle="mfma_a")
        copy(Bsh.tile(k0, bn, (BK, BN)), sB.next())     # <-- contiguous, no swizzle
        sync()

        for kk in range(0, BK, MFMA_K):
            a = sA.cur().load_frag(kk, atom="16x16x16")
            b = sB.cur().load_frag(kk, atom="16x16x16")
            acc = mfma(a, b, acc, shape="16x16x16", abfmt="bf16")

    C.tile(bm, bn, (BM, BN)).store(acc)


if __name__ == "__main__":
    raise SystemExit(
        "Reference only: FlyDSL is not installed and targets CDNA MFMA. "
        "Run ./build.sh to build+run the portable rocWMMA demonstration instead."
    )
