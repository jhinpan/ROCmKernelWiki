// vadd_asm_gfx942.cpp — GCN inline-assembly vector add for CDNA (gfx942/wave64).
//
// CROSS-COMPILE-ONLY here: this file is built with --offload-arch=gfx942 to
// prove the GCN VMEM asm path assembles. It is NOT run on this gfx1201 (RDNA4)
// box. The inline asm uses FLAT/GLOBAL VMEM instructions and an explicit
// s_waitcnt drain — the same mechanics the wiki page describes for the
// hand-written buffer_load/buffer_store kernel, reduced to a runnable inline
// block so the assembler actually validates the encodings.
//
// Each thread computes C[i] = A[i] + B[i] via:
//   global_load_dword  -> s_waitcnt vmcnt(0) -> v_add_f32 -> global_store_dword
//
// The 64-bit addresses are formed on the host side as raw pointers; the asm
// receives them in VGPR pairs ({al,ah},{bl,bh},{cl,ch}) computed from the base
// pointer + i*4. We let the compiler materialize the per-thread byte offset.

#include <hip/hip_runtime.h>
#include <cstdio>

// Inline-asm vector add. global_* ops on gfx9 take a 64-bit address in a VGPR
// pair and use the VMEM (vmcnt) counter. We drain with s_waitcnt vmcnt(0)
// before consuming the loaded value, exactly as the page's annotated kernel
// gates its pipeline.
__global__ void vadd_asm(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  // Per-thread element addresses (64-bit).
  const float* ap = A + i;
  const float* bp = B + i;
  float* cp = C + i;

  float a, b, c;
  // GLOBAL_LOAD/STORE form: address in a v[lo:hi] pair, off (no SGPR base).
  asm volatile(
      "global_load_dword %0, %3, off\n\t"   // a = *ap
      "global_load_dword %1, %4, off\n\t"   // b = *bp
      "s_waitcnt vmcnt(0)\n\t"               // drain both loads (VMEM counter)
      "v_add_f32 %2, %0, %1\n\t"             // c = a + b
      : "=&v"(a), "=&v"(b), "=v"(c)
      : "v"(ap), "v"(bp));

  asm volatile(
      "global_store_dword %1, %0, off\n\t"   // *cp = c
      "s_waitcnt vmcnt(0)\n\t"
      :
      : "v"(c), "v"(cp)
      : "memory");
}

// Host stub: present so the TU links to a full executable when desired. Not
// executed on gfx1201 — the value of this file is that --offload-arch=gfx942
// assembles the GCN VMEM instructions above.
int main() {
  const int N = 1 << 20;
  float *dA, *dB, *dC;
  if (hipMalloc(&dA, N * sizeof(float)) != hipSuccess) return 0;
  if (hipMalloc(&dB, N * sizeof(float)) != hipSuccess) return 0;
  if (hipMalloc(&dC, N * sizeof(float)) != hipSuccess) return 0;
  hipLaunchKernelGGL(vadd_asm, dim3((N + 255) / 256), dim3(256), 0, 0, dA, dB,
                     dC, N);
  hipDeviceSynchronize();
  std::printf("vadd_asm launched (gfx942 target)\n");
  hipFree(dA);
  hipFree(dB);
  hipFree(dC);
  return 0;
}
