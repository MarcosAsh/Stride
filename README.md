# Stride

SIMD-accelerated numerical optimisers in C and x86 assembly.

Stride is the compute backend for [Saddle](../Saddle). Saddle draws optimiser
trajectories on loss surfaces; Stride is the part that makes the arithmetic
fast. The core of it is a set of fused optimiser update kernels in AVX2
assembly, each checked against a C reference by a checkasm-style harness.

## Why fuse the update

A naive Adam step walks the parameter arrays several times: once to update the
first moment, once for the second, once for the bias-corrected estimates, once
for the final write. Once the arrays are bigger than cache, every one of those
passes is a round trip to memory. The fused kernel does the whole step in one
pass with the reciprocal-sqrt kept in registers. Same arithmetic, far less
memory traffic.

## Performance, measured

On the dev machine (Core Ultra 7 155H, pinned to one P-core), the fused AVX2
Adam kernel runs bit-for-bit identical to the C reference and:

- 4.5x faster than scalar C in fp32
- 2.1x faster in fp64 (half the lanes, so roughly half the win)

The slower-is-real direction matters too. A multi-pass Adam, the shape a naive
NumPy-style implementation lands on, drops to 0.64x of the single-pass C once
the data spills to DRAM. That gap is the memory-traffic argument showing up in
the numbers.

Targets I'm holding myself to for the comparisons that aren't done yet:

- gemv within 1.5-2x of OpenBLAS, with the gap explained rather than hidden
- competitive with NumPy/PyTorch CPU Adam, not "beats", though small-to-mid
  sizes can win on dispatch overhead
- solver iterates identical to a reference, faster wall clock per step

## Milestones

- [x] **M0** C reference solvers (SGD, momentum, RMSprop, Adam) and analytic
  objectives with exact gradients, checked bit-for-bit against Saddle's C Adam.
- [x] **M1** checkasm-style harness: C vs asm, ULP checks, RDTSC timing, P-core
  pinning, guarded buffers, and a self-test that proves it catches broken kernels.
- [x] **M2** fused Adam in AVX2, bit-exact, 4.5x (fp32) / 2.1x (fp64) over scalar C.
- [x] **M3** BLAS-1 in AVX2 (axpy, dot, scal), gemv on top, and a logistic
  regression objective that trains through the dispatched kernels. dot hits
  8-15x in cache; the harness checks it against a high-precision truth since
  the lane-split sum is not bit-exact.
- [ ] **M4** L-BFGS with two-loop recursion and Armijo backtracking.
- [ ] **M5** non-temporal stores and an unroll/accumulator sweep; AVX-512 paths
  checked under Intel SDE as a stretch goal.
- [ ] **M6** Saddle bridge: animate real Stride solver runs.

## Layout

```
include/stride/   public API: kernels, objectives, solvers, cpu detection
src/kernels/      update kernels, C reference plus the x86 asm under x86/
src/objectives/   analytic test functions with hand-written gradients
src/solvers/      iteration loops that drive the kernels
tests/            gradient checks, convergence tests, Saddle cross-check
tests/checkasm/   the kernel harness
```

The fp32 and fp64 variants come from one template (`*_tmpl.h`) included twice
with different `REAL`/`FN` macros, the same trick FFmpeg uses for bit-depth
variants.

## Building

```
make               # build/libstride.a
make test          # gradient + solver tests, plus the checkasm self-test
make checkasm      # kernel correctness, C vs asm
make bench         # kernel benchmarks
make check-saddle  # bit-exact comparison against ../Saddle's C Adam
```

## Hardware

Built on an Intel Core Ultra 7 155H (Meteor Lake): AVX2 and FMA, no AVX-512.
Benchmarks pin to a P-core with `taskset -c 0` so the hybrid scheduler stays
out of the timings. Any AVX-512 code is correctness-tested under Intel SDE
only and carries no performance claim.
