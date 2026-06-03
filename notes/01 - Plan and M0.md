# 01 - Plan and M0

## What this is

The asm/SIMD compute layer Saddle doesn't have. Saddle draws trajectories,
Stride is where the speed has to actually hold up. Headline kernel is a fused
Adam update: the naive version makes several passes over the parameter array
(first moment, second moment, bias correction, write), the fused one does it
in a single pass with the rsqrt in registers. Same math, less memory traffic.

## Pieces

Optimisers, tiered so each ships on its own:

- First-order fused: SGD, SGD+momentum, RMSprop, Adam. One shared update
  pattern, Adam is the one that matters.
- Quasi-Newton: L-BFGS, two-loop recursion plus Armijo backtracking. Hot path
  is dot/axpy over the history vectors and the gradient eval.
- Stretch: nonlinear CG, or Gauss-Newton for least squares.

Objectives:

- Analytic: chain Rosenbrock (n-D), Beale, Himmelblau, and an ill-conditioned
  quadratic with condition number kappa as a dial. Exact gradients, not the
  numerical ones Saddle uses.
- Real ML: logistic or ridge regression. The gradient is a gemv plus an axpy,
  which is what turns this into an ML solver instead of a maths toy.

Kernels, roughly in order of value: fused Adam, axpy, dot, scal, gemv, rsqrt
with Newton refinement. Each has a C reference and an asm version checked
against it.

Dials: fp32 first (SIMD width pays off most there), fp64 as the template twin.
AVX2+FMA baseline. Unroll factor and accumulator count are the per-kernel
tuning knobs.

Benchmark axes: dimension n, condition number kappa, dataset size for the ML
objective.

## Benchmark targets, honest

- Fused Adam vs naive multi-pass C: 1.5-3x, real and defensible.
- Fused Adam vs NumPy/PyTorch CPU: "competitive with", not "beats". Can win at
  small-to-mid n where framework dispatch dominates.
- gemv vs OpenBLAS: within 1.5-2x, explain the gap.
- Solver: identical iterates to a reference, faster per-iteration wall clock.
  Overlaid convergence curves is the plot that sells it.

## Hardware (checked 2026-06-02)

Dev machine is an Intel Core Ultra 7 155H (Meteor Lake), not the Zen 4
Framework I'd assumed:

- AVX2, FMA, AVX-VNNI yes. AVX-512 no, Intel fuses it off on the hybrid parts.
- 22 threads across P and E cores. Benchmarks have to pin to a P-core
  (`taskset -c 0`) or the numbers are noise.
- No `vrsqrt14ps`, so the rsqrt path is AVX2's `vrsqrtps` (12-bit) plus a
  Newton step.

## Milestones

- M0 (done): C reference solvers and analytic objectives, validated against
  Saddle. Notes below.
- M1: the checkasm harness. C vs asm, ULP checks, RDTSC counts. The spine.
- M2: fused Adam in AVX2.
- M3: axpy, dot, scal, gemv, then logistic regression on top.
- M4: L-BFGS with line search on the BLAS-1 kernels.
- M5: non-temporal stores plus an unroll/accumulator sweep. NT stores fit the
  story, the fused Adam kernel writes params/m/v once and never rereads them,
  so bypassing the cache is a real win on this machine. AVX-512 is a stretch,
  SDE correctness only since the hardware can't run it.
- M6: Saddle calls the backend and animates real runs.

## M0 decisions

Kernel signatures are the asm ABI from the start. The Adam kernel takes the
bias corrections (bc1 = 1 - beta1^t, bc2 = 1 - beta2^t) precomputed by the
caller, so the hot loop has no pow, no branches, nothing needing libm. M2's
asm drops in behind the same signature without touching the solver.

fp32/fp64 from templates. `*_tmpl.h` included twice with REAL/SQRT/POW/FN
macros, FFmpeg bit-depth style. Public headers declare both variants, no macro
soup leaks out.

Validation against Saddle is bit-exact, not approximate. Two levels:

1. Kernel: `stride_adam_step_f64` vs Saddle's `adam_step`, 1537 random
   elements over 25 steps, memcmp equal.
2. Solver: 500-step Adam on Rosenbrock, Stride's loop driving Saddle's surface
   with Saddle's central differences (h=1e-7), every point identical to
   `adam_optimise`.

It works because both sides do the same float ops in the same order. The
solver recomputes the bias corrections with pow() each step (not
incrementally) precisely to keep this. It also sets up M1: the harness can
demand bit-exactness from asm that doesn't reorder, and fall back to a ULP
tolerance only where reordering is the point (dot accumulators, rsqrt).

Trace layout matches Saddle, entry k is the position after k steps, so the two
trajectory arrays line up directly. M6 leans on that.

Adaptive methods get proximity assertions, not convergence assertions.
RMSprop and Adam with a constant lr stall at a noise floor near the optimum,
so their tests check distance to the minimiser, not gradient norm. SGD and
heavy ball on the quadratic do converge linearly, and that test runs them at
the optimal hyperparameters for the condition number, so "momentum beats SGD"
is guaranteed by the rates rather than luck.

## Next

M1, the harness. Kernel registry (name, C pointer, asm pointer, signature
class), random inputs across edge sizes, bit-exact or max-ULP compare, RDTSC
with serialisation and min-of-many. Asm side starts empty so M2 has a slot to
drop into.
