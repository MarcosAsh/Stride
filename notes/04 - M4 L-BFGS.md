# 04 - M4 L-BFGS

Quasi-Newton, the first solver that does more than scale the gradient. L-BFGS
builds a limited-memory inverse-Hessian approximation from the recent
(s, y) pairs and takes a scaled Newton-like step, with a line search choosing
the length.

`src/solvers/lbfgs.c`, f32 and f64.

## Structure

History is a ring buffer of `mem` pairs, s = x_{k+1} - x_k and y = g_{k+1} -
g_k. Each iteration:

1. Two-loop recursion turns the current gradient into a direction d = -H g,
   using the stored pairs and the rho = 1/(y.s) coefficients, scaled by
   gamma = (s.y)/(y.y) from the newest pair.
2. Armijo backtracking line search: start at t = 1 (the natural quasi-Newton
   step), halve until f(x + t d) <= f(x) + c1 t (g.d).
3. Push the new (s, y) pair if y.s came out positive.

The hot path is exactly dot and axpy over the history vectors, plus a scal in
the recursion. All three are the BLAS-1 kernels from M3, so passing a kernel
table runs the recursion on the AVX2 versions. M4 is mostly the recursion and
line-search logic on top of kernels that were already fast and already checked.

## Decisions

**Armijo, not Wolfe.** The plan called for Armijo backtracking. Armijo alone
does not enforce the curvature condition, so y.s can come out non-positive and
poison the rho = 1/(y.s) update. The guard is a cautious update: only push a
pair when y.s > eps |y|^2, skip it otherwise. Wolfe would let more pairs
through and tighten convergence, and it is the obvious upgrade later.

**Non-descent guard.** A stale history can occasionally produce a direction
with g.d >= 0. When that happens the step falls back to steepest descent
(d = -g) for that iteration.

**First step scaling.** With no history the direction is -g, and t = 1 would
overshoot badly on something like Rosenbrock. The first iteration scales the
initial step by 1/|g|_inf so it starts sane, then backtracks from there.

## Correctness

`tests/test_lbfgs.c`: converges on Rosenbrock 2D and 10D, Beale, and logistic
regression, all to the known minima, plus the f32 path. logreg runs through
the dispatched gemv, so the solver's gradient is on the AVX2 kernels.

The interesting check is the iteration count. L-BFGS on the conditioned
quadratic scales as sqrt(kappa):

| kappa | iters (mem=8) |
|---|---|
| 10 | 34 |
| 100 | 108 |
| 1000 | 340 |

Each 10x in kappa multiplies iterations by ~3.16 = sqrt(10), the textbook
CG-like rate, and bumping the memory from 4 to 16 barely moves it (340 -> 297
at kappa=1000), which is what a quadratic should do. That scaling is the
evidence the recursion is right, more than any single convergence point.

## The honest comparison

I first asserted L-BFGS would beat tuned momentum by 4x and it did not, 340 vs
510 at kappa=1000. That assertion was wrong, not the solver. Optimally-tuned
heavy-ball momentum also reaches the O(sqrt(kappa)) rate, so on a pure
quadratic the two are the same order and L-BFGS only edges it.

The real advantage is tuning. Momentum hits that rate only because the test
feeds it lr and beta computed from kappa. L-BFGS reaches the same rate knowing
nothing about the problem. Against plain gradient descent, which is O(kappa)
even at its optimal step, L-BFGS wins by a wide margin (340 vs thousands at
kappa=1000). So the test now checks two things: L-BFGS far outpaces untuned
gradient descent, and matches the best hand-tuned first-order method without
the tuning.

## Next, M5

The asm tuning pass. Non-temporal stores for the streaming kernels (fused Adam
writes params/m/v once and never rereads them, so bypassing the cache should
help at DRAM sizes), an unroll and accumulator-count sweep, and the rsqrt
approximation path for Adam that trades a few ULP for dropping the vsqrtps +
vdivps. AVX-512 stays a stretch, SDE correctness only, since this CPU cannot
run it.
