# 03 - M3 BLAS-1, gemv, logistic regression

The milestone that turns Stride from a maths toy into something that solves a
real ML problem. Three BLAS-1 kernels in asm, a gemv built on top, and a
logistic regression objective that drives both.

## BLAS-1: axpy, dot, scal

`src/kernels/x86/blas1_avx2.asm`, f32 and f64.

axpy (`y += a*x`) and scal (`x *= a`) are element-wise, so the asm stays
bit-exact with the C reference: plain vmulps/vaddps, no FMA, same as the M2
approach. They slot into the harness at max ULP 0.

dot is the interesting one. It uses four lane accumulators and FMA to hide the
multiply-add latency, processing 32 floats per iteration before a horizontal
reduce and a scalar tail. Four accumulators reorder the sum, and FMA fuses the
rounding, so the result does not match the sequential C dot to the bit. That
is exactly the case the harness was built to handle.

### Checking a reduction

A reduction returns one scalar whose rounding error grows with n, so demanding
a bit-exact match against the C reference is the wrong bar. Instead the harness
computes the true dot product in higher precision (long double) along with
sum|x_i y_i|, and checks both the C and the asm result against it with the
standard summation error bound, 8 * eps * n * sum|x_i y_i| plus a floor. A
skipped tail or a wrong scale factor blows past that bound by orders of
magnitude, so the check still catches real bugs while tolerating the legitimate
reordering. The self-test confirms it: a dot that drops its tail is caught.

### Numbers (Core Ultra 7 155H, core 0, speedup over scalar C)

| kernel | 256 | 4096 | 65536 | 1M | 4M |
|---|---|---|---|---|---|
| dot f32 | 8.7x | 14.6x | 10.4x | 4.3x | 2.8x |
| dot f64 | 4.3x | 4.2x | 4.3x | 2.3x | 0.9x |
| axpy f32 | 3.4x | 5.4x | 4.5x | 2.2x | 1.4x |
| axpy f64 | 2.4x | 2.8x | 2.8x | 1.2x | 1.1x |
| scal f32 | 1.1x | 1.3x | 1.2x | 0.6x | 1.0x |
| scal f64 | 1.2x | 1.7x | 1.2x | 0.8x | 0.9x |

This is the roofline showing through. dot reuses both inputs in FMAs, so it is
compute-bound and wins big while the data fits in cache (up to ~15x), then
falls toward memory bandwidth at DRAM sizes. axpy does one FMA per two loads
and a store, so it is closer to bandwidth-bound and the win fades with n. scal
is one load, one multiply, one store, almost pure bandwidth, and SIMD barely
moves it. The honest takeaway: vector width pays off for the compute-bound
kernel, not for the ones that are already waiting on memory.

## gemv

`src/blas/gemv.c`, row-major, two forms:

- `stride_gemv`:   y = A x,   a dot product per row.
- `stride_gemv_t`: y = A^T x, an axpy accumulation per row.

Both take a kernel table and call through the dispatched dot/axpy, so they
inherit the AVX2 kernels for free (pass NULL to fall back to C). gemv is the
no-trans form built from dot, gemv_t the transpose form built from axpy, which
is neat: the two BLAS-1 kernels each end up driving one half of the logistic
regression gradient.

## Logistic regression

`src/objectives/logreg.c`. The real ML objective.

    loss(w) = (1/m) sum_i [ softplus(x_i . w) - y_i (x_i . w) ] + (l2/2)|w|^2
    grad(w) = (1/m) X^T (sigmoid(X w) - y) + l2 w

The gradient is one gemv (z = X w) and one gemv_t (X^T r), with a sigmoid and a
scaled add in between. softplus and sigmoid are written in the
overflow-safe branches so large logits do not blow up exp.

Tests in `tests/test_logreg.c`:

- gemv and gemv_t against a naive triple loop.
- The analytic gradient against central differences, the same bar the analytic
  test functions are held to.
- A training run: synthetic data from a known w*, Adam for 3000 steps, loss has
  to drop by at least half and training accuracy clear 90%. It does.

The objective carries a kernel table, so the solver's per-iteration gradient
runs on the AVX2 gemv. That is the point where Stride stops being a collection
of kernels and becomes an ML solver.

## Next, M4

L-BFGS, two-loop recursion plus an Armijo backtracking line search. The hot
path is dot and axpy over the history vectors, both of which now exist in asm,
so this should mostly be the recursion and the line search logic on top of
kernels that are already fast and already checked.
