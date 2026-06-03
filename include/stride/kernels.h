#ifndef STRIDE_KERNELS_H
#define STRIDE_KERNELS_H

#include <stddef.h>

/*
 * Optimiser update kernels, the hot loops. One streaming pass over the
 * parameter arrays per call.
 *
 * The C versions are the references the asm kernels get checked against (see
 * the M1 harness). These signatures are also the asm ABI, so nothing here
 * needs libm or per-element branching.
 *
 * State arrays are updated in place. grads is read-only.
 */

/* SGD:
 *   params[i] -= lr * grads[i]
 */
void stride_sgd_step_f32(float *params, const float *grads, size_t n, float lr);
void stride_sgd_step_f64(double *params, const double *grads, size_t n, double lr);

/* SGD with momentum:
 *   vel[i]     = momentum * vel[i] + grads[i]
 *   params[i] -= lr * vel[i]
 */
void stride_sgd_momentum_step_f32(float *params, float *vel, const float *grads,
                                  size_t n, float lr, float momentum);
void stride_sgd_momentum_step_f64(double *params, double *vel, const double *grads,
                                  size_t n, double lr, double momentum);

/* RMSprop:
 *   sq[i]      = rho * sq[i] + (1 - rho) * grads[i]^2
 *   params[i] -= lr * grads[i] / (sqrt(sq[i]) + eps)
 */
void stride_rmsprop_step_f32(float *params, float *sq, const float *grads,
                             size_t n, float lr, float rho, float eps);
void stride_rmsprop_step_f64(double *params, double *sq, const double *grads,
                             size_t n, double lr, double rho, double eps);

/* Adam, fused:
 *   m[i]       = beta1 * m[i] + (1 - beta1) * grads[i]
 *   v[i]       = beta2 * v[i] + (1 - beta2) * grads[i]^2
 *   m_hat      = m[i] / bc1
 *   v_hat      = v[i] / bc2
 *   params[i] -= lr * m_hat / (sqrt(v_hat) + eps)
 *
 * The caller computes bc1 = 1 - beta1^step and bc2 = 1 - beta2^step once per
 * step, which keeps the kernel a pure streaming pass: three loads, three
 * stores, no pow, no branches. The AVX2 version (M2) implements this boundary.
 */
void stride_adam_step_f32(float *params, float *m, float *v, const float *grads,
                          size_t n, float lr, float beta1, float beta2, float eps,
                          float bc1, float bc2);
void stride_adam_step_f64(double *params, double *m, double *v, const double *grads,
                          size_t n, double lr, double beta1, double beta2, double eps,
                          double bc1, double bc2);

/*
 * BLAS-1: the primitives L-BFGS leans on (M4) and the building blocks for
 * gemv below.
 *
 * axpy and scal are element-wise, so their asm stays bit-exact with the C
 * reference. dot is a reduction: the asm splits the sum across lane
 * accumulators and uses FMA, which reorders the arithmetic, so it is NOT
 * bit-exact. The harness checks dot against a higher-precision truth with an
 * error bound instead of demanding an exact match.
 */

/* y[i] += a * x[i] */
void stride_axpy_f32(float *y, const float *x, size_t n, float a);
void stride_axpy_f64(double *y, const double *x, size_t n, double a);

/* returns sum_i x[i] * y[i] */
float stride_dot_f32(const float *x, const float *y, size_t n);
double stride_dot_f64(const double *x, const double *y, size_t n);

/* x[i] *= a */
void stride_scal_f32(float *x, size_t n, float a);
void stride_scal_f64(double *x, size_t n, double a);

/*
 * Kernel dispatch.
 *
 * One function pointer type per kernel, plus a table holding the chosen
 * implementation of each. stride_kernel_table_init picks the best one the
 * given CPU flags (see cpu.h) allow. flags=0 gives the portable C references.
 * ISA-specific implementations get added per-ISA from M2 on.
 */

typedef void (*stride_sgd_step_f32_fn)(float *, const float *, size_t, float);
typedef void (*stride_sgd_step_f64_fn)(double *, const double *, size_t, double);
typedef void (*stride_sgd_momentum_step_f32_fn)(float *, float *, const float *, size_t, float,
                                                float);
typedef void (*stride_sgd_momentum_step_f64_fn)(double *, double *, const double *, size_t,
                                                double, double);
typedef void (*stride_rmsprop_step_f32_fn)(float *, float *, const float *, size_t, float, float,
                                           float);
typedef void (*stride_rmsprop_step_f64_fn)(double *, double *, const double *, size_t, double,
                                           double, double);
typedef void (*stride_adam_step_f32_fn)(float *, float *, float *, const float *, size_t, float,
                                        float, float, float, float, float);
typedef void (*stride_adam_step_f64_fn)(double *, double *, double *, const double *, size_t,
                                        double, double, double, double, double, double);
typedef void (*stride_axpy_f32_fn)(float *, const float *, size_t, float);
typedef void (*stride_axpy_f64_fn)(double *, const double *, size_t, double);
typedef float (*stride_dot_f32_fn)(const float *, const float *, size_t);
typedef double (*stride_dot_f64_fn)(const double *, const double *, size_t);
typedef void (*stride_scal_f32_fn)(float *, size_t, float);
typedef void (*stride_scal_f64_fn)(double *, size_t, double);

typedef struct {
    stride_sgd_step_f32_fn sgd_step_f32;
    stride_sgd_step_f64_fn sgd_step_f64;
    stride_sgd_momentum_step_f32_fn sgd_momentum_step_f32;
    stride_sgd_momentum_step_f64_fn sgd_momentum_step_f64;
    stride_rmsprop_step_f32_fn rmsprop_step_f32;
    stride_rmsprop_step_f64_fn rmsprop_step_f64;
    stride_adam_step_f32_fn adam_step_f32;
    stride_adam_step_f64_fn adam_step_f64;
    stride_axpy_f32_fn axpy_f32;
    stride_axpy_f64_fn axpy_f64;
    stride_dot_f32_fn dot_f32;
    stride_dot_f64_fn dot_f64;
    stride_scal_f32_fn scal_f32;
    stride_scal_f64_fn scal_f64;
} stride_kernel_table;

void stride_kernel_table_init(stride_kernel_table *t, int cpu_flags);

/*
 * gemv, BLAS-2, row-major.
 *
 * stride_gemv:   y = A * x,   A is rows x cols, x has cols, y has rows.
 * stride_gemv_t: y = A^T * x, A is rows x cols, x has rows, y has cols.
 *
 * Both take a kernel table so they ride on the dispatched dot/axpy (pass NULL
 * to fall back to the C kernels). gemv is a row of dot products, gemv_t is a
 * sum of axpys, so this is where dot and axpy earn their keep. The logistic
 * regression gradient is one of each.
 */
void stride_gemv_f32(const stride_kernel_table *t, const float *A, const float *x, float *y,
                     size_t rows, size_t cols);
void stride_gemv_f64(const stride_kernel_table *t, const double *A, const double *x, double *y,
                     size_t rows, size_t cols);
void stride_gemv_t_f32(const stride_kernel_table *t, const float *A, const float *x, float *y,
                       size_t rows, size_t cols);
void stride_gemv_t_f64(const stride_kernel_table *t, const double *A, const double *x, double *y,
                       size_t rows, size_t cols);

#endif /* STRIDE_KERNELS_H */
