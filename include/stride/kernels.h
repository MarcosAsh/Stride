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

typedef struct {
    stride_sgd_step_f32_fn sgd_step_f32;
    stride_sgd_step_f64_fn sgd_step_f64;
    stride_sgd_momentum_step_f32_fn sgd_momentum_step_f32;
    stride_sgd_momentum_step_f64_fn sgd_momentum_step_f64;
    stride_rmsprop_step_f32_fn rmsprop_step_f32;
    stride_rmsprop_step_f64_fn rmsprop_step_f64;
    stride_adam_step_f32_fn adam_step_f32;
    stride_adam_step_f64_fn adam_step_f64;
} stride_kernel_table;

void stride_kernel_table_init(stride_kernel_table *t, int cpu_flags);

#endif /* STRIDE_KERNELS_H */
