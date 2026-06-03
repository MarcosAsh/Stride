#ifndef STRIDE_KERNELS_X86_H
#define STRIDE_KERNELS_X86_H

#include <stddef.h>

/*
 * Declarations for the hand-written x86 asm kernels.
 *
 * These match the public C ABI exactly so the dispatch table can point at
 * them directly. The bodies live in the .asm files under this directory.
 */

void stride_adam_step_f32_avx2(float *params, float *m, float *v, const float *grads, size_t n,
                               float lr, float beta1, float beta2, float eps, float bc1,
                               float bc2);
void stride_adam_step_f64_avx2(double *params, double *m, double *v, const double *grads,
                               size_t n, double lr, double beta1, double beta2, double eps,
                               double bc1, double bc2);

void stride_axpy_f32_avx2(float *y, const float *x, size_t n, float a);
void stride_axpy_f64_avx2(double *y, const double *x, size_t n, double a);
float stride_dot_f32_avx2(const float *x, const float *y, size_t n);
double stride_dot_f64_avx2(const double *x, const double *y, size_t n);
void stride_scal_f32_avx2(float *x, size_t n, float a);
void stride_scal_f64_avx2(double *x, size_t n, double a);

#endif /* STRIDE_KERNELS_X86_H */
