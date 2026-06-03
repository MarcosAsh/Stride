#ifndef STRIDE_OBJECTIVE_H
#define STRIDE_OBJECTIVE_H

#include <stddef.h>

/*
 * An objective is anything with a value and a gradient over R^n.
 *
 * eval and grad take a context pointer so objectives can carry data (a
 * dataset for logistic regression, coefficients for the quadratic). The
 * analytic test functions ignore it.
 */

typedef struct {
    size_t n;
    double (*eval)(const void *ctx, const double *x, size_t n);
    void (*grad)(const void *ctx, const double *x, double *g, size_t n);
    const void *ctx;
} stride_objective_f64;

typedef struct {
    size_t n;
    float (*eval)(const void *ctx, const float *x, size_t n);
    void (*grad)(const void *ctx, const float *x, float *g, size_t n);
    const void *ctx;
} stride_objective_f32;

/*
 * Analytic test functions with hand-written gradients.
 *
 * Saddle differentiates its surfaces numerically. These carry exact
 * gradients, so they validate solver behaviour without numerical
 * differentiation noise on top.
 */

/* Chain Rosenbrock over R^n (n >= 2):
 *   f(x) = sum_{i=0}^{n-2} (1 - x_i)^2 + 100 * (x_{i+1} - x_i^2)^2
 * Minimum f=0 at (1, ..., 1). The n=2 case is the classic banana matching
 * Saddle's surface.
 */
stride_objective_f64 stride_rosenbrock_f64(size_t n);
stride_objective_f32 stride_rosenbrock_f32(size_t n);

/* Beale (2D only). Minimum f=0 at (3, 0.5). */
stride_objective_f64 stride_beale_f64(void);
stride_objective_f32 stride_beale_f32(void);

/* Himmelblau (2D only). Four minima with f=0, e.g. (3, 2). */
stride_objective_f64 stride_himmelblau_f64(void);
stride_objective_f32 stride_himmelblau_f32(void);

/*
 * Ill-conditioned quadratic over R^n:
 *   f(x) = 0.5 * sum_i c_i * x_i^2,  c_i log-spaced from 1 to kappa
 * Minimum f=0 at the origin. kappa is the condition number, the dial for how
 * hard first-order methods have to work, and one of the benchmark axes.
 *
 * The ctx struct must outlive the objective.
 */
typedef struct {
    double kappa;
} stride_quadratic_ctx_f64;

typedef struct {
    float kappa;
} stride_quadratic_ctx_f32;

stride_objective_f64 stride_quadratic_f64(size_t n, const stride_quadratic_ctx_f64 *ctx);
stride_objective_f32 stride_quadratic_f32(size_t n, const stride_quadratic_ctx_f32 *ctx);

#endif /* STRIDE_OBJECTIVE_H */
