#ifndef STRIDE_SOLVER_H
#define STRIDE_SOLVER_H

#include <stddef.h>

#include "stride/objective.h"

/*
 * First-order solvers that drive the update kernels.
 *
 * The solver owns the iteration loop and the optimiser state (moments,
 * velocity). Each iteration is one gradient evaluation and one call into the
 * matching kernel from kernels.h. Swapping the C kernels for asm ones changes
 * nothing here.
 */

typedef enum {
    STRIDE_SGD = 0,
    STRIDE_SGD_MOMENTUM,
    STRIDE_RMSPROP,
    STRIDE_ADAM,
} stride_method;

typedef struct {
    stride_method method;
    size_t max_iters;
    double tol;      /* stop when max|g_i| < tol; 0 runs all iterations */
    double lr;
    double momentum; /* sgd+momentum */
    double rho;      /* rmsprop decay */
    double beta1;    /* adam */
    double beta2;    /* adam */
    double eps;      /* rmsprop, adam */
} stride_solver_opts_f64;

typedef struct {
    stride_method method;
    size_t max_iters;
    float tol;
    float lr;
    float momentum;
    float rho;
    float beta1;
    float beta2;
    float eps;
} stride_solver_opts_f32;

typedef struct {
    size_t iters;     /* steps actually taken */
    double grad_norm; /* inf-norm of the gradient at exit */
    int converged;    /* 1 if tol reached, 0 if max_iters hit, -1 on error */
} stride_result_f64;

typedef struct {
    size_t iters;
    float grad_norm;
    int converged;
} stride_result_f32;

/*
 * Per-iteration trace callback. Called with iter=0 for the starting point and
 * then once after every step, with the current position and the gradient
 * there. Entry k is the position after k steps, the same trajectory layout
 * Saddle's adam_optimise produces.
 */
typedef void (*stride_trace_f64)(void *user, size_t iter, const double *x, const double *g);
typedef void (*stride_trace_f32)(void *user, size_t iter, const float *x, const float *g);

/* Standard hyperparameters for each method; max_iters=1000, tol=1e-8. */
stride_solver_opts_f64 stride_solver_defaults_f64(stride_method method);
stride_solver_opts_f32 stride_solver_defaults_f32(stride_method method);

/*
 * Minimise obj starting from x (length obj->n). x is updated in place.
 * trace may be NULL.
 */
stride_result_f64 stride_minimise_f64(const stride_objective_f64 *obj, double *x,
                                      const stride_solver_opts_f64 *opts,
                                      stride_trace_f64 trace, void *trace_user);
stride_result_f32 stride_minimise_f32(const stride_objective_f32 *obj, float *x,
                                      const stride_solver_opts_f32 *opts,
                                      stride_trace_f32 trace, void *trace_user);

/*
 * L-BFGS with a two-loop recursion and Armijo backtracking line search.
 *
 * A quasi-Newton method: instead of a fixed step it builds a limited-memory
 * inverse-Hessian approximation from the last `mem` (s, y) pairs and takes a
 * scaled Newton-like step, with the line search picking the length. The hot
 * path is dot and axpy over those history vectors, so it runs on the BLAS-1
 * kernels: pass a kernel table for the AVX2 ones, or NULL for the C ones.
 *
 * Armijo alone does not guarantee the curvature condition, so a pair is only
 * stored when y . s is positive (a cautious update); otherwise it is skipped.
 */
typedef struct {
    size_t max_iters;
    double tol;         /* stop when max|g_i| < tol */
    size_t mem;         /* history pairs to keep (typically 5-10) */
    double c1;          /* Armijo sufficient-decrease constant (e.g. 1e-4) */
    double ls_decrease; /* backtracking factor in (0,1), e.g. 0.5 */
    size_t ls_max;      /* max backtracking steps before giving up */
} stride_lbfgs_opts_f64;

typedef struct {
    size_t max_iters;
    float tol;
    size_t mem;
    float c1;
    float ls_decrease;
    size_t ls_max;
} stride_lbfgs_opts_f32;

/* mem=8, c1=1e-4, ls_decrease=0.5, ls_max=40, max_iters=1000, tol=1e-8. */
stride_lbfgs_opts_f64 stride_lbfgs_defaults_f64(void);
stride_lbfgs_opts_f32 stride_lbfgs_defaults_f32(void);

stride_result_f64 stride_lbfgs_minimise_f64(const stride_objective_f64 *obj, double *x,
                                            const stride_lbfgs_opts_f64 *opts,
                                            const stride_kernel_table *kt, stride_trace_f64 trace,
                                            void *trace_user);
stride_result_f32 stride_lbfgs_minimise_f32(const stride_objective_f32 *obj, float *x,
                                            const stride_lbfgs_opts_f32 *opts,
                                            const stride_kernel_table *kt, stride_trace_f32 trace,
                                            void *trace_user);

#endif /* STRIDE_SOLVER_H */
