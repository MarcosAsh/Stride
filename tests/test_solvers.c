/*
 * Convergence tests for the reference solvers: each method on objectives
 * whose minima are known, with assertions on where the iterates end up.
 *
 * Adaptive methods (RMSprop, Adam) with a constant learning rate stall at a
 * noise floor proportional to lr near the optimum, so their assertions are on
 * proximity to the minimiser, not on gradient-norm convergence.
 */

#include <math.h>
#include <stdlib.h>

#include "stride/stride.h"
#include "test_util.h"

/* SGD and SGD+momentum on the conditioned quadratic converge linearly and
 * hit a tight gradient tolerance. Both run with their optimal hyperparameters
 * for condition number kappa, which is what makes the iteration-count
 * comparison meaningful:
 *
 *   SGD:        lr = 2/(c_max + c_min),       rate (kappa-1)/(kappa+1)
 *   Heavy ball: lr = 4/(sqrt(c_max)+sqrt(c_min))^2,
 *               beta = ((sqrt(kappa)-1)/(sqrt(kappa)+1))^2, rate sqrt(beta)
 */
static void test_sgd_quadratic(void) {
    enum { N = 20 };
    const double kappa = 10.0;
    stride_quadratic_ctx_f64 qc = {.kappa = kappa};
    stride_objective_f64 obj = stride_quadratic_f64(N, &qc);

    double x[N];

    /* Plain SGD, optimally tuned. */
    for (int i = 0; i < N; i++)
        x[i] = 1.0;
    stride_solver_opts_f64 opts = stride_solver_defaults_f64(STRIDE_SGD);
    opts.lr = 2.0 / (kappa + 1.0);
    opts.max_iters = 10000;
    opts.tol = 1e-10;
    stride_result_f64 res = stride_minimise_f64(&obj, x, &opts, NULL, NULL);
    CHECK(res.converged == 1, "sgd/quadratic did not converge: |g|=%g after %zu iters",
          res.grad_norm, res.iters);

    /* Heavy ball, optimally tuned: asymptotic rate sqrt(beta) ~ 0.52 vs
     * SGD's 0.82, so it should need far fewer iterations. */
    double xm[N];
    for (int i = 0; i < N; i++)
        xm[i] = 1.0;
    double sk = sqrt(kappa);
    stride_solver_opts_f64 mopts = stride_solver_defaults_f64(STRIDE_SGD_MOMENTUM);
    mopts.lr = 4.0 / ((sk + 1.0) * (sk + 1.0));
    mopts.momentum = ((sk - 1.0) / (sk + 1.0)) * ((sk - 1.0) / (sk + 1.0));
    mopts.max_iters = 10000;
    mopts.tol = 1e-10;
    stride_result_f64 mres = stride_minimise_f64(&obj, xm, &mopts, NULL, NULL);
    CHECK(mres.converged == 1, "momentum/quadratic did not converge: |g|=%g after %zu iters",
          mres.grad_norm, mres.iters);
    CHECK(mres.iters < res.iters, "momentum (%zu iters) not faster than sgd (%zu iters)",
          mres.iters, res.iters);
}

/* Adam on the 2D Rosenbrock from the classic start point. */
static void test_adam_rosenbrock(void) {
    stride_objective_f64 obj = stride_rosenbrock_f64(2);
    double x[2] = {-1.2, 1.0};

    stride_solver_opts_f64 opts = stride_solver_defaults_f64(STRIDE_ADAM);
    opts.lr = 0.01;
    opts.max_iters = 50000;
    opts.tol = 1e-8;

    stride_result_f64 res = stride_minimise_f64(&obj, x, &opts, NULL, NULL);
    double f = obj.eval(obj.ctx, x, 2);

    CHECK(f < 1e-6, "adam/rosenbrock final loss %g (x=%.6f, y=%.6f, %zu iters, |g|=%g)",
          f, x[0], x[1], res.iters, res.grad_norm);
    CHECK(fabs(x[0] - 1.0) < 1e-2 && fabs(x[1] - 1.0) < 1e-2,
          "adam/rosenbrock ended at (%.6f, %.6f), expected (1, 1)", x[0], x[1]);
}

/* Adam on Beale from a benign start. */
static void test_adam_beale(void) {
    stride_objective_f64 obj = stride_beale_f64();
    double x[2] = {1.0, 1.0};

    stride_solver_opts_f64 opts = stride_solver_defaults_f64(STRIDE_ADAM);
    opts.lr = 0.01;
    opts.max_iters = 50000;
    opts.tol = 1e-8;

    stride_minimise_f64(&obj, x, &opts, NULL, NULL);
    double f = obj.eval(obj.ctx, x, 2);

    CHECK(f < 1e-6, "adam/beale final loss %g (x=%.6f, y=%.6f)", f, x[0], x[1]);
    CHECK(fabs(x[0] - 3.0) < 1e-2 && fabs(x[1] - 0.5) < 1e-2,
          "adam/beale ended at (%.6f, %.6f), expected (3, 0.5)", x[0], x[1]);
}

/* Himmelblau has four minima, all with f=0; any of them counts. */
static void test_adam_himmelblau(void) {
    stride_objective_f64 obj = stride_himmelblau_f64();
    double x[2] = {0.0, 0.0};

    stride_solver_opts_f64 opts = stride_solver_defaults_f64(STRIDE_ADAM);
    opts.lr = 0.01;
    opts.max_iters = 50000;
    opts.tol = 1e-8;

    stride_minimise_f64(&obj, x, &opts, NULL, NULL);
    double f = obj.eval(obj.ctx, x, 2);

    CHECK(f < 1e-6, "adam/himmelblau final loss %g (x=%.6f, y=%.6f)", f, x[0], x[1]);
}

/* RMSprop on the quadratic: gets near the optimum, then oscillates at the
 * lr-scale noise floor. Assert on proximity, not convergence. */
static void test_rmsprop_quadratic(void) {
    enum { N = 20 };
    stride_quadratic_ctx_f64 qc = {.kappa = 10.0};
    stride_objective_f64 obj = stride_quadratic_f64(N, &qc);

    double x[N];
    for (int i = 0; i < N; i++)
        x[i] = 1.0;

    stride_solver_opts_f64 opts = stride_solver_defaults_f64(STRIDE_RMSPROP);
    opts.lr = 0.01;
    opts.max_iters = 5000;
    opts.tol = 0;

    stride_minimise_f64(&obj, x, &opts, NULL, NULL);
    double f = obj.eval(obj.ctx, x, N);

    CHECK(f < 1e-2, "rmsprop/quadratic final loss %g", f);
}

/* The fp32 path end to end: Adam on Rosenbrock in single precision. */
static void test_adam_rosenbrock_f32(void) {
    stride_objective_f32 obj = stride_rosenbrock_f32(2);
    float x[2] = {-1.2f, 1.0f};

    stride_solver_opts_f32 opts = stride_solver_defaults_f32(STRIDE_ADAM);
    opts.lr = 0.01f;
    opts.max_iters = 50000;
    opts.tol = 1e-6f;

    stride_minimise_f32(&obj, x, &opts, NULL, NULL);
    float f = obj.eval(obj.ctx, x, 2);

    CHECK(f < 1e-4f, "adam/rosenbrock f32 final loss %g (x=%.6f, y=%.6f)",
          (double)f, (double)x[0], (double)x[1]);
}

/* The trace callback fires once per recorded point with monotonically
 * increasing iteration numbers. */
static void count_trace(void *user, size_t iter, const double *x, const double *g) {
    (void)x;
    (void)g;
    size_t *count = user;
    CHECK(iter == *count, "trace iter %zu, expected %zu", iter, *count);
    (*count)++;
}

static void test_trace(void) {
    stride_objective_f64 obj = stride_rosenbrock_f64(2);
    double x[2] = {-1.2, 1.0};

    stride_solver_opts_f64 opts = stride_solver_defaults_f64(STRIDE_ADAM);
    opts.max_iters = 100;
    opts.tol = 0;

    size_t count = 0;
    stride_result_f64 res = stride_minimise_f64(&obj, x, &opts, count_trace, &count);

    CHECK(count == 101, "expected 101 trace calls (initial + 100 steps), got %zu", count);
    CHECK(res.iters == 100, "expected 100 iters, got %zu", res.iters);
}

int main(void) {
    test_sgd_quadratic();
    test_rmsprop_quadratic();
    test_adam_rosenbrock();
    test_adam_beale();
    test_adam_himmelblau();
    test_adam_rosenbrock_f32();
    test_trace();

    TEST_DONE("test_solvers");
}
