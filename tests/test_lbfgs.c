/*
 * L-BFGS convergence and the quasi-Newton payoff.
 *
 * The analytic functions check it reaches the known minima, the iteration
 * count against Adam shows the second-order method getting there in far fewer
 * steps, and the logistic regression run confirms it works on the real
 * objective through the dispatched kernels.
 */

#include <math.h>
#include <stdlib.h>

#include "stride/stride.h"
#include "stride/cpu.h"
#include "test_util.h"

static stride_kernel_table g_table;

static void test_lbfgs_rosenbrock2(void) {
    stride_objective_f64 obj = stride_rosenbrock_f64(2);
    double x[2] = {-1.2, 1.0};

    stride_lbfgs_opts_f64 opts = stride_lbfgs_defaults_f64();
    opts.max_iters = 200;
    opts.tol = 1e-8;

    stride_result_f64 res = stride_lbfgs_minimise_f64(&obj, x, &opts, &g_table, NULL, NULL);
    double f = obj.eval(obj.ctx, x, 2);

    CHECK(res.converged == 1, "lbfgs/rosenbrock2 did not converge: |g|=%g in %zu iters",
          res.grad_norm, res.iters);
    CHECK(f < 1e-12, "lbfgs/rosenbrock2 final loss %g at (%.8f, %.8f)", f, x[0], x[1]);
    CHECK(fabs(x[0] - 1.0) < 1e-5 && fabs(x[1] - 1.0) < 1e-5,
          "lbfgs/rosenbrock2 ended at (%.8f, %.8f)", x[0], x[1]);
}

static void test_lbfgs_rosenbrock_nd(void) {
    enum { N = 10 };
    stride_objective_f64 obj = stride_rosenbrock_f64(N);
    double x[N];
    for (int i = 0; i < N; i++)
        x[i] = (i % 2) ? 1.0 : -1.2;

    stride_lbfgs_opts_f64 opts = stride_lbfgs_defaults_f64();
    opts.max_iters = 500;
    opts.tol = 1e-7;

    stride_result_f64 res = stride_lbfgs_minimise_f64(&obj, x, &opts, &g_table, NULL, NULL);
    double f = obj.eval(obj.ctx, x, N);

    CHECK(res.converged == 1, "lbfgs/rosenbrock%d did not converge: |g|=%g in %zu iters", N,
          res.grad_norm, res.iters);
    CHECK(f < 1e-10, "lbfgs/rosenbrock%d final loss %g", N, f);
}

static void test_lbfgs_beale(void) {
    stride_objective_f64 obj = stride_beale_f64();
    double x[2] = {1.0, 1.0};

    stride_lbfgs_opts_f64 opts = stride_lbfgs_defaults_f64();
    opts.max_iters = 200;
    opts.tol = 1e-8;

    stride_lbfgs_minimise_f64(&obj, x, &opts, &g_table, NULL, NULL);
    double f = obj.eval(obj.ctx, x, 2);
    CHECK(f < 1e-12, "lbfgs/beale final loss %g at (%.8f, %.8f)", f, x[0], x[1]);
    CHECK(fabs(x[0] - 3.0) < 1e-5 && fabs(x[1] - 0.5) < 1e-5, "lbfgs/beale ended at (%.8f, %.8f)",
          x[0], x[1]);
}

/*
 * The quasi-Newton payoff on a conditioned quadratic.
 *
 * L-BFGS converges in O(sqrt(kappa)) iterations with no tuning. Plain gradient
 * descent, even at its optimal step, converges in O(kappa), so L-BFGS beats it
 * by a wide margin. Optimally-tuned heavy-ball momentum also reaches the
 * O(sqrt(kappa)) rate, but only because we hand it lr and beta computed from
 * kappa; L-BFGS matches that rate without being told kappa, which is the point.
 */
static void test_lbfgs_beats_first_order(void) {
    enum { N = 50 };
    double kappa = 1000.0;
    stride_quadratic_ctx_f64 qc = {.kappa = kappa};
    stride_objective_f64 obj = stride_quadratic_f64(N, &qc);

    double xl[N], xs[N], xm[N];
    for (int i = 0; i < N; i++)
        xl[i] = xs[i] = xm[i] = 1.0;

    stride_lbfgs_opts_f64 lo = stride_lbfgs_defaults_f64();
    lo.max_iters = 5000;
    lo.tol = 1e-8;
    stride_result_f64 lr = stride_lbfgs_minimise_f64(&obj, xl, &lo, &g_table, NULL, NULL);

    /* Plain gradient descent at its optimal step 2/(c_max+c_min) ~ 2/(kappa+1):
     * converges at rate (kappa-1)/(kappa+1), so O(kappa) iterations. */
    stride_solver_opts_f64 so = stride_solver_defaults_f64(STRIDE_SGD);
    so.lr = 2.0 / (kappa + 1.0);
    so.max_iters = 100000;
    so.tol = 1e-8;
    stride_result_f64 sr = stride_minimise_f64(&obj, xs, &so, NULL, NULL);

    /* Optimally-tuned heavy ball, the strongest first-order baseline. */
    double sk = sqrt(kappa);
    stride_solver_opts_f64 mo = stride_solver_defaults_f64(STRIDE_SGD_MOMENTUM);
    mo.lr = 4.0 / ((sk + 1.0) * (sk + 1.0));
    mo.momentum = ((sk - 1.0) / (sk + 1.0)) * ((sk - 1.0) / (sk + 1.0));
    mo.max_iters = 100000;
    mo.tol = 1e-8;
    stride_result_f64 mr = stride_minimise_f64(&obj, xm, &mo, NULL, NULL);

    CHECK(lr.converged == 1, "lbfgs/quadratic did not converge: |g|=%g", lr.grad_norm);
    CHECK(sr.converged == 1, "sgd/quadratic did not converge: |g|=%g", sr.grad_norm);
    CHECK(mr.converged == 1, "momentum/quadratic did not converge: |g|=%g", mr.grad_norm);

    /* vs plain GD (O(kappa)): a wide margin, L-BFGS is O(sqrt(kappa)). */
    CHECK(lr.iters < sr.iters / 4, "lbfgs (%zu) not far faster than plain gd (%zu)", lr.iters,
          sr.iters);
    /* vs optimally-tuned momentum: at least as good, with no tuning. */
    CHECK(lr.iters <= mr.iters, "lbfgs (%zu) slower than tuned momentum (%zu)", lr.iters,
          mr.iters);
}

/* logistic regression through L-BFGS, on the same synthetic setup as the
 * Adam test. */
static void test_lbfgs_logreg(void) {
    enum { M = 400, N = 16 };
    double w_true[N];
    for (int j = 0; j < N; j++)
        w_true[j] = rand_uniform(-2, 2);

    double *X = malloc(M * N * sizeof(double));
    double *y = malloc(M * sizeof(double));
    double *z = malloc(M * sizeof(double));
    double *r = malloc(M * sizeof(double));
    for (int i = 0; i < M; i++) {
        double logit = 0;
        for (int j = 0; j < N; j++) {
            double v = (j == 0) ? 1.0 : rand_uniform(-2, 2);
            X[i * N + j] = v;
            logit += v * w_true[j];
        }
        y[i] = logit > 0 ? 1.0 : 0.0;
    }

    stride_logreg_ctx_f64 c = {
        .X = X, .y = y, .m = M, .n = N, .l2 = 1e-3, .kt = &g_table, .z = z, .r = r};
    stride_objective_f64 obj = stride_logreg_f64(&c);

    double w[N];
    for (int j = 0; j < N; j++)
        w[j] = 0.0;
    double loss0 = obj.eval(obj.ctx, w, N);

    stride_lbfgs_opts_f64 opts = stride_lbfgs_defaults_f64();
    opts.max_iters = 500;
    opts.tol = 1e-8;
    stride_result_f64 res = stride_lbfgs_minimise_f64(&obj, w, &opts, &g_table, NULL, NULL);

    double loss1 = obj.eval(obj.ctx, w, N);
    CHECK(res.converged == 1, "lbfgs/logreg did not converge: |g|=%g in %zu iters", res.grad_norm,
          res.iters);
    CHECK(loss1 < loss0 * 0.5, "lbfgs/logreg did not reduce loss: %.6g -> %.6g", loss0, loss1);

    int correct = 0;
    for (int i = 0; i < M; i++) {
        double logit = 0;
        for (int j = 0; j < N; j++)
            logit += X[i * N + j] * w[j];
        if ((logit > 0 ? 1 : 0) == (int)y[i])
            correct++;
    }
    CHECK((double)correct / M > 0.9, "lbfgs/logreg accuracy only %.3f", (double)correct / M);

    free(X);
    free(y);
    free(z);
    free(r);
}

/* fp32 path end to end. */
static void test_lbfgs_f32(void) {
    stride_objective_f32 obj = stride_rosenbrock_f32(2);
    float x[2] = {-1.2f, 1.0f};

    stride_lbfgs_opts_f32 opts = stride_lbfgs_defaults_f32();
    opts.max_iters = 300;
    opts.tol = 1e-5f;

    stride_lbfgs_minimise_f32(&obj, x, &opts, &g_table, NULL, NULL);
    float f = obj.eval(obj.ctx, x, 2);
    CHECK(f < 1e-6f, "lbfgs/rosenbrock2 f32 final loss %g at (%.6f, %.6f)", (double)f, (double)x[0],
          (double)x[1]);
}

int main(void) {
    stride_kernel_table_init(&g_table, stride_cpu_flags());

    test_lbfgs_rosenbrock2();
    test_lbfgs_rosenbrock_nd();
    test_lbfgs_beale();
    test_lbfgs_beats_first_order();
    test_lbfgs_logreg();
    test_lbfgs_f32();

    TEST_DONE("test_lbfgs");
}
