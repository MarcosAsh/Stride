/*
 * Cross-validation against Saddle's C Adam implementation.
 *
 * Built by `make check-saddle`, which compiles Saddle's adam.c and surfaces.c
 * from the sibling checkout. Two levels of checking, both bit-exact:
 *
 *   1. The kernel: stride_adam_step_f64 against Saddle's adam_step on random
 *      arrays, over multiple steps.
 *   2. The whole solver loop: Stride's Adam driving Saddle's Rosenbrock
 *      surface with Saddle's numerical gradients, against adam_optimise.
 *
 * Bit-exactness is the right bar here because both sides perform the same
 * floating point operations in the same order; any difference means the
 * arithmetic diverged, not that it's "close enough".
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "stride/stride.h"
#include "test_util.h"

/* Saddle's reference implementation. */
#include "adam.h"
#include "surfaces.h"

/* Kernel-level: identical inputs through both Adam steps, repeatedly. */
static void test_step_bitexact(void) {
    enum { N = 1537 }; /* deliberately not a round number */
    static double p1[N], m1[N], v1[N];
    static double p2[N], m2[N], v2[N];
    static double g[N];

    for (int i = 0; i < N; i++) {
        p1[i] = p2[i] = rand_uniform(-5, 5);
        m1[i] = m2[i] = rand_uniform(-1, 1);
        v1[i] = v2[i] = rand_uniform(0, 1);
    }

    const double lr = 0.001, b1 = 0.9, b2 = 0.999, eps = 1e-8;

    for (int step = 1; step <= 25; step++) {
        for (int i = 0; i < N; i++)
            g[i] = rand_uniform(-10, 10);

        /* Saddle computes bias corrections internally from the step count. */
        adam_step(p1, m1, v1, g, N, step, lr, b1, b2, eps);

        /* Stride takes them precomputed, same pow(), same values. */
        double bc1 = 1.0 - pow(b1, (double)step);
        double bc2 = 1.0 - pow(b2, (double)step);
        stride_adam_step_f64(p2, m2, v2, g, N, lr, b1, b2, eps, bc1, bc2);

        CHECK(memcmp(p1, p2, sizeof(p1)) == 0, "params differ from Saddle at step %d", step);
        CHECK(memcmp(m1, m2, sizeof(m1)) == 0, "first moment differs from Saddle at step %d", step);
        CHECK(memcmp(v1, v2, sizeof(v1)) == 0, "second moment differs from Saddle at step %d", step);
    }
}

/* Objective wrapper: Saddle's Rosenbrock surface with Saddle's central
 * differences (h=1e-7), so the gradient arithmetic matches adam_optimise's
 * numerical_grad exactly. */
static double saddle_rb_eval(const void *ctx, const double *x, size_t n) {
    (void)ctx;
    (void)n;
    return rosenbrock(x[0], x[1]);
}

static void saddle_rb_grad(const void *ctx, const double *x, double *g, size_t n) {
    (void)ctx;
    (void)n;
    const double h = 1e-7;
    g[0] = (rosenbrock(x[0] + h, x[1]) - rosenbrock(x[0] - h, x[1])) / (2.0 * h);
    g[1] = (rosenbrock(x[0], x[1] + h) - rosenbrock(x[0], x[1] - h)) / (2.0 * h);
}

typedef struct {
    double *xs;
    double *ys;
    size_t count;
} trace_buf;

static void record(void *user, size_t iter, const double *x, const double *g) {
    (void)g;
    trace_buf *tb = user;
    tb->xs[iter] = x[0];
    tb->ys[iter] = x[1];
    tb->count = iter + 1;
}

/* Solver-level: full 500-step trajectory on Rosenbrock. */
static void test_trajectory_bitexact(void) {
    enum { STEPS = 500 };
    static double sx[STEPS + 1], sy[STEPS + 1], sl[STEPS + 1];
    static double tx[STEPS + 1], ty[STEPS + 1];

    const double x0 = -1.2, y0 = 1.0;
    const double lr = 0.001, b1 = 0.9, b2 = 0.999, eps = 1e-8;

    /* Saddle's loop: numerical gradients + Adam on surface 0 (Rosenbrock). */
    adam_optimise(x0, y0, STEPS, lr, b1, b2, eps, 0, sx, sy, sl);

    /* The same arithmetic through Stride's solver API. */
    stride_objective_f64 obj = {
        .n = 2,
        .eval = saddle_rb_eval,
        .grad = saddle_rb_grad,
        .ctx = NULL,
    };

    stride_solver_opts_f64 opts = stride_solver_defaults_f64(STRIDE_ADAM);
    opts.lr = lr;
    opts.beta1 = b1;
    opts.beta2 = b2;
    opts.eps = eps;
    opts.max_iters = STEPS;
    opts.tol = 0; /* run every step, like Saddle does */

    double x[2] = {x0, y0};
    trace_buf tb = {.xs = tx, .ys = ty, .count = 0};
    stride_minimise_f64(&obj, x, &opts, record, &tb);

    CHECK(tb.count == STEPS + 1, "expected %d trace entries, got %zu", STEPS + 1, tb.count);

    int mismatches = 0;
    for (int k = 0; k <= STEPS; k++) {
        if (sx[k] != tx[k] || sy[k] != ty[k])
            mismatches++;
    }
    CHECK(mismatches == 0, "%d of %d trajectory points differ from Saddle", mismatches,
          STEPS + 1);
}

int main(void) {
    test_step_bitexact();
    test_trajectory_bitexact();

    TEST_DONE("test_saddle");
}
