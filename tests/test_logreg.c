/*
 * gemv and logistic regression.
 *
 * gemv is checked against a naive triple loop with a tolerance, since it rides
 * on the lane-split dot. The logistic regression gradient is checked against
 * central differences, then the whole thing is trained on synthetic data to
 * confirm it actually learns.
 */

#include <math.h>
#include <stdlib.h>

#include "stride/stride.h"
#include "stride/cpu.h"
#include "test_util.h"

static stride_kernel_table g_table;

/* gemv: y = A x, naive reference, then compare. */
static void test_gemv(void) {
    enum { ROWS = 37, COLS = 23 };
    double *A = malloc(ROWS * COLS * sizeof(double));
    double *x = malloc(COLS * sizeof(double));
    double *y = malloc(ROWS * sizeof(double));
    double *yref = malloc(ROWS * sizeof(double));

    for (int i = 0; i < ROWS * COLS; i++)
        A[i] = rand_uniform(-2, 2);
    for (int j = 0; j < COLS; j++)
        x[j] = rand_uniform(-2, 2);

    for (int r = 0; r < ROWS; r++) {
        double acc = 0;
        for (int c = 0; c < COLS; c++)
            acc += A[r * COLS + c] * x[c];
        yref[r] = acc;
    }

    stride_gemv_f64(&g_table, A, x, y, ROWS, COLS);
    for (int r = 0; r < ROWS; r++) {
        double tol = 1e-10 * (1.0 + fabs(yref[r]));
        CHECK(fabs(y[r] - yref[r]) < tol, "gemv row %d: %.15g vs %.15g", r, y[r], yref[r]);
    }

    free(A);
    free(x);
    free(y);
    free(yref);
}

/* gemv_t: y = A^T x, naive reference. */
static void test_gemv_t(void) {
    enum { ROWS = 41, COLS = 19 };
    double *A = malloc(ROWS * COLS * sizeof(double));
    double *x = malloc(ROWS * sizeof(double));
    double *y = malloc(COLS * sizeof(double));
    double *yref = malloc(COLS * sizeof(double));

    for (int i = 0; i < ROWS * COLS; i++)
        A[i] = rand_uniform(-2, 2);
    for (int r = 0; r < ROWS; r++)
        x[r] = rand_uniform(-2, 2);

    for (int c = 0; c < COLS; c++) {
        double acc = 0;
        for (int r = 0; r < ROWS; r++)
            acc += A[r * COLS + c] * x[r];
        yref[c] = acc;
    }

    stride_gemv_t_f64(&g_table, A, x, y, ROWS, COLS);
    for (int c = 0; c < COLS; c++) {
        double tol = 1e-10 * (1.0 + fabs(yref[c]));
        CHECK(fabs(y[c] - yref[c]) < tol, "gemv_t col %d: %.15g vs %.15g", c, y[c], yref[c]);
    }

    free(A);
    free(x);
    free(y);
    free(yref);
}

/* Build a logistic regression problem with m samples, n features. */
typedef struct {
    double *X, *y, *z, *r;
    size_t m, n;
} problem;

static problem make_problem(size_t m, size_t n, const double *w_true) {
    problem p = {.m = m, .n = n};
    p.X = malloc(m * n * sizeof(double));
    p.y = malloc(m * sizeof(double));
    p.z = malloc(m * sizeof(double));
    p.r = malloc(m * sizeof(double));

    for (size_t i = 0; i < m; i++) {
        double logit = 0;
        for (size_t j = 0; j < n; j++) {
            double v = (j == 0) ? 1.0 : rand_uniform(-2, 2); /* feature 0 is the bias */
            p.X[i * n + j] = v;
            logit += v * w_true[j];
        }
        p.y[i] = logit > 0 ? 1.0 : 0.0;
    }
    return p;
}

static void free_problem(problem *p) {
    free(p->X);
    free(p->y);
    free(p->z);
    free(p->r);
}

static stride_logreg_ctx_f64 ctx_of(const problem *p, double l2) {
    stride_logreg_ctx_f64 c = {
        .X = p->X, .y = p->y, .m = p->m, .n = p->n, .l2 = l2, .kt = &g_table, .z = p->z, .r = p->r};
    return c;
}

/* Analytic gradient vs central differences. */
static void test_logreg_gradient(void) {
    enum { M = 50, N = 8 };
    double w_true[N];
    for (int j = 0; j < N; j++)
        w_true[j] = rand_uniform(-1, 1);

    problem p = make_problem(M, N, w_true);
    stride_logreg_ctx_f64 c = ctx_of(&p, 1e-2);
    stride_objective_f64 obj = stride_logreg_f64(&c);

    double w[N], g[N], wp[N];
    for (int j = 0; j < N; j++)
        w[j] = rand_uniform(-1, 1);

    obj.grad(obj.ctx, w, g, N);

    const double h = 1e-6;
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++)
            wp[k] = w[k];
        wp[j] = w[j] + h;
        double fp = obj.eval(obj.ctx, wp, N);
        wp[j] = w[j] - h;
        double fm = obj.eval(obj.ctx, wp, N);
        double num = (fp - fm) / (2.0 * h);
        double tol = 1e-5 * (1.0 + fabs(num));
        CHECK(fabs(g[j] - num) < tol, "logreg grad[%d] analytic=%.12g numeric=%.12g", j, g[j], num);
    }

    free_problem(&p);
}

/* Train with Adam and confirm the loss drops and the fit classifies well. */
static void test_logreg_convergence(void) {
    enum { M = 400, N = 16 };
    double w_true[N];
    for (int j = 0; j < N; j++)
        w_true[j] = rand_uniform(-2, 2);

    problem p = make_problem(M, N, w_true);
    stride_logreg_ctx_f64 c = ctx_of(&p, 1e-3);
    stride_objective_f64 obj = stride_logreg_f64(&c);

    double w[N];
    for (int j = 0; j < N; j++)
        w[j] = 0.0;

    double loss0 = obj.eval(obj.ctx, w, N);

    stride_solver_opts_f64 opts = stride_solver_defaults_f64(STRIDE_ADAM);
    opts.lr = 0.05;
    opts.max_iters = 3000;
    opts.tol = 0;
    stride_minimise_f64(&obj, w, &opts, NULL, NULL);

    double loss1 = obj.eval(obj.ctx, w, N);
    CHECK(loss1 < loss0 * 0.5, "logreg training did not reduce loss: %.6g -> %.6g", loss0, loss1);

    /* Training accuracy against the labels it was fit on. */
    int correct = 0;
    for (size_t i = 0; i < p.m; i++) {
        double logit = 0;
        for (size_t j = 0; j < p.n; j++)
            logit += p.X[i * p.n + j] * w[j];
        int pred = logit > 0 ? 1 : 0;
        if (pred == (int)p.y[i])
            correct++;
    }
    double acc = (double)correct / (double)p.m;
    CHECK(acc > 0.9, "logreg training accuracy only %.3f", acc);

    free_problem(&p);
}

int main(void) {
    /* Use the best kernels the CPU offers, so gemv exercises the AVX2 path. */
    stride_kernel_table_init(&g_table, stride_cpu_flags());

    test_gemv();
    test_gemv_t();
    test_logreg_gradient();
    test_logreg_convergence();

    TEST_DONE("test_logreg");
}
