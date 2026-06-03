/*
 * Validates the hand-written analytic gradients against central differences.
 * This is the test that has to pass before anything downstream (solvers,
 * kernels, asm) means anything: a wrong gradient converges to the wrong
 * place quietly.
 */

#include <math.h>
#include <stdlib.h>

#include "stride/stride.h"
#include "test_util.h"

static void check_gradient_f64(const char *name, const stride_objective_f64 *obj,
                               const double *x) {
    size_t n = obj->n;
    double *g = malloc(n * sizeof(*g));
    double *xp = malloc(n * sizeof(*xp));

    obj->grad(obj->ctx, x, g, n);

    /* h ~ eps^(1/3) balances truncation against roundoff for central
     * differences in fp64. */
    const double h = 1e-5;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++)
            xp[j] = x[j];

        xp[i] = x[i] + h;
        double fp = obj->eval(obj->ctx, xp, n);
        xp[i] = x[i] - h;
        double fm = obj->eval(obj->ctx, xp, n);
        double numeric = (fp - fm) / (2.0 * h);

        double tol = 1e-4 * fmax(1.0, fabs(numeric));
        CHECK(fabs(g[i] - numeric) < tol, "%s: grad[%zu] analytic=%.12g numeric=%.12g",
              name, i, g[i], numeric);
    }

    free(g);
    free(xp);
}

/* fp32 gradients should agree with fp64 ones to single precision. */
static void check_f32_matches_f64(const char *name, const stride_objective_f32 *obj32,
                                  const stride_objective_f64 *obj64, const double *x) {
    size_t n = obj64->n;
    double *g64 = malloc(n * sizeof(*g64));
    float *x32 = malloc(n * sizeof(*x32));
    float *g32 = malloc(n * sizeof(*g32));

    for (size_t i = 0; i < n; i++)
        x32[i] = (float)x[i];

    obj64->grad(obj64->ctx, x, g64, n);
    obj32->grad(obj32->ctx, x32, g32, n);

    for (size_t i = 0; i < n; i++) {
        double tol = 1e-4 * fmax(1.0, fabs(g64[i]));
        CHECK(fabs((double)g32[i] - g64[i]) < tol, "%s: f32/f64 grad[%zu] %.8g vs %.12g",
              name, i, (double)g32[i], g64[i]);
    }

    free(g64);
    free(x32);
    free(g32);
}

int main(void) {
    stride_objective_f64 rb2_64 = stride_rosenbrock_f64(2);
    stride_objective_f32 rb2_32 = stride_rosenbrock_f32(2);
    stride_objective_f64 be_64 = stride_beale_f64();
    stride_objective_f32 be_32 = stride_beale_f32();
    stride_objective_f64 hb_64 = stride_himmelblau_f64();
    stride_objective_f32 hb_32 = stride_himmelblau_f32();

    for (int trial = 0; trial < 20; trial++) {
        double xr[2] = {rand_uniform(-2, 2), rand_uniform(-1, 3)};
        double xb[2] = {rand_uniform(-2, 4), rand_uniform(-1, 1)};
        double xh[2] = {rand_uniform(-5, 5), rand_uniform(-5, 5)};

        check_gradient_f64("rosenbrock2", &rb2_64, xr);
        check_gradient_f64("beale", &be_64, xb);
        check_gradient_f64("himmelblau", &hb_64, xh);

        check_f32_matches_f64("rosenbrock2", &rb2_32, &rb2_64, xr);
        check_f32_matches_f64("beale", &be_32, &be_64, xb);
        check_f32_matches_f64("himmelblau", &hb_32, &hb_64, xh);
    }

    /* High-dimensional Rosenbrock and the conditioned quadratic. */
    {
        size_t n = 50;
        double *x = malloc(n * sizeof(*x));
        for (size_t i = 0; i < n; i++)
            x[i] = rand_uniform(-2, 2);

        stride_objective_f64 rbn = stride_rosenbrock_f64(n);
        check_gradient_f64("rosenbrock50", &rbn, x);

        stride_quadratic_ctx_f64 qc = {.kappa = 100.0};
        stride_objective_f64 q = stride_quadratic_f64(n, &qc);
        check_gradient_f64("quadratic", &q, x);

        free(x);
    }

    TEST_DONE("test_gradients");
}
