/*
 * BLAS-1 checks and benchmarks, one instantiation per precision.
 *
 * axpy and scal write arrays and are checked bit-exact. dot returns a scalar
 * and is checked against a higher-precision truth: the asm reorders the sum,
 * so an exact match is the wrong bar. The bound is a generous multiple of the
 * standard floating-point summation error, tight enough that a skipped tail
 * or a wrong scale still blows past it.
 */

static void FN(fill_blas)(REAL *buf, size_t n, double range) {
    for (size_t i = 0; i < n; i++)
        buf[i] = (REAL)checkasm_rand_uniform(-range, range);
}

/* ---- axpy ---- */

static void FN(check_axpy)(FNT(stride_axpy) ref, FNT(stride_axpy) newfn, const char *impl,
                           const checkasm_opts *opts) {
    if (opts->filter && !strstr("axpy" SUFFIX_STR, opts->filter))
        return;

    int fail0 = checkasm_failure_count();
    for (size_t si = 0; si < NUM_BLAS_CHECK_SIZES; si++) {
        size_t n = blas_check_sizes[si];
        for (int off = 0; off <= 1; off++) {
            size_t an = n + (size_t)off;
            REAL *y0 = checkasm_alloc(sizeof(REAL), an);
            REAL *y1 = checkasm_alloc(sizeof(REAL), an);
            REAL *x = checkasm_alloc(sizeof(REAL), an);
            REAL *xc = checkasm_alloc(sizeof(REAL), an);
            REAL a = (REAL)checkasm_rand_uniform(-3, 3);

            FN(fill_blas)(y0, an, 5.0);
            memcpy(y1, y0, an * sizeof(REAL));
            FN(fill_blas)(x, an, 5.0);
            memcpy(xc, x, an * sizeof(REAL));

            checkasm_context("axpy" SUFFIX_STR, impl, n, off);
            ref(y0 + off, x + off, n, a);
            newfn(y1 + off, x + off, n, a);

            CMP(y0, y1, an, 0, "y");
            checkasm_check_guards(y1, sizeof(REAL), an, "y");
            CMP(x, xc, an, 0, "x (read-only input)");

            checkasm_free(y0, sizeof(REAL), an);
            checkasm_free(y1, sizeof(REAL), an);
            checkasm_free(x, sizeof(REAL), an);
            checkasm_free(xc, sizeof(REAL), an);
        }
    }
    checkasm_report_check("axpy" SUFFIX_STR, impl, checkasm_failure_count() != fail0);
}

/* ---- scal ---- */

static void FN(check_scal)(FNT(stride_scal) ref, FNT(stride_scal) newfn, const char *impl,
                           const checkasm_opts *opts) {
    if (opts->filter && !strstr("scal" SUFFIX_STR, opts->filter))
        return;

    int fail0 = checkasm_failure_count();
    for (size_t si = 0; si < NUM_BLAS_CHECK_SIZES; si++) {
        size_t n = blas_check_sizes[si];
        for (int off = 0; off <= 1; off++) {
            size_t an = n + (size_t)off;
            REAL *x0 = checkasm_alloc(sizeof(REAL), an);
            REAL *x1 = checkasm_alloc(sizeof(REAL), an);
            REAL a = (REAL)checkasm_rand_uniform(-3, 3);

            FN(fill_blas)(x0, an, 5.0);
            memcpy(x1, x0, an * sizeof(REAL));

            checkasm_context("scal" SUFFIX_STR, impl, n, off);
            ref(x0 + off, n, a);
            newfn(x1 + off, n, a);

            CMP(x0, x1, an, 0, "x");
            checkasm_check_guards(x1, sizeof(REAL), an, "x");

            checkasm_free(x0, sizeof(REAL), an);
            checkasm_free(x1, sizeof(REAL), an);
        }
    }
    checkasm_report_check("scal" SUFFIX_STR, impl, checkasm_failure_count() != fail0);
}

/* ---- dot ---- */

static void FN(check_dot_value)(const char *what, size_t n, REAL got, long double truth,
                                long double sum_abs) {
    long double err = (long double)got - truth;
    if (err < 0)
        err = -err;
    /* standard summation error is ~n*eps*sum|x_i y_i|, doubled for the
     * products; 8x gives headroom, the floor covers sum_abs near zero */
    long double bound = 8.0L * (long double)EPS * (long double)n * sum_abs + 16.0L * (long double)EPS;
    if (err > bound)
        checkasm_fail("%s: |got - truth| = %Lg exceeds bound %Lg (got %.10g, truth %.10Lg)", what,
                      err, bound, (double)got, truth);
}

static void FN(check_dot)(FNT(stride_dot) ref, FNT(stride_dot) newfn, const char *impl,
                          const checkasm_opts *opts) {
    if (opts->filter && !strstr("dot" SUFFIX_STR, opts->filter))
        return;

    int fail0 = checkasm_failure_count();
    for (size_t si = 0; si < NUM_BLAS_CHECK_SIZES; si++) {
        size_t n = blas_check_sizes[si];
        for (int off = 0; off <= 1; off++) {
            size_t an = n + (size_t)off;
            REAL *x = checkasm_alloc(sizeof(REAL), an);
            REAL *y = checkasm_alloc(sizeof(REAL), an);
            REAL *xc = checkasm_alloc(sizeof(REAL), an);
            REAL *yc = checkasm_alloc(sizeof(REAL), an);

            FN(fill_blas)(x, an, 2.0);
            FN(fill_blas)(y, an, 2.0);
            memcpy(xc, x, an * sizeof(REAL));
            memcpy(yc, y, an * sizeof(REAL));

            long double truth = 0, sum_abs = 0;
            for (size_t i = 0; i < n; i++) {
                long double p = (long double)x[off + i] * (long double)y[off + i];
                truth += p;
                sum_abs += p < 0 ? -p : p;
            }

            checkasm_context("dot" SUFFIX_STR, impl, n, off);
            REAL r_ref = ref(x + off, y + off, n);
            REAL r_new = newfn(x + off, y + off, n);

            FN(check_dot_value)("dot ref", n, r_ref, truth, sum_abs);
            FN(check_dot_value)("dot", n, r_new, truth, sum_abs);
            CMP(x, xc, an, 0, "x (read-only input)");
            CMP(y, yc, an, 0, "y (read-only input)");

            checkasm_free(x, sizeof(REAL), an);
            checkasm_free(y, sizeof(REAL), an);
            checkasm_free(xc, sizeof(REAL), an);
            checkasm_free(yc, sizeof(REAL), an);
        }
    }
    checkasm_report_check("dot" SUFFIX_STR, impl, checkasm_failure_count() != fail0);
}

/* ---- benchmarks ---- */

typedef struct {
    FNT(stride_axpy) axpy;
    FNT(stride_dot) dot;
    FNT(stride_scal) scal;
    REAL *a;
    REAL *b;
    size_t n;
    REAL s;
    volatile REAL sink;
} FN(blas_bench_args);

static void FN(axpy_thunk)(void *p) {
    FN(blas_bench_args) *g = p;
    g->axpy(g->a, g->b, g->n, g->s);
}
static void FN(dot_thunk)(void *p) {
    FN(blas_bench_args) *g = p;
    g->sink = g->dot(g->a, g->b, g->n);
}
static void FN(scal_thunk)(void *p) {
    FN(blas_bench_args) *g = p;
    g->scal(g->a, g->n, g->s);
}

static void FN(bench_blas_one)(const char *name, const char *impl, void (*thunk)(void *),
                               FN(blas_bench_args) * args, const checkasm_opts *opts) {
    if (opts->filter && !strstr(name, opts->filter))
        return;
    for (size_t si = 0; si < NUM_BLAS_BENCH_SIZES; si++) {
        size_t n = blas_bench_sizes[si];
        args->a = checkasm_alloc(sizeof(REAL), n);
        args->b = checkasm_alloc(sizeof(REAL), n);
        FN(fill_blas)(args->a, n, 5.0);
        FN(fill_blas)(args->b, n, 5.0);
        args->n = n;
        args->s = (REAL)1.0001;

        long runs = (long)opts->bench_runs * 4096 / (long)(n ? n : 1);
        if (runs < 4)
            runs = 4;
        if (runs > 2048)
            runs = 2048;

        uint64_t ticks = checkasm_bench(thunk, args, (int)runs);
        checkasm_report_bench(name, impl, n, ticks);

        checkasm_free(args->a, sizeof(REAL), n);
        checkasm_free(args->b, sizeof(REAL), n);
    }
}

/* combined entry points for this precision */

static void FN(check_blas_tables)(const stride_kernel_table *ref, const stride_kernel_table *new,
                                  const char *impl, const checkasm_opts *opts) {
    if (new->FN(axpy) != ref->FN(axpy))
        FN(check_axpy)(ref->FN(axpy), new->FN(axpy), impl, opts);
    if (new->FN(dot) != ref->FN(dot))
        FN(check_dot)(ref->FN(dot), new->FN(dot), impl, opts);
    if (new->FN(scal) != ref->FN(scal))
        FN(check_scal)(ref->FN(scal), new->FN(scal), impl, opts);
}

static void FN(bench_blas_tables)(const stride_kernel_table *t,
                                  const stride_kernel_table *skip_same, const char *impl,
                                  const checkasm_opts *opts) {
    FN(blas_bench_args) args = {0};
    if (!skip_same || t->FN(axpy) != skip_same->FN(axpy)) {
        args.axpy = t->FN(axpy);
        FN(bench_blas_one)("axpy" SUFFIX_STR, impl, FN(axpy_thunk), &args, opts);
    }
    if (!skip_same || t->FN(dot) != skip_same->FN(dot)) {
        args.dot = t->FN(dot);
        FN(bench_blas_one)("dot" SUFFIX_STR, impl, FN(dot_thunk), &args, opts);
    }
    if (!skip_same || t->FN(scal) != skip_same->FN(scal)) {
        args.scal = t->FN(scal);
        FN(bench_blas_one)("scal" SUFFIX_STR, impl, FN(scal_thunk), &args, opts);
    }
}
