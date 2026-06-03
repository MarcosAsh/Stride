/*
 * per kernel check and bench logic, templated over REAL
 * included by check_kernels.c with REAL, SQRT, SUFFIX_STR, FN, FNT and CMP defined
 *
 * every kernel gets described by a small descriptor (how many state arrays,
 * which must stay positive, what scalars it takes) plus an invoke adapter
 * that knows its signature, the generic runner does the rest
 */

/* fills, v and sq style arrays hold running averages of squares so they
 * cannot go negative */
static void FN(fill_signed)(REAL *buf, size_t n, double range) {
    for (size_t i = 0; i < n; i++)
        buf[i] = (REAL)checkasm_rand_uniform(-range, range);
}

static void FN(fill_positive)(REAL *buf, size_t n, double range) {
    for (size_t i = 0; i < n; i++)
        buf[i] = (REAL)checkasm_rand_uniform(0, range);
}

/* invoke adapters, one per kernel signature */

typedef void (*FN(invoke_fn))(checkasm_fn fn, REAL *const *state, const REAL *grads, size_t n,
                              const double *sc);

static void FN(sgd_invoke)(checkasm_fn fn, REAL *const *state, const REAL *grads, size_t n,
                           const double *sc) {
    ((FNT(stride_sgd_step))fn)(state[0], grads, n, (REAL)sc[0]);
}

static void FN(momentum_invoke)(checkasm_fn fn, REAL *const *state, const REAL *grads, size_t n,
                                const double *sc) {
    ((FNT(stride_sgd_momentum_step))fn)(state[0], state[1], grads, n, (REAL)sc[0], (REAL)sc[1]);
}

static void FN(rmsprop_invoke)(checkasm_fn fn, REAL *const *state, const REAL *grads, size_t n,
                               const double *sc) {
    ((FNT(stride_rmsprop_step))fn)(state[0], state[1], grads, n, (REAL)sc[0], (REAL)sc[1],
                                   (REAL)sc[2]);
}

static void FN(adam_invoke)(checkasm_fn fn, REAL *const *state, const REAL *grads, size_t n,
                            const double *sc) {
    ((FNT(stride_adam_step))fn)(state[0], state[1], state[2], grads, n, (REAL)sc[0], (REAL)sc[1],
                                (REAL)sc[2], (REAL)sc[3], (REAL)sc[4], (REAL)sc[5]);
}

/* kernel descriptors */

typedef struct {
    const char *name;
    checkasm_fn fn;
    FN(invoke_fn) invoke;
    int num_state;
    const char *state_names[3];
    int positive[3];
    double scalars[8];
} FN(kernel_desc);

#define NUM_KERNELS 4
#define ADAM_DESC 3

static void FN(make_descs)(const stride_kernel_table *t, FN(kernel_desc) *d) {
    /* fixed step count for the adam bias corrections, any value works as long
     * as both implementations get handed the same one */
    double bc1 = 1.0 - pow(0.9, 7.0);
    double bc2 = 1.0 - pow(0.999, 7.0);

    d[0] = (FN(kernel_desc)){
        .name = "sgd_step" SUFFIX_STR,
        .fn = (checkasm_fn)t->FN(sgd_step),
        .invoke = FN(sgd_invoke),
        .num_state = 1,
        .state_names = {"params"},
        .positive = {0},
        .scalars = {0.001},
    };
    d[1] = (FN(kernel_desc)){
        .name = "sgd_momentum_step" SUFFIX_STR,
        .fn = (checkasm_fn)t->FN(sgd_momentum_step),
        .invoke = FN(momentum_invoke),
        .num_state = 2,
        .state_names = {"params", "vel"},
        .positive = {0, 0},
        .scalars = {0.001, 0.9},
    };
    d[2] = (FN(kernel_desc)){
        .name = "rmsprop_step" SUFFIX_STR,
        .fn = (checkasm_fn)t->FN(rmsprop_step),
        .invoke = FN(rmsprop_invoke),
        .num_state = 2,
        .state_names = {"params", "sq"},
        .positive = {0, 1},
        .scalars = {0.001, 0.9, 1e-8},
    };
    d[ADAM_DESC] = (FN(kernel_desc)){
        .name = "adam_step" SUFFIX_STR,
        .fn = (checkasm_fn)t->FN(adam_step),
        .invoke = FN(adam_invoke),
        .num_state = 3,
        .state_names = {"params", "m", "v"},
        .positive = {0, 0, 1},
        .scalars = {0.001, 0.9, 0.999, 1e-8, bc1, bc2},
    };
}

/* correctness, run ref and new on identical randomised state and compare
 * everything including the guard zones and the read only gradient buffer */
static void FN(check_one)(const FN(kernel_desc) *d, const char *impl, checkasm_fn new_fn,
                          int64_t max_ulp, const checkasm_opts *opts) {
    if (opts->filter && !strstr(d->name, opts->filter))
        return;

    int fail_before = checkasm_failure_count();

    for (size_t si = 0; si < NUM_CHECK_SIZES; si++) {
        size_t n = check_sizes[si];

        /* offset 1 misaligns the views, kernels must not care */
        for (int offset = 0; offset <= 1; offset++) {
            size_t alloc_n = n + (size_t)offset;
            REAL *state_ref[3] = {0}, *state_new[3] = {0};
            REAL *view_ref[3], *view_new[3];

            checkasm_context(d->name, impl, n, offset);

            for (int s = 0; s < d->num_state; s++) {
                state_ref[s] = checkasm_alloc(sizeof(REAL), alloc_n);
                state_new[s] = checkasm_alloc(sizeof(REAL), alloc_n);
                if (d->positive[s])
                    FN(fill_positive)(state_ref[s], alloc_n, 1.0);
                else
                    FN(fill_signed)(state_ref[s], alloc_n, 5.0);
                memcpy(state_new[s], state_ref[s], alloc_n * sizeof(REAL));
                view_ref[s] = state_ref[s] + offset;
                view_new[s] = state_new[s] + offset;
            }

            REAL *grads = checkasm_alloc(sizeof(REAL), alloc_n);
            REAL *grads_copy = checkasm_alloc(sizeof(REAL), alloc_n);
            FN(fill_signed)(grads, alloc_n, 10.0);
            memcpy(grads_copy, grads, alloc_n * sizeof(REAL));

            d->invoke(d->fn, view_ref, grads + offset, n, d->scalars);
            d->invoke(new_fn, view_new, grads + offset, n, d->scalars);

            /* compare the whole allocation so stray writes outside the view
             * get caught by the value diff and not just the guards */
            for (int s = 0; s < d->num_state; s++) {
                CMP(state_ref[s], state_new[s], alloc_n, max_ulp, d->state_names[s]);
                checkasm_check_guards(state_new[s], sizeof(REAL), alloc_n, d->state_names[s]);
            }

            checkasm_check_guards(grads, sizeof(REAL), alloc_n, "grads");
            CMP(grads, grads_copy, alloc_n, 0, "grads (kernel must not write its input)");

            for (int s = 0; s < d->num_state; s++) {
                checkasm_free(state_ref[s], sizeof(REAL), alloc_n);
                checkasm_free(state_new[s], sizeof(REAL), alloc_n);
            }
            checkasm_free(grads, sizeof(REAL), alloc_n);
            checkasm_free(grads_copy, sizeof(REAL), alloc_n);
        }
    }

    checkasm_report_check(d->name, impl, checkasm_failure_count() != fail_before);
}

/* benchmarking */

typedef struct {
    checkasm_fn fn;
    FN(invoke_fn) invoke;
    REAL *state[3];
    const REAL *grads;
    size_t n;
    const double *scalars;
} FN(bench_args);

static void FN(bench_thunk)(void *p) {
    FN(bench_args) *a = p;
    a->invoke(a->fn, a->state, a->grads, a->n, a->scalars);
}

static void FN(bench_one)(const FN(kernel_desc) *d, const char *impl, checkasm_fn fn,
                          const checkasm_opts *opts) {
    if (opts->filter && !strstr(d->name, opts->filter))
        return;

    for (size_t si = 0; si < NUM_BENCH_SIZES; si++) {
        size_t n = bench_sizes[si];

        REAL *state[3] = {0};
        for (int s = 0; s < d->num_state; s++) {
            state[s] = checkasm_alloc(sizeof(REAL), n);
            if (d->positive[s])
                FN(fill_positive)(state[s], n, 1.0);
            else
                FN(fill_signed)(state[s], n, 5.0);
        }
        REAL *grads = checkasm_alloc(sizeof(REAL), n);
        FN(fill_signed)(grads, n, 10.0);

        FN(bench_args) args = {
            .fn = fn,
            .invoke = d->invoke,
            .state = {state[0], state[1], state[2]},
            .grads = grads,
            .n = n,
            .scalars = d->scalars,
        };

        /* scale the run count down as n grows so total bench time stays sane */
        long runs = (long)opts->bench_runs * 4096 / (long)(n ? n : 1);
        if (runs < 4)
            runs = 4;
        if (runs > 2048)
            runs = 2048;

        uint64_t ticks = checkasm_bench(FN(bench_thunk), &args, (int)runs);
        checkasm_report_bench(d->name, impl, n, ticks);

        for (int s = 0; s < d->num_state; s++)
            checkasm_free(state[s], sizeof(REAL), n);
        checkasm_free(grads, sizeof(REAL), n);
    }
}

/* the multi pass adam an array framework performs, one pass per expression
 * with temporaries for the bias corrected moments, this is benchmark baseline
 * number one from the plan and it must produce bit identical results to the
 * fused kernel since the per element arithmetic is the same
 *
 * reusing static temps across calls is generous to it, numpy would allocate
 * fresh ones every step */
static void FN(adam_step_multipass)(REAL *params, REAL *m, REAL *v, const REAL *grads, size_t n,
                                    REAL lr, REAL beta1, REAL beta2, REAL eps, REAL bc1,
                                    REAL bc2) {
    static REAL *m_hat, *v_hat;
    static size_t cap;

    if (n > cap) {
        free(m_hat);
        free(v_hat);
        m_hat = malloc(n * sizeof(*m_hat));
        v_hat = malloc(n * sizeof(*v_hat));
        if (!m_hat || !v_hat)
            abort();
        cap = n;
    }

    for (size_t i = 0; i < n; i++)
        m[i] = beta1 * m[i] + ((REAL)1 - beta1) * grads[i];
    for (size_t i = 0; i < n; i++)
        v[i] = beta2 * v[i] + ((REAL)1 - beta2) * grads[i] * grads[i];
    for (size_t i = 0; i < n; i++)
        m_hat[i] = m[i] / bc1;
    for (size_t i = 0; i < n; i++)
        v_hat[i] = v[i] / bc2;
    for (size_t i = 0; i < n; i++)
        params[i] -= lr * m_hat[i] / (SQRT(v_hat[i]) + eps);
}

/* entry points for this precision, the public combined ones live in
 * check_kernels.c */

static void FN(check_tables)(const stride_kernel_table *ref, const stride_kernel_table *new,
                             const char *impl, const checkasm_opts *opts) {
    FN(kernel_desc) dr[NUM_KERNELS], dn[NUM_KERNELS];
    FN(make_descs)(ref, dr);
    FN(make_descs)(new, dn);

    for (int i = 0; i < NUM_KERNELS; i++)
        if (dn[i].fn != dr[i].fn)
            FN(check_one)(&dr[i], impl, dn[i].fn, 0, opts);
}

static void FN(bench_tables)(const stride_kernel_table *t, const stride_kernel_table *skip_same,
                             const char *impl, const checkasm_opts *opts) {
    FN(kernel_desc) dt[NUM_KERNELS], ds[NUM_KERNELS];
    FN(make_descs)(t, dt);
    if (skip_same)
        FN(make_descs)(skip_same, ds);

    for (int i = 0; i < NUM_KERNELS; i++) {
        if (skip_same && dt[i].fn == ds[i].fn)
            continue;
        FN(bench_one)(&dt[i], impl, dt[i].fn, opts);
    }
}

static void FN(check_baselines)(const stride_kernel_table *ref, const checkasm_opts *opts) {
    FN(kernel_desc) dr[NUM_KERNELS];
    FN(make_descs)(ref, dr);
    FN(check_one)(&dr[ADAM_DESC], "multipass_c", (checkasm_fn)FN(adam_step_multipass), 0, opts);
}

static void FN(bench_baselines)(const stride_kernel_table *ref, const checkasm_opts *opts) {
    FN(kernel_desc) dr[NUM_KERNELS];
    FN(make_descs)(ref, dr);
    FN(bench_one)(&dr[ADAM_DESC], "multipass_c", (checkasm_fn)FN(adam_step_multipass), opts);
}

#undef NUM_KERNELS
#undef ADAM_DESC
