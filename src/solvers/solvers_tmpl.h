/*
 * Solver template, included once per precision by solvers.c.
 *
 * The loop mirrors Saddle's adam_optimise: the gradient at the current point
 * drives the step, and trajectory entry k is the position after k steps. Run
 * with tol=0 and the same numerical gradients, the fp64 Adam path comes out
 * bit-identical to Saddle's.
 */

static REAL FN(inf_norm)(const REAL *v, size_t n) {
    REAL m = 0;
    for (size_t i = 0; i < n; i++) {
        REAL a = FABS(v[i]);
        if (a > m)
            m = a;
    }
    return m;
}

FN(stride_solver_opts) FN(stride_solver_defaults)(stride_method method) {
    FN(stride_solver_opts) o = {
        .method = method,
        .max_iters = 1000,
        .tol = (REAL)1e-8,
        .lr = (REAL)0.001,
        .momentum = (REAL)0.9,
        .rho = (REAL)0.9,
        .beta1 = (REAL)0.9,
        .beta2 = (REAL)0.999,
        .eps = (REAL)1e-8,
    };

    /* Plain SGD wants a larger step than the adaptive methods. */
    if (method == STRIDE_SGD || method == STRIDE_SGD_MOMENTUM)
        o.lr = (REAL)0.01;
    return o;
}

FN(stride_result) FN(stride_minimise)(const FN(stride_objective) *obj, REAL *x,
                                      const FN(stride_solver_opts) *opts,
                                      FN(stride_trace) trace, void *trace_user) {
    size_t n = obj->n;
    FN(stride_result) res = {.iters = 0, .grad_norm = 0, .converged = -1};

    REAL *g = malloc(n * sizeof(*g));
    REAL *s1 = calloc(n, sizeof(*s1)); /* momentum velocity / rmsprop sq avg / adam m */
    REAL *s2 = calloc(n, sizeof(*s2)); /* adam v */
    if (!g || !s1 || !s2)
        goto done;

    obj->grad(obj->ctx, x, g, n);
    if (trace)
        trace(trace_user, 0, x, g);

    for (size_t step = 1; step <= opts->max_iters; step++) {
        if (opts->tol > 0 && FN(inf_norm)(g, n) < opts->tol)
            break;

        switch (opts->method) {
        case STRIDE_SGD:
            FN(stride_sgd_step)(x, g, n, opts->lr);
            break;
        case STRIDE_SGD_MOMENTUM:
            FN(stride_sgd_momentum_step)(x, s1, g, n, opts->lr, opts->momentum);
            break;
        case STRIDE_RMSPROP:
            FN(stride_rmsprop_step)(x, s1, g, n, opts->lr, opts->rho, opts->eps);
            break;
        case STRIDE_ADAM: {
            /* Bias corrections recomputed with pow() each step, exactly the
             * way Saddle's adam_step does it, so fp64 trajectories stay
             * bit-identical. The kernel itself never sees a pow. */
            REAL bc1 = (REAL)1 - POW(opts->beta1, (REAL)step);
            REAL bc2 = (REAL)1 - POW(opts->beta2, (REAL)step);
            FN(stride_adam_step)(x, s1, s2, g, n, opts->lr, opts->beta1, opts->beta2,
                                 opts->eps, bc1, bc2);
            break;
        }
        }

        obj->grad(obj->ctx, x, g, n);
        if (trace)
            trace(trace_user, step, x, g);
        res.iters = step;
    }

    res.grad_norm = FN(inf_norm)(g, n);
    res.converged = (opts->tol > 0 && res.grad_norm < opts->tol) ? 1 : 0;

done:
    free(g);
    free(s1);
    free(s2);
    return res;
}
