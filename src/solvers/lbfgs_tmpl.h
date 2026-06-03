/*
 * L-BFGS, one instantiation per precision.
 *
 * History is a ring buffer of mem (s, y) pairs. The two-loop recursion turns
 * the current gradient into a search direction using those pairs, the Armijo
 * backtracking line search picks the step length, and a new pair is pushed if
 * the curvature came out positive.
 *
 * dot/axpy/scal come from the kernel table when one is passed, so the inner
 * products and vector updates over the history run on the AVX2 kernels.
 */

static REAL FN(lbfgs_inf_norm)(const REAL *v, size_t n) {
    REAL m = 0;
    for (size_t i = 0; i < n; i++) {
        REAL a = v[i] < 0 ? -v[i] : v[i];
        if (a > m)
            m = a;
    }
    return m;
}

FN(stride_lbfgs_opts) FN(stride_lbfgs_defaults)(void) {
    FN(stride_lbfgs_opts) o = {
        .max_iters = 1000,
        .tol = (REAL)1e-8,
        .mem = 8,
        .c1 = (REAL)1e-4,
        .ls_decrease = (REAL)0.5,
        .ls_max = 40,
    };
    return o;
}

FN(stride_result) FN(stride_lbfgs_minimise)(const FN(stride_objective) *obj, REAL *x,
                                            const FN(stride_lbfgs_opts) *opts,
                                            const stride_kernel_table *kt, FN(stride_trace) trace,
                                            void *trace_user) {
    size_t n = obj->n;
    size_t mem = opts->mem ? opts->mem : 1;
    FN(stride_result) res = {.iters = 0, .grad_norm = 0, .converged = -1};

    FNT(stride_dot) dot = kt ? kt->FN(dot) : FN(stride_dot);
    FNT(stride_axpy) axpy = kt ? kt->FN(axpy) : FN(stride_axpy);
    FNT(stride_scal) scal = kt ? kt->FN(scal) : FN(stride_scal);

    REAL *g = malloc(n * sizeof(REAL));
    REAL *g_new = malloc(n * sizeof(REAL));
    REAL *d = malloc(n * sizeof(REAL));     /* search direction, also q/r scratch */
    REAL *x_new = malloc(n * sizeof(REAL));
    REAL *S = malloc(mem * n * sizeof(REAL));
    REAL *Y = malloc(mem * n * sizeof(REAL));
    REAL *rho = malloc(mem * sizeof(REAL));
    REAL *alpha = malloc(mem * sizeof(REAL));
    if (!g || !g_new || !d || !x_new || !S || !Y || !rho || !alpha)
        goto done;

    obj->grad(obj->ctx, x, g, n);
    if (trace)
        trace(trace_user, 0, x, g);
    REAL fx = obj->eval(obj->ctx, x, n);

    size_t hist = 0; /* stored pairs, <= mem */
    size_t head = 0; /* next write slot in the ring */

    for (size_t step = 1; step <= opts->max_iters; step++) {
        if (opts->tol > 0 && FN(lbfgs_inf_norm)(g, n) < opts->tol)
            break;

        /* Two-loop recursion: d starts as q = g. */
        memcpy(d, g, n * sizeof(REAL));

        if (hist > 0) {
            /* first loop, newest pair to oldest */
            for (size_t j = 0; j < hist; j++) {
                size_t idx = (head + mem - 1 - j) % mem;
                alpha[idx] = rho[idx] * dot(S + idx * n, d, n);
                axpy(d, Y + idx * n, n, -alpha[idx]); /* q -= alpha * y */
            }
            /* scale by gamma from the newest pair */
            size_t nw = (head + mem - 1) % mem;
            REAL sy = dot(S + nw * n, Y + nw * n, n);
            REAL yy = dot(Y + nw * n, Y + nw * n, n);
            REAL gamma = yy > 0 ? sy / yy : (REAL)1;
            scal(d, n, gamma); /* r = gamma * q */
            /* second loop, oldest pair to newest */
            for (size_t j = 0; j < hist; j++) {
                size_t idx = (head + mem - hist + j) % mem;
                REAL beta = rho[idx] * dot(Y + idx * n, d, n);
                axpy(d, S + idx * n, n, alpha[idx] - beta); /* r += (alpha - beta) * s */
            }
        }
        /* direction is -H g */
        scal(d, n, (REAL)-1);

        /* Guard against a non-descent direction (can happen from a stale
         * history); fall back to steepest descent. */
        REAL gd = dot(g, d, n);
        if (gd >= 0) {
            memcpy(d, g, n * sizeof(REAL));
            scal(d, n, (REAL)-1);
            gd = dot(g, d, n);
        }

        /* Armijo backtracking. First step is scaled by 1/|g| so the very
         * first iteration (steepest descent) does not wildly overshoot. */
        REAL t = hist == 0 ? (REAL)1 / (FN(lbfgs_inf_norm)(g, n) + (REAL)1e-12) : (REAL)1;
        REAL fx_new = fx;
        int ok = 0;
        for (size_t ls = 0; ls < opts->ls_max; ls++) {
            memcpy(x_new, x, n * sizeof(REAL));
            axpy(x_new, d, n, t); /* x_new = x + t d */
            fx_new = obj->eval(obj->ctx, x_new, n);
            if (fx_new <= fx + opts->c1 * t * gd) {
                ok = 1;
                break;
            }
            t *= opts->ls_decrease;
        }
        if (!ok)
            break; /* line search could not make progress */

        obj->grad(obj->ctx, x_new, g_new, n);

        /* push (s, y) = (x_new - x, g_new - g) if the curvature is positive */
        REAL *s_slot = S + head * n;
        REAL *y_slot = Y + head * n;
        for (size_t i = 0; i < n; i++) {
            s_slot[i] = x_new[i] - x[i];
            y_slot[i] = g_new[i] - g[i];
        }
        REAL ys = dot(y_slot, s_slot, n);
        REAL yy = dot(y_slot, y_slot, n);
        if (ys > (REAL)CURV_EPS * (yy + (REAL)1e-30)) {
            rho[head] = (REAL)1 / ys;
            head = (head + 1) % mem;
            if (hist < mem)
                hist++;
        }

        memcpy(x, x_new, n * sizeof(REAL));
        memcpy(g, g_new, n * sizeof(REAL));
        fx = fx_new;
        if (trace)
            trace(trace_user, step, x, g);
        res.iters = step;
    }

    res.grad_norm = FN(lbfgs_inf_norm)(g, n);
    res.converged = (opts->tol > 0 && res.grad_norm < opts->tol) ? 1 : 0;

done:
    free(g);
    free(g_new);
    free(d);
    free(x_new);
    free(S);
    free(Y);
    free(rho);
    free(alpha);
    return res;
}
