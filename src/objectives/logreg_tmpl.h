/*
 * Logistic regression eval and grad, one instantiation per precision.
 *
 * Both go through gemv, so they pick up the AVX2 dot and axpy when the ctx
 * carries a kernel table. The scratch (z = Xw, r = p - y) lives in the ctx;
 * eval and grad cast away const to use it, which is the usual move for a
 * stateless-looking objective that needs working memory.
 */

/* log(1 + exp(z)), arranged so neither exp overflows */
static REAL FN(softplus)(REAL z) {
    if (z > 0)
        return z + LOG1P(EXP(-z));
    return LOG1P(EXP(z));
}

static REAL FN(sigmoid)(REAL z) {
    if (z >= 0) {
        REAL e = EXP(-z);
        return (REAL)1 / ((REAL)1 + e);
    }
    REAL e = EXP(z);
    return e / ((REAL)1 + e);
}

static REAL FN(logreg_eval)(const void *vctx, const REAL *w, size_t n) {
    FN(stride_logreg_ctx) *c = (FN(stride_logreg_ctx) *)vctx;
    (void)n;

    FN(stride_gemv)(c->kt, c->X, w, c->z, c->m, c->n); /* z = X w */

    REAL loss = 0;
    for (size_t i = 0; i < c->m; i++)
        loss += FN(softplus)(c->z[i]) - c->y[i] * c->z[i];
    loss /= (REAL)c->m;

    if (c->l2 > 0) {
        REAL s = 0;
        for (size_t j = 0; j < c->n; j++)
            s += w[j] * w[j];
        loss += (REAL)0.5 * c->l2 * s;
    }
    return loss;
}

static void FN(logreg_grad)(const void *vctx, const REAL *w, REAL *g, size_t n) {
    FN(stride_logreg_ctx) *c = (FN(stride_logreg_ctx) *)vctx;
    (void)n;

    FN(stride_gemv)(c->kt, c->X, w, c->z, c->m, c->n); /* z = X w */
    for (size_t i = 0; i < c->m; i++)
        c->r[i] = FN(sigmoid)(c->z[i]) - c->y[i]; /* r = p - y */

    FN(stride_gemv_t)(c->kt, c->X, c->r, g, c->m, c->n); /* g = X^T r */

    REAL invm = (REAL)1 / (REAL)c->m;
    for (size_t j = 0; j < c->n; j++)
        g[j] = g[j] * invm + c->l2 * w[j];
}

FN(stride_objective) FN(stride_logreg)(const FN(stride_logreg_ctx) *ctx) {
    FN(stride_objective) obj = {
        .n = ctx->n,
        .eval = FN(logreg_eval),
        .grad = FN(logreg_grad),
        .ctx = ctx,
    };
    return obj;
}
