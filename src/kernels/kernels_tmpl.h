/*
 * Update kernel template. kernels.c includes this twice, once with REAL=float
 * and once with REAL=double, to get both precisions out of one body.
 *
 * Every kernel is a single streaming pass. No branching in the loop, no
 * allocation, no libm beyond sqrt (which inlines at -O2). The Adam arithmetic
 * follows Saddle's adam_step term for term, so the two stay bit-identical in
 * fp64.
 */

void FN(stride_sgd_step)(REAL *params, const REAL *grads, size_t n, REAL lr) {
    for (size_t i = 0; i < n; i++)
        params[i] -= lr * grads[i];
}

void FN(stride_sgd_momentum_step)(REAL *params, REAL *vel, const REAL *grads,
                                  size_t n, REAL lr, REAL momentum) {
    for (size_t i = 0; i < n; i++) {
        vel[i] = momentum * vel[i] + grads[i];
        params[i] -= lr * vel[i];
    }
}

void FN(stride_rmsprop_step)(REAL *params, REAL *sq, const REAL *grads,
                             size_t n, REAL lr, REAL rho, REAL eps) {
    for (size_t i = 0; i < n; i++) {
        REAL g = grads[i];

        sq[i] = rho * sq[i] + ((REAL)1 - rho) * g * g;
        params[i] -= lr * g / (SQRT(sq[i]) + eps);
    }
}

void FN(stride_adam_step)(REAL *params, REAL *m, REAL *v, const REAL *grads,
                          size_t n, REAL lr, REAL beta1, REAL beta2, REAL eps,
                          REAL bc1, REAL bc2) {
    for (size_t i = 0; i < n; i++) {
        REAL g = grads[i];

        /* Biased first and second moment estimates. */
        m[i] = beta1 * m[i] + ((REAL)1 - beta1) * g;
        v[i] = beta2 * v[i] + ((REAL)1 - beta2) * g * g;

        /* Bias-corrected estimates. */
        REAL m_hat = m[i] / bc1;
        REAL v_hat = v[i] / bc2;

        /* Parameter update. */
        params[i] -= lr * m_hat / (SQRT(v_hat) + eps);
    }
}

void FN(stride_axpy)(REAL *y, const REAL *x, size_t n, REAL a) {
    for (size_t i = 0; i < n; i++)
        y[i] += a * x[i];
}

/* Plain sequential accumulation. This is the reference the lane-split asm gets
 * measured against, not matched bit for bit. */
REAL FN(stride_dot)(const REAL *x, const REAL *y, size_t n) {
    REAL acc = 0;
    for (size_t i = 0; i < n; i++)
        acc += x[i] * y[i];
    return acc;
}

void FN(stride_scal)(REAL *x, size_t n, REAL a) {
    for (size_t i = 0; i < n; i++)
        x[i] *= a;
}
