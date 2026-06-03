/*
 * gemv template, included once per precision by gemv.c.
 *
 * Both forms ride on the dispatched BLAS-1 kernels so they pick up the AVX2
 * dot and axpy for free. With t = NULL they fall back to the C kernels, which
 * is what the correctness tests use as their reference.
 */

void FN(stride_gemv)(const stride_kernel_table *t, const REAL *A, const REAL *x, REAL *y,
                     size_t rows, size_t cols) {
    FNT(stride_dot) dot = t ? t->FN(dot) : FN(stride_dot);
    /* y[r] = (row r of A) . x */
    for (size_t r = 0; r < rows; r++)
        y[r] = dot(A + r * cols, x, cols);
}

void FN(stride_gemv_t)(const stride_kernel_table *t, const REAL *A, const REAL *x, REAL *y,
                       size_t rows, size_t cols) {
    FNT(stride_axpy) axpy = t ? t->FN(axpy) : FN(stride_axpy);
    /* y = sum_r x[r] * (row r of A) */
    for (size_t c = 0; c < cols; c++)
        y[c] = 0;
    for (size_t r = 0; r < rows; r++)
        axpy(y, A + r * cols, cols, x[r]);
}
