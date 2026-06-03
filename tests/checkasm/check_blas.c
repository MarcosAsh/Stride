/*
 * BLAS-1 checks and benchmarks, instantiated for f32 and f64, plus a BLAS
 * self-test with deliberately broken kernels.
 */

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stride/kernels.h"
#include "checkasm.h"

static const size_t blas_check_sizes[] = {0,  1,  2,  3,  5,   7,   8,   9,   15,  16,   17,  31,
                                          32, 33, 63, 64, 65,  127, 128, 129, 255, 1000, 1537};
static const size_t blas_bench_sizes[] = {256, 4096, 65536, 1048576, 4194304};

#define NUM_BLAS_CHECK_SIZES (sizeof(blas_check_sizes) / sizeof(blas_check_sizes[0]))
#define NUM_BLAS_BENCH_SIZES (sizeof(blas_bench_sizes) / sizeof(blas_bench_sizes[0]))

#define REAL float
#define EPS FLT_EPSILON
#define SUFFIX_STR "_f32"
#define FN(name) name##_f32
#define FNT(name) name##_f32_fn
#define CMP checkasm_cmp_f32
#include "check_blas_tmpl.h"
#undef REAL
#undef EPS
#undef SUFFIX_STR
#undef FN
#undef FNT
#undef CMP

#define REAL double
#define EPS DBL_EPSILON
#define SUFFIX_STR "_f64"
#define FN(name) name##_f64
#define FNT(name) name##_f64_fn
#define CMP checkasm_cmp_f64
#include "check_blas_tmpl.h"
#undef REAL
#undef EPS
#undef SUFFIX_STR
#undef FN
#undef FNT
#undef CMP

void checkasm_check_blas(const stride_kernel_table *ref, const stride_kernel_table *new,
                         const char *impl, const checkasm_opts *opts) {
    check_blas_tables_f32(ref, new, impl, opts);
    check_blas_tables_f64(ref, new, impl, opts);
}

void checkasm_bench_blas(const stride_kernel_table *t, const stride_kernel_table *skip_same,
                         const char *impl, const checkasm_opts *opts) {
    bench_blas_tables_f32(t, skip_same, impl, opts);
    bench_blas_tables_f64(t, skip_same, impl, opts);
}

/* broken kernels for the self-test */

static void broken_axpy_tail(float *y, const float *x, size_t n, float a) {
    stride_axpy_f32(y, x, n & ~(size_t)7, a); /* drops the tail */
}

static float broken_dot_tail(const float *x, const float *y, size_t n) {
    return stride_dot_f32(x, y, n & ~(size_t)7); /* misses the last products */
}

static void broken_scal_scale(float *x, size_t n, float a) {
    stride_scal_f32(x, n, a * 1.0001f); /* wrong factor */
}

int checkasm_selftest_blas(const stride_kernel_table *ref, const checkasm_opts *opts) {
    checkasm_opts so = *opts;
    so.filter = NULL;

    /* run each broken kernel through the matching check and confirm it fails */
    int bad = 0;

    /* axpy */
    {
        int before = checkasm_failure_count();
        checkasm_set_quiet(1);
        check_axpy_f32(ref->axpy_f32, broken_axpy_tail, "selftest", &so);
        checkasm_set_quiet(0);
        int caught = checkasm_failure_count() > before;
        checkasm_set_failure_count(before);
        printf("  %-28s ... %s\n", "axpy skipped tail", caught ? "caught" : "NOT CAUGHT");
        bad |= !caught;
    }
    /* dot */
    {
        int before = checkasm_failure_count();
        checkasm_set_quiet(1);
        check_dot_f32(ref->dot_f32, broken_dot_tail, "selftest", &so);
        checkasm_set_quiet(0);
        int caught = checkasm_failure_count() > before;
        checkasm_set_failure_count(before);
        printf("  %-28s ... %s\n", "dot skipped tail", caught ? "caught" : "NOT CAUGHT");
        bad |= !caught;
    }
    /* scal */
    {
        int before = checkasm_failure_count();
        checkasm_set_quiet(1);
        check_scal_f32(ref->scal_f32, broken_scal_scale, "selftest", &so);
        checkasm_set_quiet(0);
        int caught = checkasm_failure_count() > before;
        checkasm_set_failure_count(before);
        printf("  %-28s ... %s\n", "scal wrong factor", caught ? "caught" : "NOT CAUGHT");
        bad |= !caught;
    }

    return bad;
}
