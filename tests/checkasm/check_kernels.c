/*
 * kernel checks and benchmarks, instantiated for f32 and f64 from the
 * template, plus the harness self test with deliberately broken kernels
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stride/kernels.h"
#include "checkasm.h"

/* sizes around the simd width boundaries (8 f32 / 4 f64 for avx2, double
 * that for avx512) plus a couple of large odd ones */
static const size_t check_sizes[] = {0,  1,  2,  3,  5,   7,   8,   9,   15,  16,   17,  31,
                                     32, 33, 63, 64, 65,  127, 128, 129, 255, 1000, 1537};

/* working set spans L1, L2, LLC and DRAM, the memory traffic story shows up
 * in the last two */
static const size_t bench_sizes[] = {256, 4096, 65536, 1048576, 4194304};

#define NUM_CHECK_SIZES (sizeof(check_sizes) / sizeof(check_sizes[0]))
#define NUM_BENCH_SIZES (sizeof(bench_sizes) / sizeof(bench_sizes[0]))

/* f32 instantiation */
#define REAL float
#define SQRT sqrtf
#define SUFFIX_STR "_f32"
#define FN(name) name##_f32
#define FNT(name) name##_f32_fn
#define CMP checkasm_cmp_f32
#include "check_kernels_tmpl.h"
#undef REAL
#undef SQRT
#undef SUFFIX_STR
#undef FN
#undef FNT
#undef CMP

/* f64 instantiation */
#define REAL double
#define SQRT sqrt
#define SUFFIX_STR "_f64"
#define FN(name) name##_f64
#define FNT(name) name##_f64_fn
#define CMP checkasm_cmp_f64
#include "check_kernels_tmpl.h"
#undef REAL
#undef SQRT
#undef SUFFIX_STR
#undef FN
#undef FNT
#undef CMP

/* combined public entry points */

void checkasm_check_kernels(const stride_kernel_table *ref, const stride_kernel_table *new,
                            const char *impl, const checkasm_opts *opts) {
    check_tables_f32(ref, new, impl, opts);
    check_tables_f64(ref, new, impl, opts);
}

void checkasm_bench_kernels(const stride_kernel_table *t, const stride_kernel_table *skip_same,
                            const char *impl, const checkasm_opts *opts) {
    bench_tables_f32(t, skip_same, impl, opts);
    bench_tables_f64(t, skip_same, impl, opts);
}

void checkasm_check_baselines(const stride_kernel_table *ref, const checkasm_opts *opts) {
    check_baselines_f32(ref, opts);
    check_baselines_f64(ref, opts);
}

void checkasm_bench_baselines(const stride_kernel_table *ref, const checkasm_opts *opts) {
    bench_baselines_f32(ref, opts);
    bench_baselines_f64(ref, opts);
}

/*
 * self test, each of these contains one of the classic asm kernel bugs and
 * the harness has to catch every one of them, otherwise its green checkmarks
 * mean nothing
 */

static void broken_arith(float *params, float *m, float *v, const float *grads, size_t n,
                         float lr, float beta1, float beta2, float eps, float bc1, float bc2) {
    /* subtly wrong constant */
    stride_adam_step_f32(params, m, v, grads, n, lr, beta1 * 1.000001f, beta2, eps, bc1, bc2);
}

static void broken_tail(float *params, float *m, float *v, const float *grads, size_t n, float lr,
                        float beta1, float beta2, float eps, float bc1, float bc2) {
    /* forgets the last n mod 8 elements, the classic simd tail bug */
    stride_adam_step_f32(params, m, v, grads, n & ~(size_t)7, lr, beta1, beta2, eps, bc1, bc2);
}

static void broken_overrun(float *params, float *m, float *v, const float *grads, size_t n,
                           float lr, float beta1, float beta2, float eps, float bc1, float bc2) {
    /* writes one element past the end, the other classic simd bug */
    stride_adam_step_f32(params, m, v, grads, n, lr, beta1, beta2, eps, bc1, bc2);
    params[n] = 0;
}

int checkasm_selftest(const stride_kernel_table *ref, const checkasm_opts *opts) {
    static const struct {
        const char *name;
        stride_adam_step_f32_fn fn;
    } broken[] = {
        {"wrong arithmetic", broken_arith},
        {"skipped tail", broken_tail},
        {"out of bounds write", broken_overrun},
    };

    /* self test ignores the kernel filter, it always runs on adam f32 */
    checkasm_opts so = *opts;
    so.filter = NULL;

    kernel_desc_f32 dr[4];
    make_descs_f32(ref, dr);

    int bad = 0;
    for (size_t i = 0; i < sizeof(broken) / sizeof(broken[0]); i++) {
        int before = checkasm_failure_count();

        checkasm_set_quiet(1);
        check_one_f32(&dr[3], "selftest", (checkasm_fn)broken[i].fn, 0, &so);
        checkasm_set_quiet(0);

        int caught = checkasm_failure_count() > before;
        checkasm_set_failure_count(before);

        printf("  %-28s ... %s\n", broken[i].name, caught ? "caught" : "NOT CAUGHT");
        if (!caught)
            bad = 1;
    }

    return bad;
}
