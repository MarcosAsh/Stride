#ifndef STRIDE_CHECKASM_H
#define STRIDE_CHECKASM_H

#include <stddef.h>
#include <stdint.h>

#include "stride/kernels.h"

/*
 * checkasm-style harness for Stride's kernels, modelled on FFmpeg's.
 *
 * Every kernel implementation (C baseline, asm) is checked against the
 * portable C reference on randomised inputs across edge-case sizes and
 * misaligned pointers, with canary guards around every buffer to catch
 * out-of-bounds writes. Implementations that pass get benchmarked with
 * serialised TSC reads on a pinned P-core.
 */

/* Generic kernel function pointer; cast to the right type at the call site. */
typedef void (*checkasm_fn)(void);

/* Options parsed from the command line. */
typedef struct {
    int bench;          /* run benchmarks after correctness checks */
    int bench_runs;     /* base timing repetitions (scaled down for large n) */
    int selftest;       /* run the harness self-test (broken kernels must be caught) */
    const char *filter; /* only touch kernels whose name contains this substring */
    const char *csv;    /* write benchmark results to this CSV file */
    int core;           /* core to pin to; -1 disables pinning */
    uint64_t seed;
} checkasm_opts;

/* ---- failure reporting ---- */

/* Context that failure messages get prefixed with. */
void checkasm_context(const char *kernel, const char *impl, size_t n, int offset);

/* Record a failure against the current context. */
void checkasm_fail(const char *fmt, ...);

int checkasm_failure_count(void);
void checkasm_set_failure_count(int count);
/* Suppress failure printing (used by the self-test, which expects failures). */
void checkasm_set_quiet(int quiet);

/* ---- deterministic PRNG ---- */

void checkasm_seed(uint64_t seed);
uint64_t checkasm_rand(void);
double checkasm_rand_uniform(double lo, double hi);

/* ---- guarded, aligned buffers ---- */

/* 64-byte aligned allocation of n elements with canary zones on both sides. */
void *checkasm_alloc(size_t elem_size, size_t n);
void checkasm_free(void *p, size_t elem_size, size_t n);
/* Verify the canaries; reports a failure and returns 0 if they were written. */
int checkasm_check_guards(const void *p, size_t elem_size, size_t n, const char *what);

/* ---- comparison ---- */

/* ULP distance between two values; huge sentinel for NaN mismatches. */
int64_t checkasm_ulp_f32(float a, float b);
int64_t checkasm_ulp_f64(double a, double b);

/* Element-wise comparison; max_ulp 0 demands bit-exactness. Reports at most
 * one failure per call (with a summary of how many elements differed) and
 * returns the mismatch count. */
size_t checkasm_cmp_f32(const float *ref, const float *new, size_t n, int64_t max_ulp,
                        const char *what);
size_t checkasm_cmp_f64(const double *ref, const double *new, size_t n, int64_t max_ulp,
                        const char *what);

/* ---- timing ---- */

/* Serialised TSC read. This counts TSC reference ticks, not core clock
 * cycles; comparisons between implementations measured the same way are
 * still apples to apples. */
uint64_t checkasm_read_time(void);

/* Minimum ticks per thunk(arg) call: 4 calls per measurement, best of runs. */
uint64_t checkasm_bench(void (*thunk)(void *), void *arg, int runs);

/* Record + print one benchmark measurement. */
void checkasm_report_bench(const char *kernel, const char *impl, size_t n, uint64_t ticks);

/* Print the speedup-vs-C summary table and write the CSV file if requested. */
void checkasm_finish_bench(const checkasm_opts *opts);

/* ---- status lines ---- */

/* Per-(kernel, impl) progress line: "  adam_step_f32 [multipass_c] ... OK". */
void checkasm_report_check(const char *kernel, const char *impl, int failed);

/* ---- kernel checks (check_kernels.c) ---- */

/* Check every kernel in `new` that differs from `ref`. */
void checkasm_check_kernels(const stride_kernel_table *ref, const stride_kernel_table *new,
                            const char *impl, const checkasm_opts *opts);

/* Benchmark every kernel in `t`; entries whose pointer matches `skip_same`
 * (when non-NULL) are skipped so an ISA level only reports what it adds. */
void checkasm_bench_kernels(const stride_kernel_table *t, const stride_kernel_table *skip_same,
                            const char *impl, const checkasm_opts *opts);

/* Check + benchmark the C baselines (multi-pass Adam) against the fused refs. */
void checkasm_check_baselines(const stride_kernel_table *ref, const checkasm_opts *opts);
void checkasm_bench_baselines(const stride_kernel_table *ref, const checkasm_opts *opts);

/* ---- BLAS-1 checks (check_blas.c) ---- */

void checkasm_check_blas(const stride_kernel_table *ref, const stride_kernel_table *new,
                         const char *impl, const checkasm_opts *opts);
void checkasm_bench_blas(const stride_kernel_table *t, const stride_kernel_table *skip_same,
                         const char *impl, const checkasm_opts *opts);
int checkasm_selftest_blas(const stride_kernel_table *ref, const checkasm_opts *opts);

/* Harness self-test: deliberately broken kernels must be caught.
 * Returns 0 if everything was caught, 1 otherwise. */
int checkasm_selftest(const stride_kernel_table *ref, const checkasm_opts *opts);

#endif /* STRIDE_CHECKASM_H */
