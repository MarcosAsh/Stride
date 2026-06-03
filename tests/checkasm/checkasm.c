/*
 * checkasm driver: CLI, core pinning, PRNG, guarded buffers, ULP comparison,
 * TSC timing, reporting. The per-kernel test logic lives in check_kernels.c.
 */

#define _GNU_SOURCE

#include <math.h>
#include <sched.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stride/cpu.h"
#include "stride/kernels.h"
#include "checkasm.h"

/* ---- failure reporting ---- */

static struct {
    char kernel[64];
    char impl[64];
    size_t n;
    int offset;
} ctx;

static int failures;
static int quiet;

void checkasm_context(const char *kernel, const char *impl, size_t n, int offset) {
    snprintf(ctx.kernel, sizeof(ctx.kernel), "%s", kernel);
    snprintf(ctx.impl, sizeof(ctx.impl), "%s", impl);
    ctx.n = n;
    ctx.offset = offset;
}

void checkasm_fail(const char *fmt, ...) {
    failures++;
    if (quiet)
        return;

    va_list args;
    printf("FAIL %s [%s] n=%zu offset=%d: ", ctx.kernel, ctx.impl, ctx.n, ctx.offset);
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    printf("\n");
}

int checkasm_failure_count(void) {
    return failures;
}

void checkasm_set_failure_count(int count) {
    failures = count;
}

void checkasm_set_quiet(int q) {
    quiet = q;
}

void checkasm_report_check(const char *kernel, const char *impl, int failed) {
    /* the self test runs with quiet on and prints its own lines */
    if (quiet)
        return;
    printf("  %-28s [%s] ... %s\n", kernel, impl, failed ? "FAILED" : "OK");
}

/* ---- PRNG: splitmix64, deterministic and reseedable ---- */

static uint64_t rng_state;

void checkasm_seed(uint64_t seed) {
    rng_state = seed;
}

uint64_t checkasm_rand(void) {
    rng_state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = rng_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

double checkasm_rand_uniform(double lo, double hi) {
    double u = (double)(checkasm_rand() >> 11) / (double)(1ULL << 53);
    return lo + u * (hi - lo);
}

/* ---- guarded buffers ---- */

#define GUARD_BYTES 64 /* multiple of the 64-byte alignment */
#define CANARY 0xA5

static size_t alloc_size(size_t elem_size, size_t n) {
    size_t total = GUARD_BYTES + n * elem_size + GUARD_BYTES;
    return (total + 63) & ~(size_t)63;
}

void *checkasm_alloc(size_t elem_size, size_t n) {
    size_t total = alloc_size(elem_size, n);
    uint8_t *base = aligned_alloc(64, total);
    if (!base) {
        fprintf(stderr, "checkasm: out of memory (%zu bytes)\n", total);
        exit(2);
    }
    /* Canary everywhere; the data region gets overwritten by the fills. */
    memset(base, CANARY, total);
    return base + GUARD_BYTES;
}

void checkasm_free(void *p, size_t elem_size, size_t n) {
    (void)elem_size;
    (void)n;
    if (p)
        free((uint8_t *)p - GUARD_BYTES);
}

int checkasm_check_guards(const void *p, size_t elem_size, size_t n, const char *what) {
    const uint8_t *base = (const uint8_t *)p - GUARD_BYTES;
    size_t total = alloc_size(elem_size, n);
    size_t data_end = GUARD_BYTES + n * elem_size;

    for (size_t i = 0; i < GUARD_BYTES; i++) {
        if (base[i] != CANARY) {
            checkasm_fail("%s: write before the start of the buffer", what);
            return 0;
        }
    }
    for (size_t i = data_end; i < total; i++) {
        if (base[i] != CANARY) {
            checkasm_fail("%s: write past the end of the buffer", what);
            return 0;
        }
    }
    return 1;
}

/* ---- ULP comparison ---- */

/* Map float bit patterns onto a monotonic integer scale so that adjacent
 * representable values differ by 1, across the +/-0 boundary too. */
static int64_t ordered_f32(float f) {
    int32_t i;
    memcpy(&i, &f, sizeof(i));
    return i < 0 ? -(int64_t)(i & 0x7FFFFFFF) : (int64_t)i;
}

static int64_t ordered_f64(double d) {
    int64_t i;
    memcpy(&i, &d, sizeof(i));
    return i < 0 ? -(i & 0x7FFFFFFFFFFFFFFF) : i;
}

int64_t checkasm_ulp_f32(float a, float b) {
    if (a == b)
        return 0;
    if (isnan(a) || isnan(b))
        return INT64_MAX;
    int64_t d = ordered_f32(a) - ordered_f32(b);
    return d < 0 ? -d : d;
}

int64_t checkasm_ulp_f64(double a, double b) {
    if (a == b)
        return 0;
    if (isnan(a) || isnan(b))
        return INT64_MAX;
    int64_t oa = ordered_f64(a), ob = ordered_f64(b);
    uint64_t d = oa > ob ? (uint64_t)oa - (uint64_t)ob : (uint64_t)ob - (uint64_t)oa;
    return d > INT64_MAX ? INT64_MAX : (int64_t)d;
}

size_t checkasm_cmp_f32(const float *ref, const float *new, size_t n, int64_t max_ulp,
                        const char *what) {
    size_t bad = 0, first = 0;
    int64_t worst = 0;
    for (size_t i = 0; i < n; i++) {
        int64_t d = checkasm_ulp_f32(ref[i], new[i]);
        if (d > max_ulp) {
            if (!bad)
                first = i;
            if (d > worst)
                worst = d;
            bad++;
        }
    }
    if (bad)
        checkasm_fail("%s: %zu of %zu elements differ (first at [%zu]: %.9g vs %.9g, worst %lld ulp)",
                      what, bad, n, first, (double)ref[first], (double)new[first],
                      (long long)worst);
    return bad;
}

size_t checkasm_cmp_f64(const double *ref, const double *new, size_t n, int64_t max_ulp,
                        const char *what) {
    size_t bad = 0, first = 0;
    int64_t worst = 0;
    for (size_t i = 0; i < n; i++) {
        int64_t d = checkasm_ulp_f64(ref[i], new[i]);
        if (d > max_ulp) {
            if (!bad)
                first = i;
            if (d > worst)
                worst = d;
            bad++;
        }
    }
    if (bad)
        checkasm_fail("%s: %zu of %zu elements differ (first at [%zu]: %.17g vs %.17g, worst %lld ulp)",
                      what, bad, n, first, ref[first], new[first], (long long)worst);
    return bad;
}

/* ---- timing ---- */

uint64_t checkasm_read_time(void) {
    uint32_t hi, lo;
    __asm__ volatile("lfence\n\t"
                     "rdtsc"
                     : "=a"(lo), "=d"(hi)
                     :
                     : "memory");
    return ((uint64_t)hi << 32) | lo;
}

uint64_t checkasm_bench(void (*thunk)(void *), void *arg, int runs) {
    /* Warm up: caches, branch predictors, and the core's clock. */
    thunk(arg);
    thunk(arg);

    uint64_t best = UINT64_MAX;
    for (int i = 0; i < runs; i++) {
        uint64_t t0 = checkasm_read_time();
        thunk(arg);
        thunk(arg);
        thunk(arg);
        thunk(arg);
        uint64_t t1 = checkasm_read_time();
        uint64_t d = (t1 - t0) / 4;
        if (d < best)
            best = d;
    }
    return best;
}

/* ---- benchmark recording ---- */

typedef struct {
    char kernel[64];
    char impl[32];
    size_t n;
    uint64_t ticks;
} bench_result;

static bench_result *bench_results;
static size_t num_bench, cap_bench;

void checkasm_report_bench(const char *kernel, const char *impl, size_t n, uint64_t ticks) {
    if (num_bench == cap_bench) {
        cap_bench = cap_bench ? cap_bench * 2 : 64;
        bench_results = realloc(bench_results, cap_bench * sizeof(*bench_results));
        if (!bench_results) {
            fprintf(stderr, "checkasm: out of memory\n");
            exit(2);
        }
    }

    bench_result *r = &bench_results[num_bench++];
    snprintf(r->kernel, sizeof(r->kernel), "%s", kernel);
    snprintf(r->impl, sizeof(r->impl), "%s", impl);
    r->n = n;
    r->ticks = ticks;

    printf("  %-28s %-14s n=%9zu: %12llu ticks  %8.2f ticks/elem\n", kernel, impl, n,
           (unsigned long long)ticks, n ? (double)ticks / (double)n : 0.0);
}

void checkasm_finish_bench(const checkasm_opts *opts) {
    /* Speedup summary: every non-"c" measurement against the matching "c" one. */
    int header = 0;
    for (size_t i = 0; i < num_bench; i++) {
        const bench_result *r = &bench_results[i];
        if (!strcmp(r->impl, "c"))
            continue;
        for (size_t j = 0; j < num_bench; j++) {
            const bench_result *c = &bench_results[j];
            if (strcmp(c->impl, "c") || strcmp(c->kernel, r->kernel) || c->n != r->n)
                continue;
            if (!header) {
                printf("\nspeedup vs c:\n");
                header = 1;
            }
            printf("  %-28s %-14s n=%9zu: %.2fx\n", r->kernel, r->impl, r->n,
                   (double)c->ticks / (double)r->ticks);
            break;
        }
    }

    if (opts->csv && num_bench) {
        FILE *f = fopen(opts->csv, "w");
        if (!f) {
            fprintf(stderr, "checkasm: cannot write %s\n", opts->csv);
            return;
        }
        fprintf(f, "kernel,impl,n,ticks,ticks_per_elem\n");
        for (size_t i = 0; i < num_bench; i++) {
            const bench_result *r = &bench_results[i];
            fprintf(f, "%s,%s,%zu,%llu,%.4f\n", r->kernel, r->impl, r->n,
                    (unsigned long long)r->ticks, r->n ? (double)r->ticks / (double)r->n : 0.0);
        }
        fclose(f);
        printf("\nwrote %s (%zu measurements)\n", opts->csv, num_bench);
    }
}

/* ---- core pinning ---- */

/* On Intel hybrid CPUs sysfs lists the P-cores; default to the first one. */
static int default_core(void) {
    FILE *f = fopen("/sys/devices/cpu_core/cpus", "r");
    if (f) {
        int first = -1;
        if (fscanf(f, "%d", &first) != 1)
            first = -1;
        fclose(f);
        if (first >= 0)
            return first;
    }
    return 0;
}

static int pin_to_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    return sched_setaffinity(0, sizeof(set), &set);
}

/* ---- main ---- */

static void usage(const char *argv0) {
    printf("usage: %s [options]\n"
           "  --bench           run benchmarks after correctness checks\n"
           "  --bench-runs=N    base timing repetitions (default 64)\n"
           "  --csv=FILE        write benchmark results to FILE\n"
           "  --filter=STR      only kernels whose name contains STR\n"
           "  --core=N          pin to core N (default: first P-core); -1 disables\n"
           "  --seed=N          PRNG seed (default 1)\n"
           "  --selftest        verify the harness catches broken kernels\n",
           argv0);
}

int main(int argc, char **argv) {
    checkasm_opts opts = {
        .bench = 0,
        .bench_runs = 64,
        .selftest = 0,
        .filter = NULL,
        .csv = NULL,
        .core = default_core(),
        .seed = 1,
    };

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bench")) {
            opts.bench = 1;
        } else if (!strncmp(argv[i], "--bench-runs=", 13)) {
            opts.bench_runs = atoi(argv[i] + 13);
        } else if (!strncmp(argv[i], "--csv=", 6)) {
            opts.csv = argv[i] + 6;
        } else if (!strncmp(argv[i], "--filter=", 9)) {
            opts.filter = argv[i] + 9;
        } else if (!strncmp(argv[i], "--core=", 7)) {
            opts.core = atoi(argv[i] + 7);
        } else if (!strncmp(argv[i], "--seed=", 7)) {
            opts.seed = strtoull(argv[i] + 7, NULL, 0);
        } else if (!strcmp(argv[i], "--selftest")) {
            opts.selftest = 1;
        } else {
            usage(argv[0]);
            return !!strcmp(argv[i], "--help");
        }
    }

    /* Pin before anything else so even the correctness checks run on one
     * core; on a hybrid CPU letting the scheduler bounce the process between
     * P- and E-cores makes benchmark numbers meaningless. */
    if (opts.core >= 0) {
        if (pin_to_core(opts.core))
            printf("warning: could not pin to core %d\n", opts.core);
        else
            printf("pinned to core %d\n", opts.core);
    }

    int cpu = stride_cpu_flags();
    printf("CPU flags:%s%s%s%s%s\n", cpu & STRIDE_CPU_SSE2 ? " sse2" : "",
           cpu & STRIDE_CPU_AVX ? " avx" : "", cpu & STRIDE_CPU_FMA3 ? " fma3" : "",
           cpu & STRIDE_CPU_AVX2 ? " avx2" : "", cpu & STRIDE_CPU_AVX512 ? " avx512" : "");
    printf("seed: %llu\n", (unsigned long long)opts.seed);

    checkasm_seed(opts.seed);

    stride_kernel_table ref;
    stride_kernel_table_init(&ref, 0);

    /* Self-test: the harness has to be able to catch broken kernels before
     * its passes mean anything. */
    if (opts.selftest) {
        printf("\nself-test (every broken kernel must be caught):\n");
        int st = checkasm_selftest(&ref, &opts);
        st |= checkasm_selftest_blas(&ref, &opts);
        if (st) {
            printf("\ncheckasm: SELF-TEST FAILED\n");
            return 1;
        }
    }

    /* Correctness: C baselines (multi-pass Adam vs fused). */
    printf("\nchecking C baselines against the fused references:\n");
    checkasm_check_baselines(&ref, &opts);

    /* Correctness: ISA implementation levels supported by this CPU. */
    static const struct {
        const char *name;
        int flags;
    } levels[] = {
        {"avx2", STRIDE_CPU_AVX | STRIDE_CPU_FMA3 | STRIDE_CPU_AVX2},
        {"avx512", STRIDE_CPU_AVX | STRIDE_CPU_FMA3 | STRIDE_CPU_AVX2 | STRIDE_CPU_AVX512},
    };

    for (size_t l = 0; l < sizeof(levels) / sizeof(levels[0]); l++) {
        if ((cpu & levels[l].flags) != levels[l].flags)
            continue;

        stride_kernel_table t;
        stride_kernel_table_init(&t, levels[l].flags);

        /* Only report a level if it actually adds implementations. */
        if (!memcmp(&t, &ref, sizeof(t)))
            continue;

        printf("\nchecking %s implementations:\n", levels[l].name);
        checkasm_check_kernels(&ref, &t, levels[l].name, &opts);
        checkasm_check_blas(&ref, &t, levels[l].name, &opts);
    }

    /* Benchmarks. */
    if (opts.bench) {
        printf("\nbenchmarks (TSC ticks, best of repeated runs):\n");
        checkasm_bench_kernels(&ref, NULL, "c", &opts);
        checkasm_bench_baselines(&ref, &opts);
        checkasm_bench_blas(&ref, NULL, "c", &opts);

        for (size_t l = 0; l < sizeof(levels) / sizeof(levels[0]); l++) {
            if ((cpu & levels[l].flags) != levels[l].flags)
                continue;
            stride_kernel_table t;
            stride_kernel_table_init(&t, levels[l].flags);
            if (!memcmp(&t, &ref, sizeof(t)))
                continue;
            checkasm_bench_kernels(&t, &ref, levels[l].name, &opts);
            checkasm_bench_blas(&t, &ref, levels[l].name, &opts);
        }

        checkasm_finish_bench(&opts);
    }

    if (checkasm_failure_count()) {
        printf("\ncheckasm: %d FAILURE(S)\n", checkasm_failure_count());
        printf("reproduce with --seed=%llu\n", (unsigned long long)opts.seed);
        return 1;
    }

    printf("\ncheckasm: all checks passed\n");
    return 0;
}
