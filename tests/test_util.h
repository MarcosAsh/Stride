#ifndef STRIDE_TEST_UTIL_H
#define STRIDE_TEST_UTIL_H

#include <stdio.h>

/*
 * Minimal test scaffolding shared by the M0 tests. The M1 checkasm-style
 * harness replaces this for kernel work; these stay for solver-level tests.
 */

static int test_failures = 0;

#define CHECK(cond, ...)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            test_failures++;                                                 \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                      \
            printf(__VA_ARGS__);                                             \
            printf("\n");                                                    \
        }                                                                    \
    } while (0)

#define TEST_DONE(name)                                                      \
    do {                                                                     \
        if (test_failures == 0)                                              \
            printf("%s: all passed\n", name);                                \
        else                                                                 \
            printf("%s: %d failure(s)\n", name, test_failures);              \
        return test_failures ? 1 : 0;                                        \
    } while (0)

/* Deterministic LCG so test inputs are reproducible. */
static unsigned long long test_rng_state = 0x123456789abcdefULL;

static inline double rand_uniform(double lo, double hi) {
    test_rng_state = test_rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u = (double)(test_rng_state >> 11) / (double)(1ULL << 53);
    return lo + u * (hi - lo);
}

#endif /* STRIDE_TEST_UTIL_H */
