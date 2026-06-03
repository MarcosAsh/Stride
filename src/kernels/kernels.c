#include "stride/kernels.h"
#include "stride/cpu.h"

#include <math.h>

#if defined(__x86_64__)
#include "x86/kernels_x86.h"
#endif

/* fp32 variants */
#define REAL float
#define SQRT sqrtf
#define FN(name) name##_f32
#include "kernels_tmpl.h"
#undef REAL
#undef SQRT
#undef FN

/* fp64 variants */
#define REAL double
#define SQRT sqrt
#define FN(name) name##_f64
#include "kernels_tmpl.h"
#undef REAL
#undef SQRT
#undef FN

void stride_kernel_table_init(stride_kernel_table *t, int cpu_flags) {
    /* Portable C references. Always the starting point, and what every other
     * implementation gets checked against. */
    t->sgd_step_f32 = stride_sgd_step_f32;
    t->sgd_step_f64 = stride_sgd_step_f64;
    t->sgd_momentum_step_f32 = stride_sgd_momentum_step_f32;
    t->sgd_momentum_step_f64 = stride_sgd_momentum_step_f64;
    t->rmsprop_step_f32 = stride_rmsprop_step_f32;
    t->rmsprop_step_f64 = stride_rmsprop_step_f64;
    t->adam_step_f32 = stride_adam_step_f32;
    t->adam_step_f64 = stride_adam_step_f64;

    /* ISA-specific kernels override the C pointers when the CPU supports the
     * instructions. M2 adds the fused Adam step in AVX2; the BLAS-1 kernels
     * and the rest follow in later milestones. */
#if defined(__x86_64__)
    if (cpu_flags & STRIDE_CPU_AVX2) {
        t->adam_step_f32 = stride_adam_step_f32_avx2;
        t->adam_step_f64 = stride_adam_step_f64_avx2;
    }
#endif
    (void)cpu_flags;
}
