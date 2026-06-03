#ifndef STRIDE_CPU_H
#define STRIDE_CPU_H

/*
 * Runtime CPU feature detection.
 *
 * The dispatch table uses these flags to pick the best kernel. checkasm uses
 * them to know which implementations the current machine can actually run.
 */

#define STRIDE_CPU_SSE2   (1 << 0)
#define STRIDE_CPU_AVX    (1 << 1)
#define STRIDE_CPU_FMA3   (1 << 2)
#define STRIDE_CPU_AVX2   (1 << 3)
#define STRIDE_CPU_AVX512 (1 << 4) /* F + DQ + BW + VL, the practical baseline */

/* Detect the running CPU. Checks both the CPUID feature bits and that the OS
 * saves the corresponding register state (XCR0), so a flag being set means
 * the instructions are actually usable. */
int stride_cpu_flags(void);

#endif /* STRIDE_CPU_H */
