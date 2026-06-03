#include "stride/cpu.h"

#include <stdint.h>

#if defined(__x86_64__) || defined(__i386__)

#include <cpuid.h>

/* xgetbv encoded as bytes so no -mxsave compile flag is needed; the
 * instruction only runs after the OSXSAVE cpuid bit confirms it exists. */
static uint64_t xgetbv(uint32_t index) {
    uint32_t eax, edx;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(index));
    return ((uint64_t)edx << 32) | eax;
}

int stride_cpu_flags(void) {
    int flags = 0;
    unsigned eax, ebx, ecx, edx;
    unsigned max_leaf;

    if (!__get_cpuid(0, &max_leaf, &ebx, &ecx, &edx))
        return 0;

    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx))
        return 0;

    if (edx & (1u << 26))
        flags |= STRIDE_CPU_SSE2;

    /* AVX and wider need the OS to save the extended register state:
     * OSXSAVE tells us xgetbv works, XCR0 tells us what the OS saves. */
    int osxsave = ecx & (1u << 27);
    int avx_cpu = ecx & (1u << 28);
    int fma_cpu = ecx & (1u << 12);
    int ymm_os = 0, zmm_os = 0;

    if (osxsave) {
        uint64_t xcr0 = xgetbv(0);
        ymm_os = (xcr0 & 0x06) == 0x06; /* XMM + YMM */
        zmm_os = (xcr0 & 0xe6) == 0xe6; /* + opmask + ZMM0-15 hi + ZMM16-31 */
    }

    if (avx_cpu && ymm_os) {
        flags |= STRIDE_CPU_AVX;
        if (fma_cpu)
            flags |= STRIDE_CPU_FMA3;

        if (max_leaf >= 7) {
            unsigned eax7 = 0, ebx7 = 0, ecx7 = 0, edx7 = 0;
            __get_cpuid_count(7, 0, &eax7, &ebx7, &ecx7, &edx7);

            if (ebx7 & (1u << 5))
                flags |= STRIDE_CPU_AVX2;

            /* AVX-512: F (16), DQ (17), BW (30), VL (31) together. */
            if (zmm_os && (ebx7 & (1u << 16)) && (ebx7 & (1u << 17)) &&
                (ebx7 & (1u << 30)) && (ebx7 & (1u << 31)))
                flags |= STRIDE_CPU_AVX512;
        }
    }

    return flags;
}

#else

int stride_cpu_flags(void) {
    return 0;
}

#endif
