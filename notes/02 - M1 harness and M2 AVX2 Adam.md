# 02 - M1 harness and M2 AVX2 Adam

## M1, the harness

A checkasm-style harness, modelled on FFmpeg's. Nothing on the asm side counts
until it's been checked against the C reference, and this is what does the
checking.

Pieces:

- CPU detection (`src/cpu/cpu.c`). cpuid plus xgetbv, so a flag being set
  means the OS actually saves the register state, not just that the silicon
  has the unit. Reports sse2 avx fma3 avx2 here, no avx512, which matches the
  Meteor Lake reality.
- Dispatch table (`stride_kernel_table`). One function pointer per kernel.
  `stride_kernel_table_init(t, flags)` fills it with the best implementation
  the flags allow, flags 0 gives the C references. Asm slots in behind the
  same pointers, nothing above the table changes.
- The harness itself (`tests/checkasm/`). 64-byte aligned buffers with canary
  zones on both sides, so an out-of-bounds write gets caught even when it
  lands on memory the kernel was free to touch nearby. splitmix64 PRNG,
  reseedable, every failure prints its seed. Sizes clustered around the SIMD
  width boundaries (0,1..7,8,9..15,16,17..) plus some large odd ones, each run
  at offset 0 and offset 1 so a kernel that assumes alignment gets caught.
- Comparison at a ULP tolerance, 0 meaning bit-exact. The read-only grads
  buffer is compared before and after too, so a kernel that scribbles on its
  input fails.
- Timing: lfence + rdtsc, best of many runs, four calls per measurement.
  Pinned to the first P-core via sched_setaffinity (read from
  /sys/devices/cpu_core/cpus), because on a hybrid CPU letting the process
  migrate between P and E cores turns the numbers into noise. These are TSC
  reference ticks, not core cycles, which is fine for comparing
  implementations measured the same way.

### Self-test

A harness has to be shown to catch bugs or its passes are meaningless.
`--selftest` runs three broken Adam kernels:

- wrong arithmetic, beta1 nudged by 1.000001
- skipped tail, only n & ~7 elements processed (the classic SIMD tail bug)
- out-of-bounds write, writes params[n]

All three get caught. `make test` runs the self-test as part of the suite.

### Baseline, before any asm

The multi-pass Adam (five passes, one per expression, the shape a NumPy-style
implementation lands on) is checked bit-exact against the fused single-pass C
reference and benchmarked against it. It's slower, and the gap widens with n
exactly as the memory-traffic argument predicts: 0.79x of single-pass at n=256
(in L1), down to 0.64x at n=4M (deep in DRAM) for fp32. Same arithmetic, more
trips to memory. That's the first benchmark target confirmed before a line of
assembly exists.

## M2, fused Adam in AVX2

`src/kernels/x86/adam_avx2.asm`, fp32 (8 lanes) and fp64 (4 lanes).

### Bit-exact on purpose

`gcc -O2 -std=c11` emits no FMA for the C reference (checked with -S). So a
kernel using plain vmulps/vaddps/vdivps/vsqrtps in the same order as the
scalar code runs the identical sequence of correctly-rounded IEEE ops, and
since the lanes are independent the result matches scalar C to the bit. The
harness checks at max ULP 0 and it passes.

One ordering detail bit me. C evaluates `(1-beta2)*g*g` left to right as
`((1-beta2)*g)*g`, not `(1-beta2)*(g*g)`. Get the multiplies in the wrong
order and the low bit drifts. The asm matches the C associativity.

This is the conservative first kernel: pure SIMD width plus memory traffic, no
accuracy tradeoff. The faster variant (FMA to fuse the multiply-adds, vrsqrtps
plus a Newton step instead of vsqrtps + vdivps) trades a few ULP for speed and
belongs in M5, where the harness ULP path gets used.

### The bug the harness caught

First run failed, and only on tail elements (n=127, 129, 255, ...). The vector
body uses ymm0..5 as working registers, which wipes the low lanes xmm0..5 that
held the scalar constants lr/beta1/beta2/eps/bc1/bc2. The tail loop then read
garbage. The guard zones plus the per-element diff pinned it to the tail right
away. Fix: the tail reads its constants from the surviving ymm8..15 broadcasts
and uses xmm0..3 as scratch. Exactly the kind of bug the harness exists for,
and it caught it on the first run.

### Numbers (Core Ultra 7 155H, core 0, TSC ticks/element)

| n | C f32 | avx2 f32 | speedup | C f64 | avx2 f64 | speedup |
|---|---|---|---|---|---|---|
| 256 | 8.04 | 1.79 | 4.48x | 12.56 | 6.03 | 2.08x |
| 4096 | 8.00 | 1.75 | 4.57x | 12.55 | 6.01 | 2.09x |
| 65536 | 8.00 | 1.75 | 4.57x | 12.59 | 6.01 | 2.09x |
| 1048576 | 8.06 | 1.85 | 4.55x | 12.60 | 6.17 | 2.10x |
| 4194304 | 8.12 | 1.83 | 4.50x | 13.26 | 6.17 | 2.23x |

fp32 holds around 4.5x, fp64 around 2.1x (half the lanes, half the win). 4.5x
is short of the 8x the lane width alone suggests, because the kernel is partly
bound by vdivps and vsqrtps throughput rather than pure width. That's the
honest number, and the gap is the obvious next lever (the rsqrt path).

### Headline, stated plainly

Fused AVX2 Adam is 4.5x (fp32) over the scalar C single pass, and that single
pass is already up to 1.5x faster than the multi-pass shape a naive framework
lands on. The defensible claim is the kernel against the naive multi-pass:
roughly 6-7x in fp32 at DRAM sizes, all of it bit-identical to the reference.

## Next, M3

BLAS-1 (axpy, dot, scal) and gemv, then a logistic regression objective on top
so Stride solves a real problem and not just analytic test functions. dot is
where the ULP path earns its place: splitting the accumulator across lanes
reorders the sum, so it won't be bit-exact, and that's the point.
