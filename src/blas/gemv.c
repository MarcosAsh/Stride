#include "stride/kernels.h"

#include <stddef.h>

/* fp32 */
#define REAL float
#define FN(name) name##_f32
#define FNT(name) name##_f32_fn
#include "gemv_tmpl.h"
#undef REAL
#undef FN
#undef FNT

/* fp64 */
#define REAL double
#define FN(name) name##_f64
#define FNT(name) name##_f64_fn
#include "gemv_tmpl.h"
#undef REAL
#undef FN
#undef FNT
