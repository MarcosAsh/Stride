#include "stride/objective.h"
#include "stride/kernels.h"

#include <math.h>
#include <stddef.h>

/* fp32 */
#define REAL float
#define EXP expf
#define LOG1P log1pf
#define FN(name) name##_f32
#include "logreg_tmpl.h"
#undef REAL
#undef EXP
#undef LOG1P
#undef FN

/* fp64 */
#define REAL double
#define EXP exp
#define LOG1P log1p
#define FN(name) name##_f64
#include "logreg_tmpl.h"
#undef REAL
#undef EXP
#undef LOG1P
#undef FN
