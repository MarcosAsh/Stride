#include "stride/objective.h"

#include <math.h>
#include <stddef.h>

/* fp32 variants */
#define REAL float
#define POW powf
#define FN(name) name##_f32
#include "objectives_tmpl.h"
#undef REAL
#undef POW
#undef FN

/* fp64 variants */
#define REAL double
#define POW pow
#define FN(name) name##_f64
#include "objectives_tmpl.h"
#undef REAL
#undef POW
#undef FN
