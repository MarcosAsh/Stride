#include "stride/solver.h"
#include "stride/kernels.h"

#include <math.h>
#include <stdlib.h>

/* fp32 variants */
#define REAL float
#define POW powf
#define FABS fabsf
#define FN(name) name##_f32
#include "solvers_tmpl.h"
#undef REAL
#undef POW
#undef FABS
#undef FN

/* fp64 variants */
#define REAL double
#define POW pow
#define FABS fabs
#define FN(name) name##_f64
#include "solvers_tmpl.h"
#undef REAL
#undef POW
#undef FABS
#undef FN
