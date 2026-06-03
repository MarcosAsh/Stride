#include "stride/solver.h"
#include "stride/kernels.h"

#include <stdlib.h>
#include <string.h>

/* curvature acceptance threshold, relative to |y|^2; looser in single
 * precision where the gradient differences are noisier */

/* fp32 */
#define REAL float
#define CURV_EPS 1e-6
#define FN(name) name##_f32
#define FNT(name) name##_f32_fn
#include "lbfgs_tmpl.h"
#undef REAL
#undef CURV_EPS
#undef FN
#undef FNT

/* fp64 */
#define REAL double
#define CURV_EPS 1e-10
#define FN(name) name##_f64
#define FNT(name) name##_f64_fn
#include "lbfgs_tmpl.h"
#undef REAL
#undef CURV_EPS
#undef FN
#undef FNT
