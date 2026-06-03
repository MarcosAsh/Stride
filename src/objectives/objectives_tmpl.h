/*
 * Analytic objective template, included once per precision by objectives.c.
 *
 * Every gradient here is hand-differentiated and exact. Saddle takes its
 * surface gradients numerically, these don't.
 */

/* Chain Rosenbrock:
 *   f(x) = sum_{i=0}^{n-2} (1 - x_i)^2 + 100 * (x_{i+1} - x_i^2)^2
 *
 * Each term couples x_i and x_{i+1}, so component j of the gradient picks up
 * a contribution from term j (as the leading variable) and from term j-1 (as
 * the trailing one).
 */
static REAL FN(rosenbrock_eval)(const void *ctx, const REAL *x, size_t n) {
    (void)ctx;
    REAL sum = 0;
    for (size_t i = 0; i + 1 < n; i++) {
        REAL a = (REAL)1 - x[i];
        REAL b = x[i + 1] - x[i] * x[i];
        sum += a * a + (REAL)100 * b * b;
    }
    return sum;
}

static void FN(rosenbrock_grad)(const void *ctx, const REAL *x, REAL *g, size_t n) {
    (void)ctx;
    for (size_t i = 0; i < n; i++)
        g[i] = 0;
    for (size_t i = 0; i + 1 < n; i++) {
        REAL t = x[i + 1] - x[i] * x[i];
        g[i] += -(REAL)2 * ((REAL)1 - x[i]) - (REAL)400 * x[i] * t;
        g[i + 1] += (REAL)200 * t;
    }
}

/* Beale:
 *   f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
 */
static REAL FN(beale_eval)(const void *ctx, const REAL *x, size_t n) {
    (void)ctx;
    (void)n;
    REAL X = x[0], Y = x[1];
    REAL t1 = (REAL)1.5 - X + X * Y;
    REAL t2 = (REAL)2.25 - X + X * Y * Y;
    REAL t3 = (REAL)2.625 - X + X * Y * Y * Y;
    return t1 * t1 + t2 * t2 + t3 * t3;
}

static void FN(beale_grad)(const void *ctx, const REAL *x, REAL *g, size_t n) {
    (void)ctx;
    (void)n;
    REAL X = x[0], Y = x[1];
    REAL t1 = (REAL)1.5 - X + X * Y;
    REAL t2 = (REAL)2.25 - X + X * Y * Y;
    REAL t3 = (REAL)2.625 - X + X * Y * Y * Y;
    g[0] = (REAL)2 * t1 * (Y - (REAL)1) + (REAL)2 * t2 * (Y * Y - (REAL)1) +
           (REAL)2 * t3 * (Y * Y * Y - (REAL)1);
    g[1] = (REAL)2 * t1 * X + (REAL)2 * t2 * ((REAL)2 * X * Y) +
           (REAL)2 * t3 * ((REAL)3 * X * Y * Y);
}

/* Himmelblau:
 *   f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
 */
static REAL FN(himmelblau_eval)(const void *ctx, const REAL *x, size_t n) {
    (void)ctx;
    (void)n;
    REAL X = x[0], Y = x[1];
    REAL t1 = X * X + Y - (REAL)11;
    REAL t2 = X + Y * Y - (REAL)7;
    return t1 * t1 + t2 * t2;
}

static void FN(himmelblau_grad)(const void *ctx, const REAL *x, REAL *g, size_t n) {
    (void)ctx;
    (void)n;
    REAL X = x[0], Y = x[1];
    REAL t1 = X * X + Y - (REAL)11;
    REAL t2 = X + Y * Y - (REAL)7;
    g[0] = (REAL)4 * X * t1 + (REAL)2 * t2;
    g[1] = (REAL)2 * t1 + (REAL)4 * Y * t2;
}

/* Ill-conditioned quadratic:
 *   f(x) = 0.5 * sum_i c_i * x_i^2,  c_i = kappa^(i / (n-1))
 *
 * Coefficients are recomputed on the fly; this is a test function, not a hot
 * path.
 */
static REAL FN(quadratic_coeff)(REAL kappa, size_t i, size_t n) {
    if (n <= 1 || kappa <= (REAL)1)
        return 1;
    return POW(kappa, (REAL)i / (REAL)(n - 1));
}

static REAL FN(quadratic_eval)(const void *ctx, const REAL *x, size_t n) {
    REAL kappa = ((const FN(stride_quadratic_ctx) *)ctx)->kappa;
    REAL sum = 0;
    for (size_t i = 0; i < n; i++)
        sum += (REAL)0.5 * FN(quadratic_coeff)(kappa, i, n) * x[i] * x[i];
    return sum;
}

static void FN(quadratic_grad)(const void *ctx, const REAL *x, REAL *g, size_t n) {
    REAL kappa = ((const FN(stride_quadratic_ctx) *)ctx)->kappa;
    for (size_t i = 0; i < n; i++)
        g[i] = FN(quadratic_coeff)(kappa, i, n) * x[i];
}

/* Constructors */

FN(stride_objective) FN(stride_rosenbrock)(size_t n) {
    FN(stride_objective) obj = {
        .n = n,
        .eval = FN(rosenbrock_eval),
        .grad = FN(rosenbrock_grad),
        .ctx = NULL,
    };
    return obj;
}

FN(stride_objective) FN(stride_beale)(void) {
    FN(stride_objective) obj = {
        .n = 2,
        .eval = FN(beale_eval),
        .grad = FN(beale_grad),
        .ctx = NULL,
    };
    return obj;
}

FN(stride_objective) FN(stride_himmelblau)(void) {
    FN(stride_objective) obj = {
        .n = 2,
        .eval = FN(himmelblau_eval),
        .grad = FN(himmelblau_grad),
        .ctx = NULL,
    };
    return obj;
}

FN(stride_objective) FN(stride_quadratic)(size_t n, const FN(stride_quadratic_ctx) *ctx) {
    FN(stride_objective) obj = {
        .n = n,
        .eval = FN(quadratic_eval),
        .grad = FN(quadratic_grad),
        .ctx = ctx,
    };
    return obj;
}
