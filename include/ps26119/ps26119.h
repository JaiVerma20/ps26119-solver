/* ps26119.h — C API (stable, exception-free boundary for C, Python ctypes, Fortran, ...).
 *
 * The model is passed in the same form as the C++ Model contract (CLAUDE.md §6):
 *     minimise / maximise   cᵀx + obj_offset
 *     subject to            row_lower ≤ A x ≤ row_upper,   col_lower ≤ x ≤ col_upper
 * A in CSC: col_start[n+1], row_index[nnz], value[nnz]. Infinity = HUGE_VAL (IEEE inf).
 * Dual convention as in solution.h (z = c − Aᵀy, HiGHS signs).
 *
 * No function throws or aborts: every error is returned as a status code.
 */
#ifndef PS26119_C_API_H
#define PS26119_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes (same meaning and order as ps26119::Status). */
enum {
  PS26119_OPTIMAL = 0,
  PS26119_INFEASIBLE = 1,
  PS26119_UNBOUNDED = 2,
  PS26119_ITERATION_LIMIT = 3,
  PS26119_TIME_LIMIT = 4,
  PS26119_NUMERICAL_ERROR = 5,
  PS26119_NOT_SOLVED = 6,
  PS26119_INVALID_ARGUMENT = 7 /* C API only: null pointer, bad size, invalid model */
};

enum { PS26119_ALG_AUTO = 0, PS26119_ALG_ORACLE = 1, PS26119_ALG_PDLP = 2, PS26119_ALG_R2HPDHG = 3, PS26119_ALG_SIMPLEX = 4 };
enum { PS26119_PREC_FP64 = 0, PS26119_PREC_MIXED = 1 };

typedef struct {
  int algorithm;          /* PS26119_ALG_* */
  int precision;          /* PS26119_PREC_* */
  int use_gpu;            /* 0/1 (needs a CUDA build) */
  double tolerance;       /* relative KKT target for first-order engines */
  double time_limit;      /* seconds */
  long long iteration_limit;
  int verbosity;
  int threads;            /* CPU threads (0 = all cores); results do not depend on it */
  /* Warm start (first-order engines): previous x (num_cols) and y (num_rows), or NULL. */
  const double* warm_x;
  const double* warm_y;
} ps26119_options;

typedef struct {
  int status;             /* PS26119_* */
  double objective;       /* cᵀx + obj_offset */
  double dual_objective;
  double primal_residual; /* relative, see src/pdhg/termination.h */
  double dual_residual;
  double gap;
  double certified_bound; /* rounding-proof bound on the optimum from y: lower (MIN) / upper (MAX);
                             ±HUGE_VAL when none exists, NaN when not computed */
  long long iterations;
  double seconds;
  char engine[16];
  char message[160];
  int check;              /* in-process verification on the original model of an Optimal LP answer,
                             or of the certificate of an Infeasible / Unbounded verdict:
                             1 PASS, 0 FAIL, -1 not applicable / not certified (Solution::check) */
} ps26119_result;

/* Library version string, e.g. "0.1.0". */
const char* ps26119_version(void);

/* Fills *opt with the library defaults (r2hpdhg, fp64, CPU, tolerance 1e-8). */
void ps26119_default_options(ps26119_options* opt);

/* Solves the LP. x (n), y (m), z (n) may be NULL; when non-NULL they receive the primal
 * point, row duals and reduced costs. is_integer may be NULL (all continuous).
 * sense: +1 minimise, −1 maximise. opt may be NULL (defaults). Returns result->status. */
int ps26119_solve_lp(int num_rows, int num_cols, int sense, double obj_offset, const double* c,
                     const double* col_lower, const double* col_upper, const double* row_lower,
                     const double* row_upper, const int* col_start, const int* row_index,
                     const double* value, const ps26119_options* opt, ps26119_result* result,
                     double* x, double* y, double* z);

/* As ps26119_solve_lp, plus the certificates (both may be NULL): for an Infeasible verdict
 * dual_ray (num_rows) receives the Farkas row multipliers r, for an Unbounded verdict
 * primal_ray (num_cols) receives the ray d (x then holds the feasible point it starts from).
 * They are written only when the engine returned one; result->check says whether it passed. */
int ps26119_solve_lp_ex(int num_rows, int num_cols, int sense, double obj_offset, const double* c,
                        const double* col_lower, const double* col_upper, const double* row_lower,
                        const double* row_upper, const int* col_start, const int* row_index,
                        const double* value, const ps26119_options* opt, ps26119_result* result,
                        double* x, double* y, double* z, double* dual_ray, double* primal_ray);

/* Reads an MPS file (free or fixed format) and solves it. When solution_path is non-NULL the
 * solution file (the format tools/verify.py checks) is written there. Read errors return
 * PS26119_INVALID_ARGUMENT with the reason (and line number) in result->message. */
int ps26119_solve_mps(const char* path, const ps26119_options* opt, ps26119_result* result,
                      const char* solution_path);

#ifdef __cplusplus
}
#endif

#endif /* PS26119_C_API_H */
