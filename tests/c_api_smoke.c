/* c_api_smoke.c — compiled as C (not C++) to prove ps26119.h is a real C header. */
#include <math.h>
#include <stdio.h>

#include "ps26119/ps26119.h"

int main(void) {
  /* min x + y  s.t.  x + 2y >= 2,  x, y >= 0   → optimum 1 at (0, 1) */
  const double c[] = {1, 1}, cl[] = {0, 0}, cu[] = {HUGE_VAL, HUGE_VAL};
  const double rl[] = {2}, ru[] = {HUGE_VAL};
  const int start[] = {0, 1, 2}, idx[] = {0, 0};
  const double val[] = {1, 2};
  ps26119_options o;
  ps26119_result r;
  double x[2];
  ps26119_default_options(&o);
  o.algorithm = PS26119_ALG_ORACLE;
  if (ps26119_solve_lp(1, 2, 1, 0.0, c, cl, cu, rl, ru, start, idx, val, &o, &r, x, NULL, NULL) != PS26119_OPTIMAL) {
    printf("FAIL: status %d (%s)\n", r.status, r.message);
    return 1;
  }
  if (fabs(r.objective - 1.0) > 1e-9 || fabs(x[1] - 1.0) > 1e-9) {
    printf("FAIL: objective %.17g\n", r.objective);
    return 1;
  }
  printf("ps26119 %s C API OK: objective %g\n", ps26119_version(), r.objective);
  return 0;
}
