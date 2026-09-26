// Audit driver: teammate reader + primal simplex + teammate checker; writes a ps26119 .sol
// (with our fingerprint of the model as the teammate reader sees it) for tools/verify.py.
#include <cstdio>
#include <cstring>
#include <string>
#include "gpuopt/mps_reader.hpp"
#include "gpuopt/simplex/primal_simplex.hpp"
#include "gpuopt/solution_checker.hpp"
#include "ps26119/model.h"
using namespace gpuopt;
int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: audit_driver model.mps out.sol [time_limit] [--fp-only]\n"); return 2; }
  double tl = argc > 3 ? std::atof(argv[3]) : 60.0;
  bool fp_only = argc > 4 && std::strcmp(argv[4], "--fp-only") == 0;
  MpsReadResult rr;
  try { rr = read_mps_file(argv[1]); } catch (const std::exception& e) { std::printf("READ_ERROR %s\n", e.what()); return 3; }
  const LpProblem& lp = rr.problem;
  ps26119::Model M;
  M.num_rows = lp.num_rows(); M.num_cols = lp.num_cols(); M.sense = lp.sense == ObjSense::kMinimize ? 1 : -1;
  M.obj_offset = lp.obj_offset; M.obj = lp.obj; M.col_lower = lp.col_lower; M.col_upper = lp.col_upper;
  M.row_lower = lp.row_lower; M.row_upper = lp.row_upper; M.col_start = lp.A.col_start; M.row_index = lp.A.row_index; M.value = lp.A.value;
  bool anyint = false; for (char c : lp.is_integer) anyint |= c != 0;
  if (anyint) M.is_integer.assign(lp.is_integer.begin(), lp.is_integer.end());
  std::printf("fingerprint %s\nrows %d\ncols %d\nnnz %zu\nwarnings %zu\n", M.fingerprint_hex().c_str(), M.num_rows, M.num_cols, M.nnz(), rr.warnings.size());
  for (auto& w : rr.warnings) std::printf("warning %s\n", w.c_str());
  if (fp_only) return 0;
  SimplexOptions so; so.time_limit_seconds = tl;
  SolveResult r = solve_primal_simplex(lp, so);
  CheckReport rep; bool have = r.status == SolveStatus::kOptimal;
  if (have) rep = check_solution(lp, r.x, r.row_dual);
  std::printf("status %s\niterations %lld\nseconds %.6f\nobjective %.17g\nchecker %s\nmessage %s\n", to_string(r.status), r.iterations, r.seconds, r.objective, have ? rep.summary().c_str() : "n/a", r.message.c_str());
  FILE* f = std::fopen(argv[2], "w");
  std::fprintf(f, "PS26119-SOLUTION 1\nmodel %s\nname %s\nstatus %s\nengine gpuopt-primal-simplex\nprecision fp64\nobjective %.17g\niterations %lld\nseconds %.6f\n",
    M.fingerprint_hex().c_str(), lp.name.c_str(), have ? "Optimal" : "NotSolved", r.objective, r.iterations, r.seconds);
  if (have) {
    std::fprintf(f, "COLUMNS %d\n", M.num_cols);
    for (int j = 0; j < M.num_cols; ++j) std::fprintf(f, "%d %.17g %.17g %s\n", j, r.x[j], r.col_dual[j], lp.col_names[j].c_str());
    std::fprintf(f, "ROWS %d\n", M.num_rows);
    for (int i = 0; i < M.num_rows; ++i) std::fprintf(f, "%d %.17g %.17g %s\n", i, r.row_activity[i], r.row_dual[i], lp.row_names[i].c_str());
  }
  std::fprintf(f, "END\n"); std::fclose(f);
  return 0;
}
