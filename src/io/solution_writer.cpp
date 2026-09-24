// solution_writer.cpp — see solution_writer.h for the format.
#include "io/solution_writer.h"

#include <cstdio>

namespace ps26119::io {

bool write_solution(const std::string& path, const Model& m, const Solution& s, std::string& error) {
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) {
    error = "cannot open " + path + " for writing";
    return false;
  }
  std::fprintf(f, "PS26119-SOLUTION 1\n");
  std::fprintf(f, "model %s\n", s.model_fingerprint.empty() ? m.fingerprint_hex().c_str()
                                                           : s.model_fingerprint.c_str());
  std::fprintf(f, "name %s\n", m.name.c_str());
  std::fprintf(f, "status %s\n", to_string(s.status));
  std::fprintf(f, "engine %s\n", s.engine.c_str());
  std::fprintf(f, "precision %s\n", s.precision.c_str());
  std::fprintf(f, "objective %.17g\n", s.objective);
  std::fprintf(f, "dual_objective %.17g\n", s.dual_objective);
  std::fprintf(f, "primal_residual %.6g\n", s.primal_residual);
  std::fprintf(f, "dual_residual %.6g\n", s.dual_residual);
  std::fprintf(f, "gap %.6g\n", s.gap);
  std::fprintf(f, "iterations %lld\n", static_cast<long long>(s.iterations));
  std::fprintf(f, "seconds %.6f\n", s.seconds);
  if (s.iterations_to_fast >= 0) {
    std::fprintf(f, "iterations_to_fast %lld\n", static_cast<long long>(s.iterations_to_fast));
    std::fprintf(f, "seconds_to_fast %.6f\n", s.seconds_to_fast);
  }
  if (!s.message.empty()) std::fprintf(f, "message %s\n", s.message.c_str());

  const bool have_primal = static_cast<int>(s.x.size()) == m.num_cols;
  const bool have_dual = static_cast<int>(s.z.size()) == m.num_cols && static_cast<int>(s.y.size()) == m.num_rows;
  const bool have_act = static_cast<int>(s.row_activity.size()) == m.num_rows;
  if (have_primal) {
    std::fprintf(f, "COLUMNS %d\n", m.num_cols);
    for (int j = 0; j < m.num_cols; ++j) {
      const char* name = j < static_cast<int>(m.col_names.size()) ? m.col_names[j].c_str() : "";
      std::fprintf(f, "%d %.17g %.17g %s\n", j, s.x[j], have_dual ? s.z[j] : 0.0, name);
    }
    std::fprintf(f, "ROWS %d\n", m.num_rows);
    const auto act = have_act ? s.row_activity : m.row_activity(s.x);
    for (int i = 0; i < m.num_rows; ++i) {
      const char* name = i < static_cast<int>(m.row_names.size()) ? m.row_names[i].c_str() : "";
      std::fprintf(f, "%d %.17g %.17g %s\n", i, act[i], have_dual ? s.y[i] : 0.0, name);
    }
  }
  std::fprintf(f, "END\n");
  const bool ok = std::fclose(f) == 0;
  if (!ok) error = "write failed for " + path;
  return ok;
}

}  // namespace ps26119::io
