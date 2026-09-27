// c_api.cpp — the C API (include/ps26119/ps26119.h). Exceptions never cross this boundary.
#include "ps26119/ps26119.h"

#include <cstdio>
#include <cstring>
#include <exception>

#include "io/mps_reader.h"
#include "io/solution_writer.h"
#include "ps26119/solve.h"
#include "ps26119/version.h"

using namespace ps26119;

namespace {

void set_text(char* dst, std::size_t cap, const std::string& s) {
  std::snprintf(dst, cap, "%s", s.c_str());
}

void clear(ps26119_result* r) {
  std::memset(r, 0, sizeof *r);
  r->status = PS26119_NOT_SOLVED;
  r->check = -1;
}

// C options -> Options. Returns an error text, or nullptr on success.
const char* to_options(const ps26119_options* opt, int num_rows, int num_cols, Options& o) {
  if (!opt) return nullptr;
  switch (opt->algorithm) {
    case PS26119_ALG_ORACLE: o.algorithm = Algorithm::Oracle; break;
    case PS26119_ALG_PDLP: o.algorithm = Algorithm::Pdlp; break;
    case PS26119_ALG_R2HPDHG: o.algorithm = Algorithm::R2hpdhg; break;
    case PS26119_ALG_SIMPLEX: o.algorithm = Algorithm::Simplex; break;
    case PS26119_ALG_AUTO: o.algorithm = Algorithm::Auto; break;
    default: return "unknown algorithm";
  }
  o.precision = opt->precision == PS26119_PREC_MIXED ? Precision::Mixed : Precision::Fp64;
  o.use_gpu = opt->use_gpu != 0;
  if (opt->tolerance > 0) o.tolerance = opt->tolerance;
  if (opt->time_limit > 0) o.time_limit = opt->time_limit;
  if (opt->iteration_limit > 0) o.iteration_limit = opt->iteration_limit;
  o.verbosity = opt->verbosity;
  o.threads = opt->threads;
  if (opt->warm_x) o.warm_x.assign(opt->warm_x, opt->warm_x + num_cols);
  if (opt->warm_y) o.warm_y.assign(opt->warm_y, opt->warm_y + num_rows);
  return nullptr;
}

void fill(const Solution& s, ps26119_result* result) {
  result->status = static_cast<int>(s.status);  // enum orders match (checked by test)
  result->objective = s.objective;
  result->dual_objective = s.dual_objective;
  result->primal_residual = s.primal_residual;
  result->dual_residual = s.dual_residual;
  result->gap = s.gap;
  result->certified_bound = s.certified_bound;
  result->iterations = s.iterations;
  result->seconds = s.seconds;
  result->check = s.check == "PASS" ? 1 : s.check == "FAIL" ? 0 : -1;
  set_text(result->engine, sizeof result->engine, s.engine);
  set_text(result->message, sizeof result->message, s.message);
}

}  // namespace

extern "C" {

const char* ps26119_version(void) { return kVersion; }

void ps26119_default_options(ps26119_options* opt) {
  if (!opt) return;
  const Options d;
  opt->algorithm = PS26119_ALG_AUTO;
  opt->precision = PS26119_PREC_FP64;
  opt->use_gpu = 0;
  opt->tolerance = d.tolerance;
  opt->time_limit = d.time_limit;
  opt->iteration_limit = d.iteration_limit;
  opt->verbosity = 0;
  opt->threads = 1;
  opt->warm_x = nullptr;
  opt->warm_y = nullptr;
}

int ps26119_solve_lp(int num_rows, int num_cols, int sense, double obj_offset, const double* c,
                     const double* col_lower, const double* col_upper, const double* row_lower,
                     const double* row_upper, const int* col_start, const int* row_index, const double* value,
                     const ps26119_options* opt, ps26119_result* result, double* x, double* y, double* z) {
  return ps26119_solve_lp_ex(num_rows, num_cols, sense, obj_offset, c, col_lower, col_upper, row_lower, row_upper,
                             col_start, row_index, value, opt, result, x, y, z, nullptr, nullptr);
}

int ps26119_solve_lp_ex(int num_rows, int num_cols, int sense, double obj_offset, const double* c,
                        const double* col_lower, const double* col_upper, const double* row_lower,
                        const double* row_upper, const int* col_start, const int* row_index, const double* value,
                        const ps26119_options* opt, ps26119_result* result, double* x, double* y, double* z,
                        double* dual_ray, double* primal_ray) {
  if (!result) return PS26119_INVALID_ARGUMENT;
  clear(result);
  auto invalid = [&](const char* why) {
    result->status = PS26119_INVALID_ARGUMENT;
    set_text(result->message, sizeof result->message, why);
    return result->status;
  };
  if (num_rows < 0 || num_cols < 0) return invalid("negative dimension");
  if (!col_start) return invalid("col_start is NULL");
  if (num_cols > 0 && (!c || !col_lower || !col_upper)) return invalid("column data is NULL");
  if (num_rows > 0 && (!row_lower || !row_upper)) return invalid("row bounds are NULL");
  const int nnz = col_start[num_cols];
  if (nnz < 0) return invalid("col_start[n] < 0");
  if (nnz > 0 && (!row_index || !value)) return invalid("matrix arrays are NULL");

  try {
    Model m;
    m.num_rows = num_rows;
    m.num_cols = num_cols;
    m.sense = sense;
    m.obj_offset = obj_offset;
    m.obj.assign(c, c + num_cols);
    m.col_lower.assign(col_lower, col_lower + num_cols);
    m.col_upper.assign(col_upper, col_upper + num_cols);
    m.row_lower.assign(row_lower, row_lower + num_rows);
    m.row_upper.assign(row_upper, row_upper + num_rows);
    m.col_start.assign(col_start, col_start + num_cols + 1);
    m.row_index.assign(row_index, row_index + nnz);
    m.value.assign(value, value + nnz);
    if (auto err = m.validate(); !err.empty()) return invalid(("invalid model: " + err).c_str());

    Options o;
    if (const char* bad = to_options(opt, num_rows, num_cols, o)) return invalid(bad);
    const Solution s = solve(m, o);
    fill(s, result);
    if (x && s.x.size() == static_cast<std::size_t>(num_cols)) std::memcpy(x, s.x.data(), sizeof(double) * num_cols);
    if (y && s.y.size() == static_cast<std::size_t>(num_rows)) std::memcpy(y, s.y.data(), sizeof(double) * num_rows);
    if (z && s.z.size() == static_cast<std::size_t>(num_cols)) std::memcpy(z, s.z.data(), sizeof(double) * num_cols);
    if (dual_ray && s.dual_ray.size() == static_cast<std::size_t>(num_rows))
      std::memcpy(dual_ray, s.dual_ray.data(), sizeof(double) * num_rows);
    if (primal_ray && s.primal_ray.size() == static_cast<std::size_t>(num_cols))
      std::memcpy(primal_ray, s.primal_ray.data(), sizeof(double) * num_cols);
    return result->status;
  } catch (const std::exception& e) {
    result->status = PS26119_NUMERICAL_ERROR;
    set_text(result->message, sizeof result->message, e.what());
    return result->status;
  } catch (...) {
    result->status = PS26119_NUMERICAL_ERROR;
    set_text(result->message, sizeof result->message, "unknown error");
    return result->status;
  }
}

int ps26119_solve_mps(const char* path, const ps26119_options* opt, ps26119_result* result,
                      const char* solution_path) {
  if (!result) return PS26119_INVALID_ARGUMENT;
  clear(result);
  auto invalid = [&](const std::string& why) {
    result->status = PS26119_INVALID_ARGUMENT;
    set_text(result->message, sizeof result->message, why);
    return result->status;
  };
  if (!path) return invalid("path is NULL");
  try {
    Model m;
    std::string err;
    if (!io::read_mps(path, m, err)) return invalid("read error: " + err);
    if (opt && (opt->warm_x || opt->warm_y)) return invalid("warm start is not supported by ps26119_solve_mps");
    Options o;
    if (const char* bad = to_options(opt, m.num_rows, m.num_cols, o)) return invalid(bad);
    const Solution s = solve(m, o);
    fill(s, result);
    if (solution_path && !io::write_solution(solution_path, m, s, err)) {
      set_text(result->message, sizeof result->message, "solved, but " + err);
    }
    return result->status;
  } catch (const std::exception& e) {
    result->status = PS26119_NUMERICAL_ERROR;
    set_text(result->message, sizeof result->message, e.what());
    return result->status;
  } catch (...) {
    result->status = PS26119_NUMERICAL_ERROR;
    set_text(result->message, sizeof result->message, "unknown error");
    return result->status;
  }
}

}  // extern "C"
