// scaling.cpp — see scaling.h for the math and citations.
#include "pdhg/scaling.h"

#include <algorithm>
#include <cmath>

namespace ps26119::pdhg {
namespace {

// Scale A (CSR, m×n) in place by rows r and columns c: a_ij ← r_i a_ij c_j.
void apply(la::Csr<double>& A, const std::vector<double>& r, const std::vector<double>& c) {
  for (int i = 0; i < A.rows; ++i)
    for (std::int64_t k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) A.val[k] *= r[i] * c[A.col[k]];
}

// p = ∞ uses max|a|, p = 1 uses Σ|a|. Returns per-row and per-column norms.
void norms(const la::Csr<double>& A, bool inf_norm, std::vector<double>& rn, std::vector<double>& cn) {
  rn.assign(A.rows, 0.0);
  cn.assign(A.cols, 0.0);
  for (int i = 0; i < A.rows; ++i)
    for (std::int64_t k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
      const double a = std::fabs(A.val[k]);
      if (inf_norm) {
        rn[i] = std::max(rn[i], a);
        cn[A.col[k]] = std::max(cn[A.col[k]], a);
      } else {
        rn[i] += a;
        cn[A.col[k]] += a;
      }
    }
}

double safe_inv_sqrt(double v) { return v > 0.0 && std::isfinite(v) ? 1.0 / std::sqrt(v) : 1.0; }

}  // namespace

ScaledProblem make_scaled_problem(const Model& model, const ScalingOptions& opt) {
  ScaledProblem sp;
  sp.m = model.num_rows;
  sp.n = model.num_cols;
  sp.sense = model.sense;
  sp.original = &model;
  sp.A_orig = la::csr_from_model(model);
  sp.At_orig = la::csr_transpose_from_model(model);
  sp.A = sp.A_orig;
  sp.row_scale.assign(sp.m, 1.0);
  sp.col_scale.assign(sp.n, 1.0);

  std::vector<double> rn, cn, rf(sp.m), cf(sp.n);
  double log10_range = 0.0;
  {
    double amin = kInf, amax = 0.0;
    for (double v : sp.A_orig.val)
      if (v != 0.0) amin = std::min(amin, std::fabs(v)), amax = std::max(amax, std::fabs(v));
    if (amax > 0) log10_range = std::log10(amax / amin);
  }
  sp.log10_range = log10_range;
  sp.geometric_mean_applied = opt.geometric_mean_iterations > 0 && log10_range >= opt.geometric_mean_min_log10_range;
  if (sp.geometric_mean_applied) {
    // factors are recomputed from the ORIGINAL entries each sweep (Gauss–Seidel between
    // rows and columns), then applied once
    const la::Csr<double>& A0 = sp.A_orig;
    std::vector<double> r(sp.m, 1.0), c(sp.n, 1.0), lo, hi;
    auto clampf = [](double v) { return std::min(std::max(v, 1e-20), 1e20); };
    for (int it = 0; it < opt.geometric_mean_iterations; ++it) {
      lo.assign(sp.m, kInf);
      hi.assign(sp.m, 0.0);
      for (int i = 0; i < sp.m; ++i)
        for (std::int64_t k = A0.row_ptr[i]; k < A0.row_ptr[i + 1]; ++k) {
          const double v = std::fabs(A0.val[k]) * c[A0.col[k]];
          if (v > 0 && std::isfinite(v)) lo[i] = std::min(lo[i], v), hi[i] = std::max(hi[i], v);
        }
      for (int i = 0; i < sp.m; ++i)
        if (hi[i] > 0) r[i] = clampf(1.0 / (std::sqrt(lo[i]) * std::sqrt(hi[i])));
      lo.assign(sp.n, kInf);
      hi.assign(sp.n, 0.0);
      for (int i = 0; i < sp.m; ++i)
        for (std::int64_t k = A0.row_ptr[i]; k < A0.row_ptr[i + 1]; ++k) {
          const int j = A0.col[k];
          const double v = std::fabs(A0.val[k]) * r[i];
          if (v > 0 && std::isfinite(v)) lo[j] = std::min(lo[j], v), hi[j] = std::max(hi[j], v);
        }
      for (int j = 0; j < sp.n; ++j)
        if (hi[j] > 0) c[j] = clampf(1.0 / (std::sqrt(lo[j]) * std::sqrt(hi[j])));
    }
    apply(sp.A, r, c);
    sp.row_scale = r;
    sp.col_scale = c;
  }
  for (int it = 0; it < opt.ruiz_iterations; ++it) {
    norms(sp.A, true, rn, cn);
    for (int i = 0; i < sp.m; ++i) rf[i] = safe_inv_sqrt(rn[i]);
    for (int j = 0; j < sp.n; ++j) cf[j] = safe_inv_sqrt(cn[j]);
    apply(sp.A, rf, cf);
    for (int i = 0; i < sp.m; ++i) sp.row_scale[i] *= rf[i];
    for (int j = 0; j < sp.n; ++j) sp.col_scale[j] *= cf[j];
  }
  if (opt.pock_chambolle) {  // α = 1
    norms(sp.A, false, rn, cn);
    for (int i = 0; i < sp.m; ++i) rf[i] = safe_inv_sqrt(rn[i]);
    for (int j = 0; j < sp.n; ++j) cf[j] = safe_inv_sqrt(cn[j]);
    apply(sp.A, rf, cf);
    for (int i = 0; i < sp.m; ++i) sp.row_scale[i] *= rf[i];
    for (int j = 0; j < sp.n; ++j) sp.col_scale[j] *= cf[j];
  }

  // Scaled vectors before the bound/objective scalars.
  sp.c.resize(sp.n);
  sp.col_lower.resize(sp.n);
  sp.col_upper.resize(sp.n);
  for (int j = 0; j < sp.n; ++j) {
    sp.c[j] = sp.col_scale[j] * model.sense * model.obj[j];
    sp.col_lower[j] = model.col_lower[j] / sp.col_scale[j];
    sp.col_upper[j] = model.col_upper[j] / sp.col_scale[j];
  }
  sp.row_lower.resize(sp.m);
  sp.row_upper.resize(sp.m);
  for (int i = 0; i < sp.m; ++i) {
    sp.row_lower[i] = sp.row_scale[i] * model.row_lower[i];
    sp.row_upper[i] = sp.row_scale[i] * model.row_upper[i];
  }

  if (opt.bound_objective_rescaling) {
    double b2 = 0.0, c2 = 0.0;
    for (int i = 0; i < sp.m; ++i) {
      const double lo = std::isfinite(sp.row_lower[i]) ? std::fabs(sp.row_lower[i]) : 0.0;
      const double up = std::isfinite(sp.row_upper[i]) ? std::fabs(sp.row_upper[i]) : 0.0;
      const double b = std::max(lo, up);
      b2 += b * b;
    }
    for (double v : sp.c) c2 += v * v;
    sp.bound_scale = 1.0 / (std::sqrt(b2) + 1.0);
    sp.obj_scale = 1.0 / (std::sqrt(c2) + 1.0);
    for (double& v : sp.row_lower) v *= sp.bound_scale;
    for (double& v : sp.row_upper) v *= sp.bound_scale;
    for (double& v : sp.col_lower) v *= sp.bound_scale;
    for (double& v : sp.col_upper) v *= sp.bound_scale;
    for (double& v : sp.c) v *= sp.obj_scale;
  }
  sp.At = la::transpose(sp.A);
  return sp;
}

void ScaledProblem::unscale_primal(const std::vector<double>& xs, std::vector<double>& x) const {
  // Clip into the original bounds: the scaled x̂ is a projection, but unscaling can leave
  // a last-ulp excursion.
  x.resize(n);
  for (int j = 0; j < n; ++j)
    x[j] = std::min(std::max(col_scale[j] * xs[j] / bound_scale, original->col_lower[j]), original->col_upper[j]);
}

void ScaledProblem::unscale_dual(const std::vector<double>& ys, std::vector<double>& y) const {
  y.resize(m);
  for (int i = 0; i < m; ++i) y[i] = row_scale[i] * ys[i] / obj_scale;
}

}  // namespace ps26119::pdhg
