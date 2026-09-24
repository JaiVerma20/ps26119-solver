// scaling.h — diagonal preconditioning for the first-order engines.
//
// Citations:
//   D. Ruiz, "A scaling algorithm to equilibrate both rows and columns norms in matrices"
//     (RAL-TR-2001-034) — ∞-norm equilibration.
//   T. Pock, A. Chambolle, "Diagonal preconditioning for first order primal-dual algorithms"
//     (ICCV 2011) — α = 1: rows by 1/sqrt(‖row‖₁), columns by 1/sqrt(‖col‖₁).
//   D. Applegate et al., "Practical large-scale linear programming using primal-dual hybrid
//     gradient" (PDLP, NeurIPS 2021) — Ruiz(10) then Pock-Chambolle(α=1).
//   A. R. Curtis, J. K. Reid, "On the automatic scaling of matrices for Gaussian elimination"
//     (JIMA 1972) — geometric-mean scaling; used as a first stage by cuPDLPx
//     (src/preconditioner.cu, geometric_mean_rescaling).
//   H. Lu, Z. Peng, J. Yang, "cuPDLPx" (arXiv 2507.14051) — additional bound/objective
//     rescaling so that ‖b̃‖ ≈ ‖c̃‖ ≈ 1 (approach informed by MIT-Lu-Lab/cuPDLPx
//     src/preconditioner.cu; our own code).
//
// Scaled problem (engines only ever see this):
//     min c̃ᵀx̃  s.t.  r̃l ≤ Ã x̃ ≤ r̃u,  l̃ ≤ x̃ ≤ ũ        (always a MIN problem)
//   Ã = R A C,  c̃ = os · C (sense·c),  r̃ = bs · R r,  l̃ = bs · l / C
//   with R = diag(row_scale), C = diag(col_scale), bs = bound_scale, os = obj_scale.
// Unscaling:  x = C x̃ / bs,   y_min = R ỹ / os  (then y_out = sense · y_min).
// Invariants: all scale factors are finite and > 0; empty rows / columns get factor 1.
#pragma once

#include <vector>

#include "la/csr.h"
#include "ps26119/model.h"

namespace ps26119::pdhg {

struct ScalingOptions {
  // Geometric-mean equilibration first (cuPDLPx: 12 sweeps, always): row factor
  // r_i = 1/sqrt(min_j |a_ij c_j| · max_j |a_ij c_j|), then columns likewise, alternating.
  // ADAPTIVE (ours): applied only when the entries of A span at least
  // 10^geometric_mean_min_log10_range. Evidence (bench/results/netlib-full-*-{gm12,gm4}-*
  // vs the untagged run at f10527c, 93 Netlib LPs, 60 s each): always-on solves 85 vs 81
  // and cuts geomean iterations 24%, but costs 20–25% on the well-scaled refinery models
  // (range 10^2.5) and most small Netlib models; the threshold 10^4.5 keeps all 85 and
  // the geomean (17,375 vs 17,274) while leaving well-scaled models untouched.
  int geometric_mean_iterations = 12;
  double geometric_mean_min_log10_range = 4.5;  // 0 = always apply
  int ruiz_iterations = 10;
  bool pock_chambolle = true;
  bool bound_objective_rescaling = true;
};

struct ScaledProblem {
  int m = 0, n = 0;
  int sense = 1;
  la::Csr<double> A;   // Ã, m×n
  la::Csr<double> At;  // Ãᵀ, n×m
  std::vector<double> c, col_lower, col_upper, row_lower, row_upper;  // scaled data
  std::vector<double> row_scale, col_scale;                           // R, C
  double bound_scale = 1.0, obj_scale = 1.0;                          // bs, os
  double log10_range = 0.0;             // log10(max|a| / min|a|) of the original matrix
  bool geometric_mean_applied = false;  // adaptive decision, see ScalingOptions

  // Original (unscaled) data, kept for fp64 KKT evaluation in the original space.
  const Model* original = nullptr;
  la::Csr<double> A_orig;   // A, m×n
  la::Csr<double> At_orig;  // Aᵀ, n×m

  void unscale_primal(const std::vector<double>& xs, std::vector<double>& x) const;
  // Returns y in the MIN form (multiply by sense for the reported dual).
  void unscale_dual(const std::vector<double>& ys, std::vector<double>& y_min) const;
};

ScaledProblem make_scaled_problem(const Model& model, const ScalingOptions& opt);

}  // namespace ps26119::pdhg
