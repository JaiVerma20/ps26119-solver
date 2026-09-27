// test_differential.cpp — Level 2/3 of the testing hierarchy (docs/ARCHITECTURE.md):
// every LP engine on the same random models, through the same solve() entry point.
//
//   exact engines:  dd oracle (ours) · tableau oracle (teammate's) · primal simplex
//                   (presolve off and on) — must agree on the status, and on the optimum
//                   to 1e-7 relative; every Optimal passes the in-process gate.
//   first-order:    r²HPDHG at 1e-8 — when the exact engines say Optimal it must reach
//                   Optimal with the same optimum (1e-6); when they say Infeasible or
//                   Unbounded it must never say Optimal.
// Two generators: the teammate's (integer coefficients in [−5, 5]) and a harder one with
// real coefficients and rows scaled over six orders of magnitude.
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <string>
#include <cstdio>

#include "oracle/dense_tableau.h"
#include "ps26119/solve.h"
#include "unit/random_lp.h"

using namespace ps26119;

namespace {

// Real-valued coefficients in ±[0.1, 10], rows scaled by 10^U(-3,3), every row and bound
// kind. 80% of the models have rows built around A·x0 for a point x0 inside the bounds
// (feasible by construction); the rest get random right-hand sides.
randlp::RandomLp random_real_lp(std::mt19937& rng, int max_rows, int max_cols, double density) {
  std::uniform_int_distribution<int> dim_m(1, max_rows), dim_n(1, max_cols), kind(0, 5);
  std::uniform_real_distribution<double> unit(0.0, 1.0), mag(0.1, 10.0), expo(-3.0, 3.0), c(-5.0, 5.0);
  randlp::RandomLp out;
  Model& lp = out.lp;
  const int m = dim_m(rng), n = dim_n(rng);
  lp.sense = unit(rng) < 0.5 ? 1 : -1;
  lp.obj_offset = c(rng);
  std::vector<double> row_scale(m);
  for (double& sc : row_scale) sc = std::pow(10.0, expo(rng));
  std::vector<la::Triplet> t;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      if (unit(rng) < density) t.push_back({i, j, (unit(rng) < 0.5 ? -1 : 1) * mag(rng) * row_scale[i]});
  randlp::set_matrix(lp, la::build_csc(m, n, t));
  std::vector<double> x0(n);
  out.all_boxed = true;
  for (int j = 0; j < n; ++j) {
    lp.obj.push_back(c(rng));
    const double base = c(rng), width = 0.5 + 10 * unit(rng);
    switch (kind(rng)) {
      case 0: lp.col_lower.push_back(0); lp.col_upper.push_back(kInf); x0[j] = 3 * unit(rng); break;
      case 1: lp.col_lower.push_back(base); lp.col_upper.push_back(base + width); x0[j] = base + width * unit(rng); break;
      case 2: lp.col_lower.push_back(-kInf); lp.col_upper.push_back(base); x0[j] = base - 3 * unit(rng); break;
      case 3: lp.col_lower.push_back(-kInf); lp.col_upper.push_back(kInf); x0[j] = base; break;
      case 4: lp.col_lower.push_back(base); lp.col_upper.push_back(base); x0[j] = base; break;
      default: lp.col_lower.push_back(base); lp.col_upper.push_back(kInf); x0[j] = base + 3 * unit(rng); break;
    }
    if (!std::isfinite(lp.col_lower[j]) || !std::isfinite(lp.col_upper[j])) out.all_boxed = false;
  }
  out.known_feasible = unit(rng) < 0.8;
  const std::vector<double> ax0 = lp.row_activity(x0);
  for (int i = 0; i < m; ++i) {
    const double centre = out.known_feasible ? ax0[i] : 2 * c(rng) * row_scale[i];
    const double slack = out.known_feasible ? 4 * unit(rng) * row_scale[i] : 0.0;
    switch (kind(rng)) {
      case 0: lp.row_lower.push_back(centre); lp.row_upper.push_back(centre); break;
      case 1: lp.row_lower.push_back(-kInf); lp.row_upper.push_back(centre + slack); break;
      case 2: lp.row_lower.push_back(centre - slack); lp.row_upper.push_back(kInf); break;
      case 3: lp.row_lower.push_back(centre - slack); lp.row_upper.push_back(centre + slack + row_scale[i]); break;
      default: lp.row_lower.push_back(-kInf); lp.row_upper.push_back(centre + slack); break;
    }
  }
  lp.is_integer.assign(n, 0);
  return out;
}

struct Tally {
  int optimal = 0, infeasible = 0, unbounded = 0, mismatches = 0, pdhg_limits = 0;
  // Infeasible / Unbounded verdicts of the simplex and r2hpdhg and how many carried a
  // certificate that passed the gate (core/certificates.h).
  int claims_simplex = 0, certified_simplex = 0, claims_pdhg = 0, certified_pdhg = 0;
};

void run(const char* label, int trials, unsigned seed, bool real, int max_rows, int max_cols, double density) {
  std::mt19937 rng(seed);
  Tally t;
  for (int trial = 0; trial < trials; ++trial) {
    const randlp::RandomLp r = real ? random_real_lp(rng, max_rows, max_cols, density)
                                     : randlp::random_lp(rng, max_rows, max_cols, density);
    const Model& lp = r.lp;
    auto with = [&](Algorithm a, bool presolve) {
      Options o;
      o.algorithm = a;
      o.presolve = presolve;
      o.iteration_limit = a == Algorithm::R2hpdhg ? 400000 : o.iteration_limit;
      return solve(lp, o);
    };
    const Solution dd = with(Algorithm::Oracle, false);
    const Solution tab = oracle::solve_dense_tableau(lp);
    const Solution sx = with(Algorithm::Simplex, false);
    const Solution sxp = with(Algorithm::Simplex, true);
    const Solution pd = with(Algorithm::R2hpdhg, true);

    bool ok = dd.status == tab.status && dd.status == sx.status && dd.status == sxp.status;
    const double ref = dd.objective, tol = 1e-7 * (1 + std::fabs(ref));
    if (ok && dd.status == Status::Optimal) {
      ok = std::fabs(tab.objective - ref) <= tol && std::fabs(sx.objective - ref) <= tol &&
           std::fabs(sxp.objective - ref) <= tol && dd.check == "PASS" && sx.check == "PASS" && sxp.check == "PASS";
      if (pd.status == Status::Optimal) {
        ok = ok && std::fabs(pd.objective - ref) <= 1e-6 * (1 + std::fabs(ref)) && pd.check == "PASS";
      } else if (pd.status == Status::IterationLimit || pd.status == Status::TimeLimit) {
        ++t.pdhg_limits;  // slow, not wrong; counted and bounded below
      } else {
        ok = false;
      }
    } else if (ok) {
      ok = pd.status != Status::Optimal;  // never Optimal on an infeasible/unbounded model
    }
    if (!ok) {
      ++t.mismatches;
      if (t.mismatches <= 5) {
        ADD_FAILURE() << label << " trial " << trial << " (" << lp.num_rows << "x" << lp.num_cols << "): dd "
                      << to_string(dd.status) << " " << dd.objective << " | tableau " << to_string(tab.status) << " "
                      << tab.objective << " | simplex " << to_string(sx.status) << " " << sx.objective << " ["
                      << sx.message << "] | simplex+presolve " << to_string(sxp.status) << " " << sxp.objective
                      << " | r2hpdhg " << to_string(pd.status) << " " << pd.objective;
      }
    }
    if (r.known_feasible && dd.status == Status::Infeasible) {
      ++t.mismatches;
      ADD_FAILURE() << label << " trial " << trial << ": feasible-by-construction model reported Infeasible";
    }
    if (r.all_boxed && dd.status == Status::Unbounded) {
      ++t.mismatches;
      ADD_FAILURE() << label << " trial " << trial << ": all-boxed model reported Unbounded";
    }
    for (const Solution* e : {&sx, &sxp}) {
      if (e->status == Status::Infeasible || e->status == Status::Unbounded) {
        ++t.claims_simplex;
        if (e->check == "PASS") ++t.certified_simplex;
      }
    }
    if (pd.status == Status::Infeasible || pd.status == Status::Unbounded) {
      ++t.claims_pdhg;
      if (pd.check == "PASS") ++t.certified_pdhg;
    }
    if (dd.status == Status::Optimal) ++t.optimal;
    if (dd.status == Status::Infeasible) ++t.infeasible;
    if (dd.status == Status::Unbounded) ++t.unbounded;
  }
  std::printf("  %-34s %d LPs: %d optimal, %d infeasible, %d unbounded, %d mismatches, %d PDHG limits\n", label,
              trials, t.optimal, t.infeasible, t.unbounded, t.mismatches, t.pdhg_limits);
  std::printf("  %-34s certified infeasible/unbounded verdicts: simplex %d/%d, r2hpdhg %d/%d\n", "", t.certified_simplex,
              t.claims_simplex, t.certified_pdhg, t.claims_pdhg);
  EXPECT_EQ(t.mismatches, 0);
  EXPECT_EQ(t.certified_simplex, t.claims_simplex);  // every Infeasible / Unbounded verdict is certified
  EXPECT_EQ(t.certified_pdhg, t.claims_pdhg);
  EXPECT_GT(t.optimal, trials / 5);                // a meaningful mix of outcomes
  EXPECT_LE(t.pdhg_limits, t.optimal / 50 + 1);    // first-order limits must stay rare on these sizes
}

}  // namespace

TEST(Differential, IntegerCoefficientLps) { run("integer coefficients (<=10x12)", 800, 101, false, 10, 12, 0.5); }

TEST(Differential, RealCoefficientsBadlyScaledRows) {
  run("real coeffs, rows 1e-3..1e3 (<=12x15)", 800, 202, true, 12, 15, 0.45);
}

TEST(Differential, SparserMediumLps) { run("sparse medium (<=30x40)", 200, 303, true, 30, 40, 0.15); }
