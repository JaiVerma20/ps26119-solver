// test_gpu.cpp — CUDA backend vs CPU backend (compiled only with PS26119_ENABLE_CUDA).
//   1. op-by-op: the same sequence of backend calls on both backends gives the same
//      vectors (fp64: ~1e-12 relative; fp32: ~1e-5) and the same KKT statistics;
//   2. end to end: both engines, both precisions, GPU answers equal CPU answers within
//      tolerance on the small Netlib set, and pass the verifier-grade KKT check.
#include <gtest/gtest.h>

#include <cmath>
#include <algorithm>

#include "io/lpm_reader.h"
#include "kkt_check.h"
#include "pdhg/backend.h"
#include "pdhg/scaling.h"
#include "ps26119/solve.h"

using namespace ps26119;

namespace {

std::string data(const std::string& rel) { return std::string(PS26119_SOURCE_DIR) + "/data/" + rel; }
const char* kNetlib[] = {"afiro", "sc50a", "sc50b", "kb2", "adlittle", "blend", "share2b", "sc105", "stocfor1", "recipe"};

double max_rel_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0;
  for (std::size_t i = 0; i < a.size(); ++i) d = std::max(d, std::fabs(a[i] - b[i]) / (1 + std::fabs(a[i])));
  return d;
}

}  // namespace

TEST(Gpu, BackendOpsMatchCpu) {
  for (const char* name : {"afiro", "share2b", "stocfor1"}) {
    Model m;
    ASSERT_TRUE(io::read_lpm(data(std::string("netlib_small/") + name + ".lpm"), m).ok);
    auto sp = pdhg::make_scaled_problem(m, {});
    for (Precision prec : {Precision::Fp64, Precision::Mixed}) {
      SCOPED_TRACE(std::string(name) + (prec == Precision::Fp64 ? " fp64" : " mixed"));
      std::string err;
      auto cpu = pdhg::make_cpu_backend();
      auto gpu = pdhg::make_backend(true, err);
      ASSERT_TRUE(gpu) << err;
      const double tol = prec == Precision::Fp64 ? 1e-12 : 1e-4;
      std::vector<std::vector<double>> got[2];
      pdhg::KktStats k[2];
      pdhg::Backend* bs[2] = {cpu.get(), gpu.get()};
      for (int t = 0; t < 2; ++t) {
        auto& b = *bs[t];
        b.setup(sp);
        b.set_precision(prec);
        int x = b.create(pdhg::Space::Primal), x0 = b.create(pdhg::Space::Primal), xh = b.create(pdhg::Space::Primal);
        int xb = b.create(pdhg::Space::Primal), g = b.create(pdhg::Space::Primal);
        int y = b.create(pdhg::Space::Dual), y0 = b.create(pdhg::Space::Dual), yh = b.create(pdhg::Space::Dual);
        int yb = b.create(pdhg::Space::Dual), ax = b.create(pdhg::Space::Dual);
        std::vector<double> xi(sp.n), yi(sp.m);
        for (int j = 0; j < sp.n; ++j) xi[j] = std::sin(1.0 + j);
        for (int i = 0; i < sp.m; ++i) yi[i] = std::cos(2.0 + i);
        b.upload(x, xi);
        b.upload(y, yi);
        b.copy(x0, x);
        b.copy(y0, y);
        for (int it = 0; it < 20; ++it) {
          const double w = (it + 1.0) / (it + 2.0);
          b.spmv_t(y, g);
          b.r2h_primal(x, x0, g, 0.3, w, 1.0, xh, xb);
          b.spmv(xb, ax);
          b.r2h_dual(y, y0, ax, 0.4, w, 1.0, yh, yb);
        }
        b.primal_step(x, g, 0.2, xb);
        b.dual_step(y, ax, 0.5, yb);
        b.axpby(0.3, xh, -1.7, xb);
        for (int v : {x, xh, xb, y, yh, yb}) {
          std::vector<double> out;
          b.download(v, out);
          got[t].push_back(out);
        }
        got[t].push_back({b.dot(x, xh), b.norm2(y), b.diff_norm2(x, x0)});
        k[t] = b.kkt(xh, yh);
      }
      for (std::size_t v = 0; v < got[0].size(); ++v) EXPECT_LT(max_rel_diff(got[0][v], got[1][v]), tol) << "vector " << v;
      // KKT statistics are evaluated in fp64, but in mixed precision they are functions of fp32
      // iterates, which CPU and GPU round differently (summation order, FMA). First NVIDIA run
      // (2026-09-27): every fp64 comparison within 1e-9, every mixed one within 1.6e-6
      // relative — fp32 noise. fp64 keeps 1e-9; mixed uses 1e-5 (the "fp32: ~1e-5" above).
      const double kt = prec == Precision::Fp64 ? 1e-9 : 1e-5;
      EXPECT_NEAR(k[0].primal_residual, k[1].primal_residual, kt * (1 + k[0].primal_residual));
      EXPECT_NEAR(k[0].dual_residual, k[1].dual_residual, kt * (1 + k[0].dual_residual));
      EXPECT_NEAR(k[0].primal_obj, k[1].primal_obj, kt * (1 + std::fabs(k[0].primal_obj)));
      EXPECT_NEAR(k[0].dual_obj, k[1].dual_obj, kt * (1 + std::fabs(k[0].dual_obj)));
      EXPECT_NEAR(k[0].primal_max_rel, k[1].primal_max_rel, 1e-12 + kt * k[0].primal_max_rel);
      EXPECT_NEAR(k[0].dual_max_rel, k[1].dual_max_rel, 1e-12 + kt * k[0].dual_max_rel);
    }
  }
}

TEST(Gpu, EnginesMatchCpuOnSmallNetlib) {
  for (Algorithm alg : {Algorithm::Pdlp, Algorithm::R2hpdhg}) {
    for (Precision prec : {Precision::Fp64, Precision::Mixed}) {
      for (const char* name : kNetlib) {
        SCOPED_TRACE(std::string(name) + " " + to_string(alg) + " " + to_string(prec));
        Model m;
        ASSERT_TRUE(io::read_lpm(data(std::string("netlib_small/") + name + ".lpm"), m).ok);
        Options o;
        o.algorithm = alg;
        o.precision = prec;
        o.time_limit = 120;
        auto cpu = solve(m, o);
        o.use_gpu = true;
        auto gpu = solve(m, o);
        ASSERT_EQ(cpu.status, Status::Optimal) << cpu.message;
        ASSERT_EQ(gpu.status, Status::Optimal) << gpu.message;
        EXPECT_NEAR(gpu.objective, cpu.objective, 1e-6 * (1 + std::fabs(cpu.objective)));
        const auto k = test::kkt(m, gpu);
        EXPECT_LE(k.primal, 1e-6);
        EXPECT_LE(k.dual, 1e-6);
        EXPECT_LE(k.gap, 1e-6);
        // fp64 on GPU should follow the CPU trajectory closely (reduction order differs,
        // so exact equality is not expected)
        if (prec == Precision::Fp64) EXPECT_NEAR(double(gpu.iterations), double(cpu.iterations), 0.25 * cpu.iterations + 256);
      }
    }
  }
}

// Long rows (> 1024 nonzeros) take the block-per-row SpMV path; the Netlib set has none,
// so build a model with a dense linking row (and a dense column, for Aᵀ).
TEST(Gpu, LongRowKernelMatchesCpu) {
  const int m = 1500, n = 3000;  // row 0 has 3000 entries, column 0 has 1500 (both > 1024)
  Model mod;
  mod.num_rows = m;
  mod.num_cols = n;
  mod.obj.resize(n);
  mod.col_lower.assign(n, 0.0);
  mod.col_upper.assign(n, 10.0);
  mod.row_lower.assign(m, -kInf);
  mod.row_upper.assign(m, 50.0);
  mod.col_start.push_back(0);
  for (int j = 0; j < n; ++j) {
    mod.obj[j] = -1.0 - (j % 13) * 0.1;
    mod.row_index.push_back(0);  // row 0 is dense: n = 3000 entries
    mod.value.push_back(1.0 + (j % 5) * 0.25);
    const int i = 1 + j % (m - 1);
    mod.row_index.push_back(i);
    mod.value.push_back(0.5 + (j % 7) * 0.1);
    if (j == 0)  // column 0 is dense too (touches every row) ⇒ long row in Aᵀ
      for (int r = 1; r < m; ++r)
        if (r != i) mod.row_index.push_back(r), mod.value.push_back(0.3);
    mod.col_start.push_back(static_cast<int>(mod.value.size()));
  }
  // column 0 entries must be unique and the col_start consistent
  ASSERT_EQ(mod.validate(), "");
  auto sp = pdhg::make_scaled_problem(mod, {});
  std::string err;
  auto cpu = pdhg::make_cpu_backend();
  auto gpu = pdhg::make_backend(true, err);
  ASSERT_TRUE(gpu) << err;
  for (Precision prec : {Precision::Fp64, Precision::Mixed}) {
    std::vector<double> out[2][2];
    for (int t = 0; t < 2; ++t) {
      pdhg::Backend& b = t ? *gpu : *cpu;
      b.setup(sp);
      b.set_precision(prec);
      int x = b.create(pdhg::Space::Primal), g = b.create(pdhg::Space::Primal);
      int y = b.create(pdhg::Space::Dual), ax = b.create(pdhg::Space::Dual);
      std::vector<double> xi(n), yi(m);
      for (int j = 0; j < n; ++j) xi[j] = std::sin(0.1 * j);
      for (int i = 0; i < m; ++i) yi[i] = std::cos(0.3 * i);
      b.upload(x, xi);
      b.upload(y, yi);
      b.spmv(x, ax);
      b.spmv_t(y, g);
      b.download(ax, out[t][0]);
      b.download(g, out[t][1]);
    }
    const double tol = prec == Precision::Fp64 ? 1e-12 : 1e-4;
    EXPECT_LT(max_rel_diff(out[0][0], out[1][0]), tol);
    EXPECT_LT(max_rel_diff(out[0][1], out[1][1]), tol);
  }
  Options o;
  o.algorithm = Algorithm::R2hpdhg;
  auto c = solve(mod, o);
  o.use_gpu = true;
  auto gsol = solve(mod, o);
  ASSERT_EQ(c.status, Status::Optimal) << c.message;
  ASSERT_EQ(gsol.status, Status::Optimal) << gsol.message;
  EXPECT_NEAR(gsol.objective, c.objective, 1e-6 * (1 + std::fabs(c.objective)));
}
