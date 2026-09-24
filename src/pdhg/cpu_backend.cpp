// cpu_backend.cpp — CPU implementation of pdhg::Backend.
//
// Working precision fp64 or fp32 (mixed precision): iterate vectors and the scaled matrix
// are held in the working type T; all reductions accumulate in double; kkt() maps the
// iterate back to the original space and evaluates it in double on the original matrix.
// This is the reference implementation of every GPU kernel's math (CLAUDE.md §3).
// Loops are sequential by default (deterministic); OpenMP only when enabled.
#include <cmath>
#include <type_traits>

#include "pdhg/backend.h"

namespace ps26119::pdhg {
namespace {

template <class T>
T clamp(T v, T lo, T hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

template <class T>
struct Store {
  la::Csr<T> A, At;
  std::vector<T> c, l, u, rl, ru;
  std::vector<std::vector<T>> vecs;
};

class CpuBackend final : public Backend {
 public:
  std::string name() const override { return "cpu"; }

  void setup(const ScaledProblem& sp) override {
    sp_ = &sp;
    load(d_, sp);
    load(f_, sp);
  }

  void set_precision(Precision p) override {
    if (p == prec_) return;
    // convert every vector to the new working type
    if (p == Precision::Mixed) {
      for (std::size_t k = 0; k < d_.vecs.size(); ++k) f_.vecs[k].assign(d_.vecs[k].begin(), d_.vecs[k].end());
    } else {
      for (std::size_t k = 0; k < f_.vecs.size(); ++k) d_.vecs[k].assign(f_.vecs[k].begin(), f_.vecs[k].end());
    }
    prec_ = p;
  }
  Precision precision() const override { return prec_; }

  int create(Space s) override {
    const std::size_t len = s == Space::Primal ? sp_->n : sp_->m;
    d_.vecs.emplace_back(prec_ == Precision::Fp64 ? len : 0, 0.0);
    f_.vecs.emplace_back(prec_ == Precision::Mixed ? len : 0, 0.0f);
    sizes_.push_back(len);
    return static_cast<int>(sizes_.size()) - 1;
  }

  void upload(int v, const std::vector<double>& src) override {
    with([&](auto& S) { S.vecs[v].assign(src.begin(), src.end()); });
  }
  void download(int v, std::vector<double>& dst) override {
    with([&](auto& S) { dst.assign(S.vecs[v].begin(), S.vecs[v].end()); });
  }
  void fill(int v, double a) override {
    with([&](auto& S) {
      using T = typename std::decay_t<decltype(S.vecs[v])>::value_type;
      S.vecs[v].assign(sizes_[v], static_cast<T>(a));
    });
  }
  void copy(int dst, int src) override { with([&](auto& S) { S.vecs[dst] = S.vecs[src]; }); }
  void axpby(double a, int x, double b, int y) override {
    with([&](auto& S) {
      using T = typename std::decay_t<decltype(S.vecs[y])>::value_type;
      auto& Y = S.vecs[y];
      const auto& X = S.vecs[x];
      const T ta = static_cast<T>(a), tb = static_cast<T>(b);
      for (std::size_t i = 0; i < Y.size(); ++i) Y[i] = ta * X[i] + tb * Y[i];
    });
  }
  void spmv(int x, int out) override {
    with([&](auto& S) { S.A.template multiply<double>(S.vecs[x].data(), S.vecs[out].data()); });
  }
  void spmv_t(int y, int out) override {
    with([&](auto& S) { S.At.template multiply<double>(S.vecs[y].data(), S.vecs[out].data()); });
  }
  double dot(int a, int b) override {
    double s = 0;
    with([&](auto& S) {
      const auto &A = S.vecs[a], &B = S.vecs[b];
      for (std::size_t i = 0; i < A.size(); ++i) s += static_cast<double>(A[i]) * static_cast<double>(B[i]);
    });
    return s;
  }
  double norm2(int a) override { return std::sqrt(dot(a, a)); }
  double diff_norm2(int a, int b) override {
    double s = 0;
    with([&](auto& S) {
      const auto &A = S.vecs[a], &B = S.vecs[b];
      for (std::size_t i = 0; i < A.size(); ++i) {
        const double d = static_cast<double>(A[i]) - static_cast<double>(B[i]);
        s += d * d;
      }
    });
    return std::sqrt(s);
  }

  void primal_step(int x, int aty, double tau, int xhat) override {
    with([&](auto& S) {
      using T = typename std::decay_t<decltype(S.c)>::value_type;
      const T t = static_cast<T>(tau);
      const auto &X = S.vecs[x], &G = S.vecs[aty];
      auto& H = S.vecs[xhat];
      for (std::size_t j = 0; j < X.size(); ++j) H[j] = clamp<T>(X[j] - t * (S.c[j] - G[j]), S.l[j], S.u[j]);
    });
  }

  void dual_step(int y, int ax, double sigma, int yhat) override {
    with([&](auto& S) {
      using T = typename std::decay_t<decltype(S.c)>::value_type;
      const T s = static_cast<T>(sigma);
      const auto &Y = S.vecs[y], &AX = S.vecs[ax];
      auto& H = S.vecs[yhat];
      for (std::size_t i = 0; i < Y.size(); ++i) {
        const T v = AX[i] - Y[i] / s;
        H[i] = Y[i] - s * AX[i] + s * clamp<T>(v, S.rl[i], S.ru[i]);
      }
    });
  }

  void r2h_primal(int x, int x0, int aty, double tau, double w, double rho, int xhat, int xbar) override {
    with([&](auto& S) {
      using T = typename std::decay_t<decltype(S.c)>::value_type;
      const T t = static_cast<T>(tau), tw = static_cast<T>(w), a = static_cast<T>(2 * rho),
              b = static_cast<T>(1 - 2 * rho);
      auto& X = S.vecs[x];
      const auto &X0 = S.vecs[x0], &G = S.vecs[aty];
      auto &H = S.vecs[xhat], &B = S.vecs[xbar];
      for (std::size_t j = 0; j < X.size(); ++j) {
        const T h = clamp<T>(X[j] - t * (S.c[j] - G[j]), S.l[j], S.u[j]);
        H[j] = h;
        B[j] = 2 * h - X[j];
        X[j] = tw * (a * h + b * X[j]) + (1 - tw) * X0[j];
      }
    });
  }

  void r2h_dual(int y, int y0, int ax, double sigma, double w, double rho, int yhat, int ybar) override {
    with([&](auto& S) {
      using T = typename std::decay_t<decltype(S.c)>::value_type;
      const T s = static_cast<T>(sigma), tw = static_cast<T>(w), a = static_cast<T>(2 * rho),
              b = static_cast<T>(1 - 2 * rho);
      auto& Y = S.vecs[y];
      const auto &Y0 = S.vecs[y0], &AX = S.vecs[ax];
      auto& H = S.vecs[yhat];
      T* B = ybar >= 0 ? S.vecs[ybar].data() : nullptr;
      for (std::size_t i = 0; i < Y.size(); ++i) {
        const T v = AX[i] - Y[i] / s;
        const T h = Y[i] - s * AX[i] + s * clamp<T>(v, S.rl[i], S.ru[i]);
        H[i] = h;
        if (B) B[i] = 2 * h - Y[i];
        Y[i] = tw * (a * h + b * Y[i]) + (1 - tw) * Y0[i];
      }
    });
  }

  KktStats kkt(int xs, int ys) override {
    std::vector<double> xsd, ysd, x, y;
    download(xs, xsd);
    download(ys, ysd);
    sp_->unscale_primal(xsd, x);
    sp_->unscale_dual(ysd, y);
    return kkt_original(*sp_, x, y);
  }

  // Shared with tests and the engines' final report.
  static KktStats kkt_original(const ScaledProblem& sp, const std::vector<double>& x, const std::vector<double>& y);

 private:
  template <class T>
  static void load(Store<T>& S, const ScaledProblem& sp) {
    S.A = sp.A.cast<T>();
    S.At = sp.At.cast<T>();
    S.c.assign(sp.c.begin(), sp.c.end());
    S.l.assign(sp.col_lower.begin(), sp.col_lower.end());
    S.u.assign(sp.col_upper.begin(), sp.col_upper.end());
    S.rl.assign(sp.row_lower.begin(), sp.row_lower.end());
    S.ru.assign(sp.row_upper.begin(), sp.row_upper.end());
  }

  template <class F>
  void with(F&& f) {
    if (prec_ == Precision::Fp64) f(d_);
    else f(f_);
  }

  const ScaledProblem* sp_ = nullptr;
  Precision prec_ = Precision::Fp64;
  Store<double> d_;
  Store<float> f_;
  std::vector<std::size_t> sizes_;
};

KktStats CpuBackend::kkt_original(const ScaledProblem& sp, const std::vector<double>& x, const std::vector<double>& y) {
  const Model& M = *sp.original;
  const double sense = M.sense;
  KktStats k;
  std::vector<double> ax(sp.m), aty(sp.n);
  sp.A_orig.multiply<double>(x.data(), ax.data());
  sp.At_orig.multiply<double>(y.data(), aty.data());
  double rp2 = 0, rd2 = 0, b2 = 0, c2 = 0, p = 0, d = 0, cinf = 0, pmax = 0, dmax = 0;
  auto elem_viol = [](double v, double lo, double up) {
    if (v < lo) return (lo - v) / (1.0 + std::fabs(lo));
    if (v > up) return (v - up) / (1.0 + std::fabs(up));
    return 0.0;
  };
  for (int i = 0; i < sp.m; ++i) {
    const double lo = M.row_lower[i], up = M.row_upper[i];
    const double viol = ax[i] < lo ? lo - ax[i] : (ax[i] > up ? ax[i] - up : 0.0);
    rp2 += viol * viol;
    pmax = std::max(pmax, elem_viol(ax[i], lo, up));
    const double bl = std::isfinite(lo) ? std::fabs(lo) : 0.0, bu = std::isfinite(up) ? std::fabs(up) : 0.0;
    b2 += std::max(bl, bu) * std::max(bl, bu);
    // y is sign-feasible for PDHG outputs; any sign violation (infinite bound with the
    // "wrong" multiplier) is charged to the dual residual rather than the objective.
    if (y[i] > 0) {
      if (std::isfinite(lo)) d += y[i] * lo;
      else rd2 += y[i] * y[i], dmax = std::max(dmax, y[i]);
    } else if (y[i] < 0) {
      if (std::isfinite(up)) d += y[i] * up;
      else rd2 += y[i] * y[i], dmax = std::max(dmax, -y[i]);
    }
  }
  for (int j = 0; j < sp.n; ++j) {
    const double cj = sense * M.obj[j];
    c2 += cj * cj;
    cinf = std::max(cinf, std::fabs(cj));
    p += cj * x[j];
    pmax = std::max(pmax, elem_viol(x[j], M.col_lower[j], M.col_upper[j]));
    const double lam = cj - aty[j];
    const double lo = M.col_lower[j], up = M.col_upper[j];
    double z;
    if (std::isfinite(lo) && std::isfinite(up)) z = lam;
    else if (std::isfinite(lo)) z = std::max(lam, 0.0);
    else if (std::isfinite(up)) z = std::min(lam, 0.0);
    else z = 0.0;
    rd2 += (lam - z) * (lam - z);
    dmax = std::max(dmax, std::fabs(lam - z));
    if (z > 0) d += z * lo;
    else if (z < 0) d += z * up;
  }
  k.primal_residual = std::sqrt(rp2);
  k.dual_residual = std::sqrt(rd2);
  k.primal_obj = p;
  k.dual_obj = d;
  k.b_norm = std::sqrt(b2);
  k.c_norm = std::sqrt(c2);
  k.primal_max_rel = pmax;
  k.dual_max_rel = dmax / (1.0 + cinf);
  return k;
}

}  // namespace

std::unique_ptr<Backend> make_cpu_backend() { return std::make_unique<CpuBackend>(); }

KktStats kkt_on_original(const ScaledProblem& sp, const std::vector<double>& x, const std::vector<double>& y_min) {
  return CpuBackend::kkt_original(sp, x, y_min);
}

std::unique_ptr<Backend> make_backend(bool use_gpu, std::string& error) {
  if (!use_gpu) return make_cpu_backend();
#if defined(PS26119_HAVE_CUDA)
  return make_cuda_backend();
#else
  error = "this build has no CUDA backend (configure with -DPS26119_ENABLE_CUDA=ON on a GPU machine)";
  return nullptr;
#endif
}

}  // namespace ps26119::pdhg
