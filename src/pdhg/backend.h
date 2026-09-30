// backend.h — the ONLY interface the first-order engines use for O(n)/O(nnz) work.
//
// Engine logic (restarts, primal weight, termination, precision policy) runs on the host
// in double. Everything that touches vectors goes through this interface so a CUDA (or
// later HIP / Metal) backend is a drop-in replacement, and the Mac can unit-test the exact
// same math with the CPU backend.
//
// Vectors live inside the backend and are addressed by integer handles. A backend has a
// WORKING precision for iterate vectors and SpMV (fp64, or fp32 for mixed precision);
// reductions (dot, norms) always accumulate in fp64 and kkt() always runs in fp64 on the
// original matrix, so restart and termination decisions are fp64 decisions.
//
// Sign conventions of the steps (scaled problem min c̃ᵀx s.t. Ãx ∈ [r̃l,r̃u], x ∈ [l̃,ũ];
// Lagrangian c̃ᵀx − yᵀÃx + min_{s∈[r̃l,r̃u]} yᵀs; derivation in pdhg/pdlp.h):
//   primal_step:  x̂ = proj_[l̃,ũ]( x − τ (c̃ − aty) )                 aty = Ãᵀy
//   dual_step:    ŷ = y − σ ax + σ · proj_[r̃l,r̃u]( ax − y/σ )          ax  = Ã x̄
// Fused r²HPDHG steps (w = Halpern weight, ρ = reflection coefficient):
//   r2h_primal:   x̂ = primal_step;  x̄ = 2x̂ − x;
//                 x ← w·(2ρ x̂ + (1−2ρ) x) + (1−w)·x0
//   r2h_dual:     ŷ = dual_step;    ȳ = 2ŷ − y (optional);
//                 y ← w·(2ρ ŷ + (1−2ρ) y) + (1−w)·y0
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pdhg/scaling.h"
#include "pdhg/termination.h"
#include "ps26119/options.h"

namespace ps26119::pdhg {

enum class Space { Primal, Dual };  // vectors of size n or m

class Backend {
 public:
  virtual ~Backend() = default;
  virtual std::string name() const = 0;
  // Uploads the scaled problem (Ã, Ãᵀ, c̃, bounds) and the original matrix for kkt().
  virtual void setup(const ScaledProblem& sp) = 0;
  // Working precision for iterate vectors and SpMV. Switching converts all vectors.
  virtual void set_precision(Precision p) = 0;
  virtual Precision precision() const = 0;

  virtual int create(Space s) = 0;
  virtual void upload(int v, const std::vector<double>& src) = 0;
  virtual void download(int v, std::vector<double>& dst) = 0;
  virtual void fill(int v, double a) = 0;
  virtual void copy(int dst, int src) = 0;
  virtual void axpby(double a, int x, double b, int y) = 0;  // y ← a·x + b·y
  virtual void spmv(int x, int out) = 0;                     // out ← Ã x
  virtual void spmv_t(int y, int out) = 0;                   // out ← Ãᵀ y
  virtual double dot(int a, int b) = 0;                      // fp64 accumulation
  virtual double norm2(int a) = 0;
  virtual double diff_norm2(int a, int b) = 0;  // ‖a − b‖₂

  virtual void primal_step(int x, int aty, double tau, int xhat_out) = 0;
  virtual void dual_step(int y, int ax, double sigma, int yhat_out) = 0;
  virtual void r2h_primal(int x, int x0, int aty, double tau, double w, double rho, int xhat_out, int xbar_out) = 0;
  virtual void r2h_dual(int y, int y0, int ax, double sigma, double w, double rho, int yhat_out, int ybar_out) = 0;

  // KKT statistics of the scaled iterate (x̃, ỹ) mapped back to the ORIGINAL problem,
  // computed in fp64 (see termination.h).
  virtual KktStats kkt(int x_scaled, int y_scaled) = 0;

  // Synchronise (GPU: wait for the stream). CPU: no-op.
  virtual void sync() {}
};

std::unique_ptr<Backend> make_cpu_backend();
// Plain view of an LP's vectors (original space, original sense), for KKT evaluation of
// data that is not a Model (e.g. one scenario of a batch sharing A).
struct LpView {
  int m, n, sense;
  const double *c, *col_lower, *col_upper, *row_lower, *row_upper;
};
// Scratch vectors for kkt_general (A x and Aᵀ y); reusing one across calls avoids allocating
// (and page-faulting) two O(m + n) vectors at every termination check.
struct KktWorkspace {
  std::vector<double> ax, aty;
};
// fp64 KKT statistics (termination.h) of (x, y_min) for the LP (A, lp); At = Aᵀ. Runs on the
// thread pool; the reductions use fixed chunks (la::parallel_reduce), so the result does not
// depend on the thread count.
KktStats kkt_general(const la::Csr<double>& A, const la::Csr<double>& At, const LpView& lp, const std::vector<double>& x,
                     const std::vector<double>& y_min, KktWorkspace* ws = nullptr);
// fp64 KKT statistics of an ORIGINAL-space point (x, y in min form); see termination.h.
KktStats kkt_on_original(const ScaledProblem& sp, const std::vector<double>& x, const std::vector<double>& y_min,
                         KktWorkspace* ws = nullptr);
#if defined(PS26119_HAVE_CUDA)
std::unique_ptr<Backend> make_cuda_backend();  // src/gpu/cuda_backend.cu
#endif
// Returns nullptr + message when the requested backend is unavailable in this build.
std::unique_ptr<Backend> make_backend(bool use_gpu, std::string& error);

}  // namespace ps26119::pdhg
