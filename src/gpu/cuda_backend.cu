// cuda_backend.cu — CUDA implementation of pdhg::Backend (compiled only with
// -DPS26119_ENABLE_CUDA=ON; the Mac build never sees this file).
//
// The math of every kernel is identical to src/pdhg/cpu_backend.cpp (the CPU reference);
// tests/unit/test_gpu.cpp checks both backends op by op and end to end.
//
// Design (CLAUDE.md §9, docs/RESEARCH.md §5.2):
//   * All problem data resident on the device: CSR of Ã and Ãᵀ (fp64 + fp32 copies),
//     c̃ and bounds (fp64 + fp32), and the ORIGINAL A, Aᵀ in fp64 for KKT evaluation.
//   * Our own SpMV kernels (no cuSPARSE): rows are split at setup into
//       - short rows: a sub-warp of TPR ∈ {4, 8, 16, 32} threads per row, TPR chosen from
//         the mean row length (warp-per-row when rows are long enough);
//       - long rows (> kLongRow nonzeros): one thread block per row with a block reduction
//         ("vector" variant), so a few dense linking rows do not serialise a warp.
//   * Fused elementwise kernels for the PDHG / reflected-Halpern updates (one pass over
//     x or y per step, projection included).
//   * Reductions (dot, norms, KKT) are two-pass block reductions with a FIXED grid, so
//     results are deterministic run to run. They are the only host synchronisations,
//     and the engines call them only every K iterations (termination / restart checks).
//   * Working precision fp64 or fp32 (mixed): iterate buffers and SpMV in the working
//     type; reductions and KKT always accumulate in fp64.
// Approach informed by MIT-Lu-Lab/cuPDLPx src/solver.cu (fused update kernels) — our own
// code; cuPDLPx uses cuSPARSE for SpMV, we do not.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "pdhg/backend.h"

namespace ps26119::pdhg {
namespace {

#define PS_CUDA_CHECK(expr)                                                                        \
  do {                                                                                             \
    cudaError_t e_ = (expr);                                                                       \
    if (e_ != cudaSuccess)                                                                         \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_) + " in " #expr); \
  } while (0)

constexpr int kBlock = 256;
constexpr int kReduceBlocks = 1024;  // fixed ⇒ deterministic reductions
constexpr std::int64_t kLongRow = 1024;

// ------------------------------------------------------------------ device memory (RAII)
template <class T>
class DevBuf {
 public:
  DevBuf() = default;
  explicit DevBuf(std::size_t n) { alloc(n); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  DevBuf(DevBuf&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr, o.n_ = 0; }
  DevBuf& operator=(DevBuf&& o) noexcept {
    if (this != &o) {
      release();
      p_ = o.p_, n_ = o.n_;
      o.p_ = nullptr, o.n_ = 0;
    }
    return *this;
  }
  ~DevBuf() { release(); }

  void alloc(std::size_t n) {
    release();
    if (n) {
      const cudaError_t e = cudaMalloc(&p_, n * sizeof(T));
      if (e == cudaErrorMemoryAllocation) {
        (void)cudaGetLastError();  // clear it (not sticky): the caller gets a clean NotSolved
        p_ = nullptr;
        throw std::bad_alloc();  // solve() reports NotSolved "out of memory", not NumericalError
      }
      PS_CUDA_CHECK(e);
    }
    n_ = n;
  }
  void upload(const T* h, std::size_t n) {
    if (n != n_) alloc(n);
    if (n) PS_CUDA_CHECK(cudaMemcpy(p_, h, n * sizeof(T), cudaMemcpyHostToDevice));
  }
  void upload(const std::vector<T>& h) { upload(h.data(), h.size()); }
  void download(T* h) const {
    if (n_) PS_CUDA_CHECK(cudaMemcpy(h, p_, n_ * sizeof(T), cudaMemcpyDeviceToHost));
  }
  T* get() { return p_; }
  const T* get() const { return p_; }
  std::size_t size() const { return n_; }

 private:
  void release() {
    if (p_) cudaFree(p_);  // never throw from a destructor
    p_ = nullptr;
    n_ = 0;
  }
  T* p_ = nullptr;
  std::size_t n_ = 0;
};

inline int grid_for(std::int64_t n) {
  const std::int64_t g = (n + kBlock - 1) / kBlock;
  return static_cast<int>(std::max<std::int64_t>(1, std::min<std::int64_t>(g, 65535LL * 32)));
}

template <class T>
__device__ __forceinline__ T dclamp(T v, T lo, T hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// ------------------------------------------------------------------ block reductions
__device__ __forceinline__ double warp_sum(double v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}
__device__ __forceinline__ double warp_max(double v) {
  for (int off = 16; off > 0; off >>= 1) v = fmax(v, __shfl_down_sync(0xffffffffu, v, off));
  return v;
}
// Result valid in thread 0. Safe to call repeatedly in one kernel.
__device__ double block_sum(double v) {
  __shared__ double sh[32];
  __syncthreads();
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) sh[wid] = v;
  __syncthreads();
  v = threadIdx.x < (blockDim.x >> 5) ? sh[lane] : 0.0;
  if (wid == 0) v = warp_sum(v);
  return v;
}
__device__ double block_max(double v) {
  __shared__ double sh[32];
  __syncthreads();
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  v = warp_max(v);
  if (lane == 0) sh[wid] = v;
  __syncthreads();
  v = threadIdx.x < (blockDim.x >> 5) ? sh[lane] : 0.0;
  if (wid == 0) v = warp_max(v);
  return v;
}

// ------------------------------------------------------------------ SpMV kernels
// Sub-warp CSR SpMV: TPR threads per row, rows given by an index list (or identity).
template <int TPR, class T, class Acc>
__global__ void k_spmv_subwarp(int n_rows, const int* __restrict__ rows, const std::int64_t* __restrict__ rp,
                               const int* __restrict__ col, const T* __restrict__ val, const T* __restrict__ x,
                               T* __restrict__ y) {
  const std::int64_t tid = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t g = tid / TPR;
  const int lane = threadIdx.x % TPR;
  Acc s = 0;
  int r = -1;
  if (g < n_rows) {
    r = rows ? rows[g] : static_cast<int>(g);
    for (std::int64_t k = rp[r] + lane; k < rp[r + 1]; k += TPR) s += static_cast<Acc>(val[k]) * static_cast<Acc>(x[col[k]]);
  }
  // every lane of the warp takes part in the shuffles (no early return above)
  for (int off = TPR / 2; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off, TPR);
  if (g < n_rows && lane == 0) y[r] = static_cast<T>(s);
}

// Block-per-row CSR SpMV for long rows.
template <class T>
__global__ void k_spmv_block(const int* __restrict__ rows, const std::int64_t* __restrict__ rp,
                             const int* __restrict__ col, const T* __restrict__ val, const T* __restrict__ x,
                             T* __restrict__ y) {
  const int r = rows[blockIdx.x];
  double s = 0;
  for (std::int64_t k = rp[r] + threadIdx.x; k < rp[r + 1]; k += blockDim.x)
    s += static_cast<double>(val[k]) * static_cast<double>(x[col[k]]);
  s = block_sum(s);
  if (threadIdx.x == 0) y[r] = static_cast<T>(s);
}

// ------------------------------------------------------------------ elementwise kernels
template <class T>
__global__ void k_fill(std::int64_t n, T* v, T a) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x)
    v[i] = a;
}

template <class T>
__global__ void k_axpby(std::int64_t n, T a, const T* __restrict__ x, T b, T* __restrict__ y) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x)
    y[i] = a * x[i] + b * y[i];
}

template <class S, class D>
__global__ void k_convert(std::int64_t n, const S* __restrict__ s, D* __restrict__ d) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x)
    d[i] = static_cast<D>(s[i]);
}

template <class T>
__global__ void k_primal_step(std::int64_t n, const T* __restrict__ x, const T* __restrict__ aty,
                              const T* __restrict__ c, const T* __restrict__ l, const T* __restrict__ u, T tau,
                              T* __restrict__ out) {
  for (std::int64_t j = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; j < n;
       j += static_cast<std::int64_t>(gridDim.x) * blockDim.x)
    out[j] = dclamp<T>(x[j] - tau * (c[j] - aty[j]), l[j], u[j]);
}

template <class T>
__global__ void k_dual_step(std::int64_t m, const T* __restrict__ y, const T* __restrict__ ax,
                            const T* __restrict__ rl, const T* __restrict__ ru, T sigma, T* __restrict__ out) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < m;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const T v = ax[i] - y[i] / sigma;
    out[i] = y[i] - sigma * ax[i] + sigma * dclamp<T>(v, rl[i], ru[i]);
  }
}

// x̂ = proj(x − τ(c − aty)); x̄ = 2x̂ − x; x ← w(2ρ x̂ + (1−2ρ) x) + (1−w) x0
template <class T>
__global__ void k_r2h_primal(std::int64_t n, T* __restrict__ x, const T* __restrict__ x0, const T* __restrict__ aty,
                             const T* __restrict__ c, const T* __restrict__ l, const T* __restrict__ u, T tau, T w,
                             T a, T b, T* __restrict__ xhat, T* __restrict__ xbar) {
  for (std::int64_t j = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; j < n;
       j += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const T xj = x[j];
    const T h = dclamp<T>(xj - tau * (c[j] - aty[j]), l[j], u[j]);
    xhat[j] = h;
    xbar[j] = T(2) * h - xj;
    x[j] = w * (a * h + b * xj) + (T(1) - w) * x0[j];
  }
}

template <class T>
__global__ void k_r2h_dual(std::int64_t m, T* __restrict__ y, const T* __restrict__ y0, const T* __restrict__ ax,
                           const T* __restrict__ rl, const T* __restrict__ ru, T sigma, T w, T a, T b,
                           T* __restrict__ yhat, T* __restrict__ ybar) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < m;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const T yi = y[i];
    const T v = ax[i] - yi / sigma;
    const T h = yi - sigma * ax[i] + sigma * dclamp<T>(v, rl[i], ru[i]);
    yhat[i] = h;
    if (ybar) ybar[i] = T(2) * h - yi;
    y[i] = w * (a * h + b * yi) + (T(1) - w) * y0[i];
  }
}

// Σ a_i b_i (or Σ (a_i − b_i)² when diff) per block, into partial[blockIdx.x].
template <class T>
__global__ void k_dot_partial(std::int64_t n, const T* __restrict__ a, const T* __restrict__ b, bool diff,
                              double* __restrict__ partial) {
  double s = 0;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    if (diff) {
      const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
      s += d * d;
    } else {
      s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
  }
  s = block_sum(s);
  if (threadIdx.x == 0) partial[blockIdx.x] = s;
}

// Final reduction of `count` groups of q values (q ≤ 8): first `nsum` are sums, rest maxes.
__global__ void k_reduce_final(int count, int q, int nsum, const double* __restrict__ partial, double* __restrict__ out) {
  for (int k = 0; k < q; ++k) {
    double v = 0;
    for (int b = threadIdx.x; b < count; b += blockDim.x) {
      const double p = partial[static_cast<std::int64_t>(b) * q + k];
      v = k < nsum ? v + p : fmax(v, p);
    }
    v = k < nsum ? block_sum(v) : block_max(v);
    if (threadIdx.x == 0) out[k] = v;
  }
}

// ---- KKT (fp64, original space). See termination.h for the definitions.
template <class T>
__global__ void k_unscale(std::int64_t n, const T* __restrict__ s, const double* __restrict__ scale, double factor,
                          const double* __restrict__ lo, const double* __restrict__ up, double* __restrict__ out) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    double v = scale[i] * static_cast<double>(s[i]) * factor;
    if (lo) v = dclamp<double>(v, lo[i], up[i]);
    out[i] = v;
  }
}

__device__ __forceinline__ double elem_viol(double v, double lo, double up) {
  if (v < lo) return (lo - v) / (1.0 + fabs(lo));
  if (v > up) return (v - up) / (1.0 + fabs(up));
  return 0.0;
}

// per block: [rp2, rd2, d, pmax, dmax]  (3 sums, 2 maxes)
__global__ void k_kkt_rows(std::int64_t m, const double* __restrict__ ax, const double* __restrict__ y,
                           const double* __restrict__ rl, const double* __restrict__ ru, double* __restrict__ partial) {
  double rp2 = 0, rd2 = 0, d = 0, pmax = 0, dmax = 0;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < m;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const double lo = rl[i], up = ru[i], a = ax[i], yi = y[i];
    const double viol = a < lo ? lo - a : (a > up ? a - up : 0.0);
    rp2 += viol * viol;
    pmax = fmax(pmax, elem_viol(a, lo, up));
    if (yi > 0) {
      if (isfinite(lo)) d += yi * lo;
      else rd2 += yi * yi, dmax = fmax(dmax, yi);
    } else if (yi < 0) {
      if (isfinite(up)) d += yi * up;
      else rd2 += yi * yi, dmax = fmax(dmax, -yi);
    }
  }
  rp2 = block_sum(rp2);
  rd2 = block_sum(rd2);
  d = block_sum(d);
  pmax = block_max(pmax);
  dmax = block_max(dmax);
  if (threadIdx.x == 0) {
    double* o = partial + static_cast<std::int64_t>(blockIdx.x) * 5;
    o[0] = rp2, o[1] = rd2, o[2] = d, o[3] = pmax, o[4] = dmax;
  }
}

// per block: [rd2, d, p, pmax, dmax]
__global__ void k_kkt_cols(std::int64_t n, const double* __restrict__ x, const double* __restrict__ aty,
                           const double* __restrict__ c, const double* __restrict__ l, const double* __restrict__ u,
                           double* __restrict__ partial) {
  double rd2 = 0, d = 0, p = 0, pmax = 0, dmax = 0;
  for (std::int64_t j = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; j < n;
       j += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const double cj = c[j], lo = l[j], up = u[j], xj = x[j];
    p += cj * xj;
    pmax = fmax(pmax, elem_viol(xj, lo, up));
    const double lam = cj - aty[j];
    double z;
    if (isfinite(lo) && isfinite(up)) z = lam;
    else if (isfinite(lo)) z = fmax(lam, 0.0);
    else if (isfinite(up)) z = fmin(lam, 0.0);
    else z = 0.0;
    rd2 += (lam - z) * (lam - z);
    dmax = fmax(dmax, fabs(lam - z));
    if (z > 0) d += z * lo;
    else if (z < 0) d += z * up;
  }
  rd2 = block_sum(rd2);
  d = block_sum(d);
  p = block_sum(p);
  pmax = block_max(pmax);
  dmax = block_max(dmax);
  if (threadIdx.x == 0) {
    double* o = partial + static_cast<std::int64_t>(blockIdx.x) * 5;
    o[0] = rd2, o[1] = d, o[2] = p, o[3] = pmax, o[4] = dmax;
  }
}

// ------------------------------------------------------------------ device CSR
template <class T>
struct DevCsr {
  int rows = 0, cols = 0;
  DevBuf<std::int64_t> rp;
  DevBuf<int> col;
  DevBuf<T> val;
  DevBuf<int> short_rows, long_rows;  // empty short_rows ⇒ identity over all rows
  int n_short = 0, n_long = 0, tpr = 32;

  void upload(const la::Csr<double>& a) {
    rows = a.rows;
    cols = a.cols;
    rp.upload(a.row_ptr);
    col.upload(a.col);
    std::vector<T> v(a.val.begin(), a.val.end());
    val.upload(v);
    std::vector<int> s, l;
    std::int64_t short_nnz = 0;
    for (int i = 0; i < a.rows; ++i) {
      const std::int64_t len = a.row_ptr[i + 1] - a.row_ptr[i];
      if (len > kLongRow) l.push_back(i);
      else s.push_back(i), short_nnz += len;
    }
    n_short = static_cast<int>(s.size());
    n_long = static_cast<int>(l.size());
    if (n_long > 0) {
      short_rows.upload(s);
      long_rows.upload(l);
    }
    const double mean = n_short ? static_cast<double>(short_nnz) / n_short : 0.0;
    tpr = mean <= 4 ? 4 : mean <= 8 ? 8 : mean <= 16 ? 16 : 32;
  }

  // y = this · x. Short rows accumulate in T (fp32 in mixed mode), long rows in fp64.
  void multiply(const T* x, T* y) const {
    if (n_short > 0) {
      const int* rows_ptr = n_long > 0 ? short_rows.get() : nullptr;
      const std::int64_t threads = static_cast<std::int64_t>(n_short) * tpr;
      const int grid = static_cast<int>((threads + kBlock - 1) / kBlock);
      switch (tpr) {
        case 4: k_spmv_subwarp<4, T, T><<<grid, kBlock>>>(n_short, rows_ptr, rp.get(), col.get(), val.get(), x, y); break;
        case 8: k_spmv_subwarp<8, T, T><<<grid, kBlock>>>(n_short, rows_ptr, rp.get(), col.get(), val.get(), x, y); break;
        case 16: k_spmv_subwarp<16, T, T><<<grid, kBlock>>>(n_short, rows_ptr, rp.get(), col.get(), val.get(), x, y); break;
        default: k_spmv_subwarp<32, T, T><<<grid, kBlock>>>(n_short, rows_ptr, rp.get(), col.get(), val.get(), x, y); break;
      }
    }
    if (n_long > 0) k_spmv_block<T><<<n_long, kBlock>>>(long_rows.get(), rp.get(), col.get(), val.get(), x, y);
    PS_CUDA_CHECK(cudaGetLastError());
  }
};

template <class T>
struct DevData {
  DevCsr<T> A, At;
  DevBuf<T> c, l, u, rl, ru;
  std::vector<DevBuf<T>> vecs;
};

// ------------------------------------------------------------------ the backend
class CudaBackend final : public Backend {
 public:
  CudaBackend() {
    int count = 0;
    PS_CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) throw std::runtime_error("no CUDA device");
    cudaDeviceProp prop{};
    PS_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    name_ = std::string("cuda:") + prop.name;
    partial_.alloc(static_cast<std::size_t>(kReduceBlocks) * 5 * 2);
    result_.alloc(8);
  }

  std::string name() const override { return name_; }

  void setup(const ScaledProblem& sp) override {
    sp_ = &sp;
    load(d_, sp);
    load(f_, sp);
    // original-space data (fp64) for KKT
    A_orig_.upload(sp.A_orig);
    At_orig_.upload(sp.At_orig);
    const Model& M = *sp.original;
    std::vector<double> c(sp.n);
    double c2 = 0, cinf = 0, b2 = 0;
    for (int j = 0; j < sp.n; ++j) {
      c[j] = M.sense * M.obj[j];
      c2 += c[j] * c[j];
      cinf = std::max(cinf, std::fabs(c[j]));
    }
    for (int i = 0; i < sp.m; ++i) {
      const double bl = std::isfinite(M.row_lower[i]) ? std::fabs(M.row_lower[i]) : 0.0;
      const double bu = std::isfinite(M.row_upper[i]) ? std::fabs(M.row_upper[i]) : 0.0;
      b2 += std::max(bl, bu) * std::max(bl, bu);
    }
    c_norm_ = std::sqrt(c2);
    c_inf_ = cinf;
    b_norm_ = std::sqrt(b2);
    oc_.upload(c);
    ol_.upload(M.col_lower);
    ou_.upload(M.col_upper);
    orl_.upload(M.row_lower);
    oru_.upload(M.row_upper);
    col_scale_.upload(sp.col_scale);
    row_scale_.upload(sp.row_scale);
    xo_.alloc(sp.n);
    yo_.alloc(sp.m);
    axo_.alloc(sp.m);
    atyo_.alloc(sp.n);
  }

  void set_precision(Precision p) override {
    if (p == prec_) return;
    for (std::size_t k = 0; k < sizes_.size(); ++k) {
      const auto n = static_cast<std::int64_t>(sizes_[k]);
      if (p == Precision::Mixed) k_convert<double, float><<<grid_for(n), kBlock>>>(n, d_.vecs[k].get(), f_.vecs[k].get());
      else k_convert<float, double><<<grid_for(n), kBlock>>>(n, f_.vecs[k].get(), d_.vecs[k].get());
    }
    PS_CUDA_CHECK(cudaGetLastError());
    prec_ = p;
  }
  Precision precision() const override { return prec_; }

  int create(Space s) override {
    const std::size_t len = s == Space::Primal ? sp_->n : sp_->m;
    d_.vecs.emplace_back(len);
    f_.vecs.emplace_back(len);
    sizes_.push_back(len);
    return static_cast<int>(sizes_.size()) - 1;
  }

  void upload(int v, const std::vector<double>& src) override {
    if (prec_ == Precision::Fp64) {
      d_.vecs[v].upload(src);
    } else {
      std::vector<float> f(src.begin(), src.end());
      f_.vecs[v].upload(f);
    }
  }
  void download(int v, std::vector<double>& dst) override {
    dst.resize(sizes_[v]);
    if (prec_ == Precision::Fp64) {
      d_.vecs[v].download(dst.data());
    } else {
      std::vector<float> f(sizes_[v]);
      f_.vecs[v].download(f.data());
      dst.assign(f.begin(), f.end());
    }
  }
  void fill(int v, double a) override {
    const auto n = static_cast<std::int64_t>(sizes_[v]);
    with([&](auto& D) {
      using T = std::remove_pointer_t<decltype(D.vecs[v].get())>;
      k_fill<T><<<grid_for(n), kBlock>>>(n, D.vecs[v].get(), static_cast<T>(a));
    });
    PS_CUDA_CHECK(cudaGetLastError());
  }
  void copy(int dst, int src) override {
    with([&](auto& D) {
      using T = std::remove_pointer_t<decltype(D.vecs[dst].get())>;
      PS_CUDA_CHECK(cudaMemcpyAsync(D.vecs[dst].get(), D.vecs[src].get(), sizes_[dst] * sizeof(T),
                                    cudaMemcpyDeviceToDevice));
    });
  }
  void axpby(double a, int x, double b, int y) override {
    const auto n = static_cast<std::int64_t>(sizes_[y]);
    with([&](auto& D) {
      using T = std::remove_pointer_t<decltype(D.vecs[y].get())>;
      k_axpby<T><<<grid_for(n), kBlock>>>(n, static_cast<T>(a), D.vecs[x].get(), static_cast<T>(b), D.vecs[y].get());
    });
    PS_CUDA_CHECK(cudaGetLastError());
  }
  void spmv(int x, int out) override { with([&](auto& D) { D.A.multiply(D.vecs[x].get(), D.vecs[out].get()); }); }
  void spmv_t(int y, int out) override { with([&](auto& D) { D.At.multiply(D.vecs[y].get(), D.vecs[out].get()); }); }

  double dot(int a, int b) override { return reduce_pair(a, b, false); }
  double norm2(int a) override { return std::sqrt(reduce_pair(a, a, false)); }
  double diff_norm2(int a, int b) override { return std::sqrt(reduce_pair(a, b, true)); }

  void primal_step(int x, int aty, double tau, int xhat) override {
    const auto n = static_cast<std::int64_t>(sp_->n);
    with([&](auto& D) {
      using T = std::remove_pointer_t<decltype(D.c.get())>;
      k_primal_step<T><<<grid_for(n), kBlock>>>(n, D.vecs[x].get(), D.vecs[aty].get(), D.c.get(), D.l.get(), D.u.get(),
                                                static_cast<T>(tau), D.vecs[xhat].get());
    });
    PS_CUDA_CHECK(cudaGetLastError());
  }
  void dual_step(int y, int ax, double sigma, int yhat) override {
    const auto m = static_cast<std::int64_t>(sp_->m);
    with([&](auto& D) {
      using T = std::remove_pointer_t<decltype(D.c.get())>;
      k_dual_step<T><<<grid_for(m), kBlock>>>(m, D.vecs[y].get(), D.vecs[ax].get(), D.rl.get(), D.ru.get(),
                                              static_cast<T>(sigma), D.vecs[yhat].get());
    });
    PS_CUDA_CHECK(cudaGetLastError());
  }
  void r2h_primal(int x, int x0, int aty, double tau, double w, double rho, int xhat, int xbar) override {
    const auto n = static_cast<std::int64_t>(sp_->n);
    with([&](auto& D) {
      using T = std::remove_pointer_t<decltype(D.c.get())>;
      k_r2h_primal<T><<<grid_for(n), kBlock>>>(n, D.vecs[x].get(), D.vecs[x0].get(), D.vecs[aty].get(), D.c.get(),
                                               D.l.get(), D.u.get(), static_cast<T>(tau), static_cast<T>(w),
                                               static_cast<T>(2 * rho), static_cast<T>(1 - 2 * rho),
                                               D.vecs[xhat].get(), D.vecs[xbar].get());
    });
    PS_CUDA_CHECK(cudaGetLastError());
  }
  void r2h_dual(int y, int y0, int ax, double sigma, double w, double rho, int yhat, int ybar) override {
    const auto m = static_cast<std::int64_t>(sp_->m);
    with([&](auto& D) {
      using T = std::remove_pointer_t<decltype(D.c.get())>;
      k_r2h_dual<T><<<grid_for(m), kBlock>>>(m, D.vecs[y].get(), D.vecs[y0].get(), D.vecs[ax].get(), D.rl.get(),
                                             D.ru.get(), static_cast<T>(sigma), static_cast<T>(w),
                                             static_cast<T>(2 * rho), static_cast<T>(1 - 2 * rho), D.vecs[yhat].get(),
                                             ybar >= 0 ? D.vecs[ybar].get() : nullptr);
    });
    PS_CUDA_CHECK(cudaGetLastError());
  }

  KktStats kkt(int xs, int ys) override {
    const auto n = static_cast<std::int64_t>(sp_->n), m = static_cast<std::int64_t>(sp_->m);
    with([&](auto& D) {
      using T = std::remove_pointer_t<decltype(D.c.get())>;
      k_unscale<T><<<grid_for(n), kBlock>>>(n, D.vecs[xs].get(), col_scale_.get(), 1.0 / sp_->bound_scale, ol_.get(),
                                            ou_.get(), xo_.get());
      k_unscale<T><<<grid_for(m), kBlock>>>(m, D.vecs[ys].get(), row_scale_.get(), 1.0 / sp_->obj_scale, nullptr,
                                            nullptr, yo_.get());
    });
    A_orig_.multiply(xo_.get(), axo_.get());
    At_orig_.multiply(yo_.get(), atyo_.get());
    const int gr = std::min<int>(kReduceBlocks, grid_for(std::max<std::int64_t>(m, 1)));
    const int gc = std::min<int>(kReduceBlocks, grid_for(std::max<std::int64_t>(n, 1)));
    double* prow = partial_.get();
    double* pcol = partial_.get() + static_cast<std::size_t>(kReduceBlocks) * 5;
    k_kkt_rows<<<gr, kBlock>>>(m, axo_.get(), yo_.get(), orl_.get(), oru_.get(), prow);
    k_kkt_cols<<<gc, kBlock>>>(n, xo_.get(), atyo_.get(), oc_.get(), ol_.get(), ou_.get(), pcol);
    // rows: [rp2, rd2, d | pmax, dmax]; cols: [rd2, d, p | pmax, dmax]
    k_reduce_final<<<1, kBlock>>>(gr, 5, 3, prow, result_.get());
    k_reduce_final<<<1, kBlock>>>(gc, 5, 3, pcol, scratch());
    PS_CUDA_CHECK(cudaGetLastError());
    double r[5], s[5];
    PS_CUDA_CHECK(cudaMemcpy(r, result_.get(), 5 * sizeof(double), cudaMemcpyDeviceToHost));
    PS_CUDA_CHECK(cudaMemcpy(s, scratch(), 5 * sizeof(double), cudaMemcpyDeviceToHost));
    KktStats k;
    k.primal_residual = std::sqrt(r[0]);
    k.dual_residual = std::sqrt(r[1] + s[0]);
    k.dual_obj = r[2] + s[1];
    k.primal_obj = s[2];
    k.primal_max_rel = std::max(r[3], s[3]);
    k.dual_max_rel = std::max(r[4], s[4]) / (1.0 + c_inf_);
    k.b_norm = b_norm_;
    k.c_norm = c_norm_;
    return k;
  }

  void sync() override { PS_CUDA_CHECK(cudaDeviceSynchronize()); }

 private:
  template <class T>
  static void load(DevData<T>& D, const ScaledProblem& sp) {
    D.A.upload(sp.A);
    D.At.upload(sp.At);
    auto up = [](DevBuf<T>& b, const std::vector<double>& v) {
      std::vector<T> t(v.begin(), v.end());
      b.upload(t);
    };
    up(D.c, sp.c);
    up(D.l, sp.col_lower);
    up(D.u, sp.col_upper);
    up(D.rl, sp.row_lower);
    up(D.ru, sp.row_upper);
  }

  template <class F>
  void with(F&& f) {
    if (prec_ == Precision::Fp64) f(d_);
    else f(f_);
  }

  double reduce_pair(int a, int b, bool diff) {
    const auto n = static_cast<std::int64_t>(sizes_[a]);
    const int g = std::min<int>(kReduceBlocks, grid_for(std::max<std::int64_t>(n, 1)));
    with([&](auto& D) {
      using T = std::remove_pointer_t<decltype(D.vecs[a].get())>;
      k_dot_partial<T><<<g, kBlock>>>(n, D.vecs[a].get(), D.vecs[b].get(), diff, partial_.get());
    });
    k_reduce_final<<<1, kBlock>>>(g, 1, 1, partial_.get(), result_.get());
    PS_CUDA_CHECK(cudaGetLastError());
    double r = 0;
    PS_CUDA_CHECK(cudaMemcpy(&r, result_.get(), sizeof(double), cudaMemcpyDeviceToHost));
    return r;
  }

  double* scratch() { return scratch_.size() ? scratch_.get() : (scratch_.alloc(8), scratch_.get()); }

  std::string name_;
  const ScaledProblem* sp_ = nullptr;
  Precision prec_ = Precision::Fp64;
  DevData<double> d_;
  DevData<float> f_;
  std::vector<std::size_t> sizes_;
  // original-space fp64 data for kkt()
  DevCsr<double> A_orig_, At_orig_;
  DevBuf<double> oc_, ol_, ou_, orl_, oru_, col_scale_, row_scale_, xo_, yo_, axo_, atyo_;
  DevBuf<double> partial_, result_, scratch_;
  double c_norm_ = 0, c_inf_ = 0, b_norm_ = 0;
};

}  // namespace

std::unique_ptr<Backend> make_cuda_backend() { return std::make_unique<CudaBackend>(); }

}  // namespace ps26119::pdhg
