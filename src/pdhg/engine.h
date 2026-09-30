// engine.h — options and shared machinery for the first-order LP engines
// (PDLP-style restarted PDHG and r²HPDHG).
//
// Shared pieces (engine.cpp):
//   * EngineContext: scaling, backend creation, step size, timing, milestones.
//   * PrecisionPolicy: mixed precision = fp32 iterate + SpMV until either the target is
//     reached (decided on fp64 KKT of the original problem, so an fp32 run never claims
//     more than it has) or fp32 stalls / reaches tol::kMixedPrecisionSwitch; then the
//     iterate is promoted to fp64 and the engine restarts from it ("final polish").
//   * finish(): unscale, compute z = c − Aᵀy, activities, objective, and set the status.
#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include "pdhg/backend.h"
#include "pdhg/scaling.h"
#include "ps26119/model.h"
#include "ps26119/options.h"
#include "ps26119/solution.h"

namespace ps26119::pdhg {

struct EngineOptions {
  double tolerance = tol::kFirstOrderHigh;
  std::vector<double> warm_x, warm_y;  // see Options
  double warm_primal_weight = 0.0;
  double kkt_b_norm = -1, kkt_c_norm = -1, kkt_obj_shift = 0;  // see Options
  double time_limit = 3600.0;
  std::int64_t iteration_limit = 100'000'000;
  int check_every = 64;  // K: iterations between KKT / restart evaluations
  Precision precision = Precision::Fp64;
  bool use_gpu = false;
  int verbosity = 0;
  ScalingOptions scaling;
  // r²HPDHG (defaults: cuPDLPx)
  double reflection = 1.0;  // ρ
  double restart_sufficient = 0.2, restart_necessary = 0.5, restart_artificial = 0.36;
  double pid_kp = 0.99, pid_ki = 0.01, pid_kd = 0.0, pid_integral_decay = 0.3;
  // Safeguard (ours): |Δ log ω| per restart ≤ this. log(10) ⇒ at most ×10 / ÷10 per restart.
  double pid_max_log_step = 2.302585092994046;
  // Safeguard (ours): PID displacements below pid_noise_rel·(1 + ‖iterate‖) count as degenerate
  // (rounding noise); see r2hpdhg.h. 0 = cuPDLPx behaviour (absolute 1e-16 guard only).
  double pid_noise_rel = 1e-10;
  // PDLP-style (defaults: cuPDLP.jl / PDLP)
  double pdlp_restart_necessary = 0.8;
  double pdlp_primal_weight_smoothing = 0.5;  // θ
};

EngineOptions engine_options_from(const Options& o);
// Applies Options::engine_params; returns the first unknown name, or "" if all are known.
std::string apply_engine_params(const Options& o, EngineOptions& e);

class PrecisionPolicy {
 public:
  PrecisionPolicy(Backend& b, const EngineOptions& opt);
  // Called at every KKT check. Returns true when the backend was just promoted to fp64
  // (the engine should then restart from the current point).
  bool on_check(double rel_kkt, std::int64_t iteration);
  std::int64_t switch_iteration() const { return switched_at_; }
  // Warm start already more accurate than fp32 can hold: go straight to fp64.
  void on_warm_start(double warm_rel_kkt);

 private:
  Backend& b_;
  bool mixed_;
  double target_;
  double best_ = 1e300;
  int checks_since_best_ = 0;
  std::int64_t switched_at_ = -1;
};

// Result of the ray test (infeasibility detection), computed in fp64 on the ORIGINAL problem
// from a scaled direction (Δx, Δy) = T(z) − z (r²HPDHG) or z_k − z_{k−1} (PDLP-style).
// When an LP is infeasible or unbounded these differences converge to the infimal
// displacement vector, whose parts are certificates (Applegate, Lubin, Hinder,
// "Infeasibility detection with primal-dual hybrid gradient for LP", Math. Prog. 2024;
// approach informed by cuPDLPx src/utils.cu compute_infeasibility_information).
//   dual ray r (min form), projected so that r_i > 0 only if rl_i is finite and r_i < 0
//   only if ru_i is finite; λ = −Aᵀr;
//     violation_j = λ_j⁺·[l_j = −∞] + λ_j⁻·[u_j = +∞]
//     objective   = Σ_i (r_i>0 ? rl_i : ru_i)·r_i + Σ_j (λ_j>0 ? l_j : u_j)·λ_j (finite parts)
//     PRIMAL INFEASIBLE if objective > 0 and max violation ≤ ε·objective (Farkas).
//   primal ray d, projected onto the recession cone of [l,u]; row violation
//     (A d)_i⁻·[rl_i finite] + (A d)_i⁺·[ru_i finite];
//     DUAL INFEASIBLE if sense·cᵀd < 0 and max violation ≤ ε·(−sense·cᵀd).
//   Both after normalising by max(‖ray‖∞, ‖violation‖∞). ε = tol::kFirstOrderInfeasible.
struct RayTest {
  bool primal_infeasible = false;
  bool dual_infeasible = false;
  double dual_ray_objective = 0, dual_ray_violation = 0;
  double primal_ray_objective = 0, primal_ray_violation = 0;
  std::vector<double> dual_ray;    // r (min form, original rows): returned as Solution::dual_ray
  std::vector<double> primal_ray;  // d (original columns): returned as Solution::primal_ray
};
// Scratch vectors reused across ray tests (they run at every termination check).
struct RayWorkspace {
  std::vector<double> dx, dy, r, g, d, ad;
};
RayTest ray_test(const ScaledProblem& sp, const std::vector<double>& dx_scaled, const std::vector<double>& dy_scaled,
                 RayWorkspace* ws = nullptr);

class EngineContext {
 public:
  EngineContext(const Model& model, const EngineOptions& opt, const char* engine_name);

  bool ok() const { return backend_ != nullptr; }
  const std::string& error() const { return error_; }
  Backend& backend() { return *backend_; }
  const ScaledProblem& problem() const { return sp_; }
  double eta() const { return eta_; }
  double initial_primal_weight() const;
  double elapsed() const;

  // Records the 1e-4 milestone; returns true if the target tolerance is met.
  bool record(const KktStats& k, std::int64_t iteration);
  bool out_of_time() const { return elapsed() > opt_.time_limit; }

  // Ray test on backend vectors (scaled directions). Returns Infeasible / Unbounded when a
  // certificate is found (Unbounded additionally needs a nearly primal-feasible iterate:
  // a primal ray alone only proves "dual infeasible"), otherwise NotSolved = keep going.
  Status check_infeasibility(int dx_scaled, int dy_scaled, const KktStats& current, std::string& message);

  // Backend KKT of scaled (x, y) with the normalization overrides applied.
  KktStats kkt(int x_scaled, int y_scaled) { return adjust(backend_->kkt(x_scaled, y_scaled)); }
  KktStats adjust(KktStats k) const {
    if (opt_.kkt_b_norm >= 0) k.b_norm = opt_.kkt_b_norm;
    if (opt_.kkt_c_norm >= 0) k.c_norm = opt_.kkt_c_norm;
    k.obj_shift = opt_.kkt_obj_shift;
    return k;
  }

  // Loads Options::warm_x / warm_y (original space) into scaled backend vectors x, y.
  // x is clipped into its bounds and y projected onto its sign-feasible set. Returns true
  // if a warm start was applied.
  bool apply_warm_start(int x, int y, PrecisionPolicy& policy);
  // The engine reports its current primal weight so the Solution can carry it.
  void note_primal_weight(double w) { primal_weight_ = w; }

  // Builds the Solution from the scaled point (x̂, ŷ) held in backend vectors.
  Solution finish(Status status, int x_scaled, int y_scaled, std::int64_t iterations, const std::string& message);

 private:
  const Model& model_;
  const EngineOptions& opt_;
  const char* name_;
  ScaledProblem sp_;
  std::vector<double> dual_ray_, primal_ray_;  // certificate of the last Infeasible / Unbounded verdict
  RayWorkspace ray_ws_;
  std::unique_ptr<Backend> backend_;
  std::string error_;
  double eta_ = 1.0;
  std::chrono::steady_clock::time_point t0_;
  double setup_seconds_ = 0;
  double primal_weight_ = std::numeric_limits<double>::quiet_NaN();
  std::int64_t fast_iterations_ = -1;
  double fast_seconds_ = -1;
};

}  // namespace ps26119::pdhg
