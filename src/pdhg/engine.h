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
  // PDLP-style (defaults: cuPDLP.jl / PDLP)
  double pdlp_restart_necessary = 0.8;
  double pdlp_primal_weight_smoothing = 0.5;  // θ
};

EngineOptions engine_options_from(const Options& o);

class PrecisionPolicy {
 public:
  PrecisionPolicy(Backend& b, const EngineOptions& opt);
  // Called at every KKT check. Returns true when the backend was just promoted to fp64
  // (the engine should then restart from the current point).
  bool on_check(double rel_kkt, std::int64_t iteration);
  std::int64_t switch_iteration() const { return switched_at_; }

 private:
  Backend& b_;
  bool mixed_;
  double target_;
  double best_ = 1e300;
  int checks_since_best_ = 0;
  std::int64_t switched_at_ = -1;
};

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

  // Builds the Solution from the scaled point (x̂, ŷ) held in backend vectors.
  Solution finish(Status status, int x_scaled, int y_scaled, std::int64_t iterations, const std::string& message);

 private:
  const Model& model_;
  const EngineOptions& opt_;
  const char* name_;
  ScaledProblem sp_;
  std::unique_ptr<Backend> backend_;
  std::string error_;
  double eta_ = 1.0;
  std::chrono::steady_clock::time_point t0_;
  double setup_seconds_ = 0;
  std::int64_t fast_iterations_ = -1;
  double fast_seconds_ = -1;
};

}  // namespace ps26119::pdhg
