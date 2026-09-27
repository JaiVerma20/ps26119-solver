// certificates.h — independent checks of Infeasible / Unbounded claims (LP).
//
// Every LP status other than a limit is a mathematical claim. Optimal is re-checked by the
// gate with core/solution_checker; this file does the same for the other two:
//
// Infeasible — a Farkas certificate r ∈ Rᵐ (row multipliers, any sign). For every x with
//   rl ≤ Ax ≤ ru and l ≤ x ≤ u we have 0 = rᵀ(Ax) − (Aᵀr)ᵀx ≥ L₀(r), where L₀ is the
//   zero-objective Lagrangian bound Σᵢ min_{s∈[rlᵢ,ruᵢ]} rᵢ s + Σⱼ min_{x∈[lⱼ,uⱼ]} (−Aᵀr)ⱼ x.
//   So L₀(r) > 0 proves that no feasible x exists (Farkas' lemma: such an r exists iff the LP
//   is infeasible). Checked in two stages:
//     1. rigorous: L₀ evaluated with the rounding-proof machinery of core/safe_bound.h (the
//        certified bound of the model with objective 0). A PASS here is immune to rounding.
//     2. only if 1 cannot decide — a floating-point ray whose Aᵀr is 1e-16 instead of 0 on a
//        FREE column makes the rigorous L₀ = −∞ — the literature's tolerance test (PDLP;
//        Applegate et al. 2021): with the parts of Aᵀr (and r) that point at infinite bounds
//        counted as violation v, and s = max(‖r‖∞, v): L₀/s > 0 and v/s ≤ tol::kVerifyRay·L₀/s.
//   `rigorous` says which stage passed.
//
// Unbounded — a feasible point x (verifier tolerances, as for Optimal) and a ray d in the
//   recession cone with (sense·c)ᵀd < 0, tested with tol::kVerifyRay (not rounding-proof).
//
// tools/verify.py implements both checks again with an independent MPS reader.
#pragma once

#include <string>
#include <vector>

#include "ps26119/model.h"

namespace ps26119 {

struct CertificateCheck {
  bool present = false;  // a certificate of the right size was supplied
  bool passed = false;
  double measure = 0.0;  // Infeasible: certified L₀(r) (> 0 passes); Unbounded: −(sense·c)ᵀd for ‖d‖∞ = 1
  double violation = 0.0;  // normalised violation (Unbounded: recession cone; Infeasible stage 2)
  bool rigorous = false;   // Infeasible: passed the rounding-proof stage
  std::string detail;
};

CertificateCheck check_infeasibility_certificate(const Model& model, const std::vector<double>& dual_ray);
CertificateCheck check_unboundedness_certificate(const Model& model, const std::vector<double>& x,
                                                 const std::vector<double>& primal_ray);

}  // namespace ps26119
