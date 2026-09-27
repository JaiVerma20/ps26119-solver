// solution_writer.h — write a Solution in the text format tools/verify.py reads.
//
//   PS26119-SOLUTION 1
//   <key> <value>            model, name, status, engine, precision, objective, ...
//   COLUMNS <n>
//   <j> <x_j> <z_j> <name>   %.17g, so values round-trip exactly
//   ROWS <m>
//   <i> <activity_i> <y_i> <name>
//   DUAL_RAY <m> | PRIMAL_RAY <n>      optional certificates (Infeasible / Unbounded),
//   <index> <value>                    one line per entry (core/certificates.h)
//   END
//
// Duals follow the convention in include/ps26119/solution.h (z = c − Aᵀy, HiGHS signs).
#pragma once

#include <string>

#include "ps26119/model.h"
#include "ps26119/solution.h"

namespace ps26119::io {

bool write_solution(const std::string& path, const Model& model, const Solution& sol, std::string& error);

}  // namespace ps26119::io
