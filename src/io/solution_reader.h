// solution_reader.h — read a solution file written by solution_writer (for warm starts
// and tests). Only the header status/objective and the COLUMNS / ROWS blocks are parsed.
#pragma once

#include <string>

#include "ps26119/solution.h"

namespace ps26119::io {

// Fills sol.x, sol.z (from COLUMNS), sol.row_activity, sol.y (from ROWS), sol.status,
// sol.objective, sol.model_fingerprint. Returns false with `error` set on malformed input.
bool read_solution(const std::string& path, Solution& sol, std::string& error);

}  // namespace ps26119::io
