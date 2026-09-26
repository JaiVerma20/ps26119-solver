// simplex_scaling.h - row and column scaling of the constraint matrix (simplex engine).
//
// Origin: gpuopt include/gpuopt/simplex/scaling.hpp (Shivanshu Vats, c192dd0).
// The first-order engines use their own scaling (src/pdhg/scaling.h).
//
// Real models mix coefficients like 1e-4 and 1e+6 in the same row. Scaling
// replaces A by R A C (R, C diagonal) so that entries are close to 1, which
// makes pivot choices and tolerances meaningful. Factors are powers of two,
// so scaling and unscaling are exact in floating point.
#pragma once

#include <vector>

#include "la/csc.h"

namespace ps26119::simplex {

struct Scaling {
  std::vector<double> row;  // R: scaled row i = row[i] * original row i
  std::vector<double> col;  // C: original x_j = col[j] * scaled x_j
};

// Geometric-mean scaling (alternating row / column passes), then column
// equilibration so the largest entry of every column is about 1.
Scaling compute_scaling(const la::SparseMatrixCSC& A, int passes = 6);

// Max / min absolute nonzero of R A C (1 for an empty matrix).
double scaled_dynamism(const la::SparseMatrixCSC& A, const Scaling& s);

}  // namespace ps26119::simplex
