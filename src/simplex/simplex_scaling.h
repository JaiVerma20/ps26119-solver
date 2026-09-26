// scaling.hpp - row and column scaling of the constraint matrix.
//
// Real models mix coefficients like 1e-4 and 1e+6 in the same row. Scaling
// replaces A by R A C (R, C diagonal) so that entries are close to 1, which
// makes pivot choices and tolerances meaningful. Factors are powers of two,
// so scaling and unscaling are exact in floating point.
#pragma once

#include <vector>

#include "gpuopt/problem.hpp"

namespace gpuopt {

struct Scaling {
  std::vector<double> row;  // R: scaled row i = row[i] * original row i
  std::vector<double> col;  // C: original x_j = col[j] * scaled x_j
};

// Geometric-mean scaling (alternating row / column passes), then column
// equilibration so the largest entry of every column is about 1.
Scaling compute_scaling(const SparseMatrixCSC& A, int passes = 6);

// Max / min absolute nonzero of R A C (1 for an empty matrix).
double scaled_dynamism(const SparseMatrixCSC& A, const Scaling& s);

}  // namespace gpuopt
