// mps_reader.hpp - Layer 1 front end: reads MPS files into an LpProblem.
//
// Supported sections: NAME, OBJSENSE, OBJNAME, ROWS, COLUMNS (with integer
// MARKER blocks), RHS, RANGES, BOUNDS (UP LO FX FR MI PL BV LI UI), ENDATA.
// Both free format (whitespace separated) and fixed format (column
// positions, names may contain spaces) are supported; kAuto tries free
// format first and falls back to fixed.
//
// Not yet supported (reported as an error, never silently ignored):
// gzip input, QUADOBJ/QMATRIX (Layer 6), SOS, semi-continuous bounds.
#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "gpuopt/problem.hpp"

namespace gpuopt {

enum class MpsFormat { kAuto, kFree, kFixed };

struct MpsReadOptions {
  MpsFormat format = MpsFormat::kAuto;
  // Absolute values at or above this are treated as infinite in bounds,
  // RHS and RANGES (the usual MPS convention).
  double infinity_threshold = 1e30;
  // If true, integer MARKER columns without explicit bounds get [0, 1]
  // (old IBM convention). Default false gives [0, +inf) like modern solvers.
  bool integer_default_binary = false;
};

class MpsParseError : public std::runtime_error {
 public:
  MpsParseError(const std::string& message, int line);
  int line() const { return line_; }

 private:
  int line_;
};

struct MpsReadResult {
  LpProblem problem;
  std::vector<std::string> warnings;
  MpsFormat format_used = MpsFormat::kFree;
};

MpsReadResult read_mps_file(const std::string& path, const MpsReadOptions& options = {});
MpsReadResult read_mps_string(std::string_view text, const MpsReadOptions& options = {});

}  // namespace gpuopt
