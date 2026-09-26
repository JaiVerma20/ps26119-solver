// status.cpp — string conversions for Status / Algorithm / Precision and CLI exit codes.
#include <string>

#include "ps26119/options.h"
#include "ps26119/solution.h"

namespace ps26119 {

const char* to_string(Status s) {
  switch (s) {
    case Status::Optimal: return "Optimal";
    case Status::Infeasible: return "Infeasible";
    case Status::Unbounded: return "Unbounded";
    case Status::IterationLimit: return "IterationLimit";
    case Status::TimeLimit: return "TimeLimit";
    case Status::NumericalError: return "NumericalError";
    case Status::NotSolved: return "NotSolved";
  }
  return "NotSolved";
}

bool status_from_string(const std::string& s, Status& out) {
  for (Status st : {Status::Optimal, Status::Infeasible, Status::Unbounded, Status::IterationLimit,
                    Status::TimeLimit, Status::NumericalError, Status::NotSolved}) {
    if (s == to_string(st)) {
      out = st;
      return true;
    }
  }
  return false;
}

int exit_code(Status s) {
  switch (s) {
    case Status::Optimal: return 0;
    case Status::Infeasible:
    case Status::Unbounded:
    case Status::IterationLimit:
    case Status::TimeLimit: return 1;
    case Status::NumericalError:
    case Status::NotSolved: return 5;
  }
  return 5;
}

const char* to_string(Algorithm a) {
  switch (a) {
    case Algorithm::Auto: return "auto";
    case Algorithm::Oracle: return "oracle";
    case Algorithm::Pdlp: return "pdlp";
    case Algorithm::R2hpdhg: return "r2hpdhg";
    case Algorithm::Simplex: return "simplex";
  }
  return "auto";
}

const char* to_string(Precision p) { return p == Precision::Mixed ? "mixed" : "fp64"; }

bool algorithm_from_string(const std::string& s, Algorithm& out) {
  for (Algorithm a : {Algorithm::Auto, Algorithm::Oracle, Algorithm::Pdlp, Algorithm::R2hpdhg, Algorithm::Simplex}) {
    if (s == to_string(a)) {
      out = a;
      return true;
    }
  }
  return false;
}

bool precision_from_string(const std::string& s, Precision& out) {
  if (s == "fp64") out = Precision::Fp64;
  else if (s == "mixed") out = Precision::Mixed;
  else return false;
  return true;
}

}  // namespace ps26119
