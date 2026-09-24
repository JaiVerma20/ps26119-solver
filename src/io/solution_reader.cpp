// solution_reader.cpp — see solution_reader.h. Format: solution_writer.h.
#include "io/solution_reader.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

namespace ps26119::io {

bool read_solution(const std::string& path, Solution& sol, std::string& error) {
  std::ifstream in(path);
  if (!in) {
    error = "cannot open " + path;
    return false;
  }
  sol = Solution{};
  std::string line;
  if (!std::getline(in, line) || line.rfind("PS26119-SOLUTION", 0) != 0) {
    error = "not a ps26119 solution file";
    return false;
  }
  auto read_block = [&](int count, std::vector<double>& a, std::vector<double>& b) -> bool {
    a.resize(count);
    b.resize(count);
    for (int k = 0; k < count; ++k) {
      if (!std::getline(in, line)) return false;
      std::istringstream ls(line);
      int idx;
      std::string va, vb;
      if (!(ls >> idx >> va >> vb) || idx != k) return false;
      a[k] = std::strtod(va.c_str(), nullptr);  // strtod handles inf/nan spellings
      b[k] = std::strtod(vb.c_str(), nullptr);
    }
    return true;
  };
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    std::string key;
    ls >> key;
    if (key.empty()) continue;
    if (key == "END") return true;
    std::string rest;
    std::getline(ls >> std::ws, rest);
    if (key == "status") {
      if (!status_from_string(rest, sol.status)) {
        error = "unknown status '" + rest + "'";
        return false;
      }
    } else if (key == "objective") {
      sol.objective = std::strtod(rest.c_str(), nullptr);
    } else if (key == "model") {
      sol.model_fingerprint = rest;
    } else if (key == "primal_weight") {
      sol.primal_weight = std::strtod(rest.c_str(), nullptr);
    } else if (key == "engine") {
      sol.engine = rest;
    } else if (key == "COLUMNS" || key == "ROWS") {
      const int count = std::atoi(rest.c_str());
      const bool ok = key == "COLUMNS" ? read_block(count, sol.x, sol.z) : read_block(count, sol.row_activity, sol.y);
      if (!ok) {
        error = "malformed " + key + " block";
        return false;
      }
    }
  }
  error = "missing END";
  return false;
}

}  // namespace ps26119::io
