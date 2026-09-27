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
  // Numbers must parse completely (strtod also accepts inf/nan spellings); counts must be sane —
  // a corrupted file used to reach vector::resize with a negative count via atoi.
  auto number = [](const std::string& t, double& v) {
    char* end = nullptr;
    v = std::strtod(t.c_str(), &end);
    return !t.empty() && end == t.c_str() + t.size();
  };
  auto count_of = [](const std::string& t, int& n) {
    char* end = nullptr;
    const long v = std::strtol(t.c_str(), &end, 10);
    if (t.empty() || *end != '\0' || v < 0 || v > 1'000'000'000) return false;
    n = static_cast<int>(v);
    return true;
  };
  auto read_block = [&](int count, std::vector<double>& a, std::vector<double>& b) -> bool {
    a.resize(count);
    b.resize(count);
    for (int k = 0; k < count; ++k) {
      if (!std::getline(in, line)) return false;
      std::istringstream ls(line);
      int idx;
      std::string va, vb;
      if (!(ls >> idx >> va >> vb) || idx != k || !number(va, a[k]) || !number(vb, b[k])) return false;
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
      if (!number(rest, sol.objective)) {
        error = "malformed objective '" + rest + "'";
        return false;
      }
    } else if (key == "model") {
      sol.model_fingerprint = rest;
    } else if (key == "primal_weight") {
      if (!number(rest, sol.primal_weight)) {
        error = "malformed primal_weight '" + rest + "'";
        return false;
      }
    } else if (key == "engine") {
      sol.engine = rest;
    } else if (key == "DUAL_RAY" || key == "PRIMAL_RAY") {
      std::vector<double>& v = key == "DUAL_RAY" ? sol.dual_ray : sol.primal_ray;
      int count = 0;
      if (!count_of(rest, count)) {
        error = "malformed " + key + " count '" + rest + "'";
        return false;
      }
      v.assign(count, 0.0);
      for (int k = 0; k < count; ++k) {
        std::istringstream rl;
        int idx = -1;
        std::string val;
        if (!std::getline(in, line) || !(rl.str(line), rl >> idx >> val) || idx != k || !number(val, v[k])) {
          error = "malformed " + key + " block";
          return false;
        }
      }
    } else if (key == "COLUMNS" || key == "ROWS") {
      int count = 0;
      if (!count_of(rest, count)) {
        error = "malformed " + key + " count '" + rest + "'";
        return false;
      }
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
