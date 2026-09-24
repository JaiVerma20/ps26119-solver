// lpm_reader.h — load our `.lpm` text format (spec: tools/lpm.py docstring) into a Model.
//
// .lpm is produced by tools/mps_to_lpm.py (highspy) and by the bench generators. It lets
// the solver run on real instances without depending on the MPS reader (owned by a
// teammate). The optional "# fingerprint <hex>" comment written by the Python side is
// returned so tests can check that both sides agree on the model.
#pragma once

#include <string>

#include "ps26119/model.h"

namespace ps26119::io {

struct LpmReadResult {
  bool ok = false;
  std::string error;                 // set when !ok
  std::string declared_fingerprint;  // from the "# fingerprint" comment, may be empty
};

LpmReadResult read_lpm(const std::string& path, Model& model);
LpmReadResult read_lpm_string(const std::string& text, Model& model);

// Writes a Model as .lpm (used by tests and generators on the C++ side).
bool write_lpm(const std::string& path, const Model& model, std::string& error);

}  // namespace ps26119::io
