// mps_reader.h — the exception-free MPS reading contract used by the CLI and the C API.
//
// Implementation: the teammate's parser (Shivanshu Vats; io/mps_parser.h, mps_reader.cpp),
// integrated from gpuopt@c192dd0. Cross-checked against an independent reader (highspy,
// tools/verify.py): bit-identical models (Model::fingerprint) on all 93 Netlib LPs and 14
// small MIPLIB 3 models (docs/audit/INITIAL_AUDIT.md).
//
// Contract: fill `model` according to CLAUDE.md §6 (include/ps26119/model.h). Return true
// on success; on failure return false and put a human-readable reason (with the line
// number when there is one) in `error`. Non-fatal remarks go to `warnings` if given.
#pragma once

#include <string>
#include <vector>

#include "ps26119/model.h"

namespace ps26119::io {

bool read_mps(const std::string& path, Model& model, std::string& error,
              std::vector<std::string>* warnings = nullptr);

}  // namespace ps26119::io
