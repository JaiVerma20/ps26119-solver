// mps_reader.h — OWNED BY TEAMMATE. Interface declaration only.
//
// Do not implement, rewrite or "fix" this reader here (CLAUDE.md §2). The teammate
// provides src/io/mps_reader.cpp; CMake compiles it automatically when the file exists
// and defines PS26119_HAVE_MPS_READER. Until then, convert MPS with tools/mps_to_lpm.py
// and load the .lpm file with lpm_reader.h.
//
// Contract: fill `model` according to CLAUDE.md §6 (the Model contract in
// include/ps26119/model.h). Return true on success; on failure return false and put a
// human-readable reason in `error`.
#pragma once

#include <string>

#include "ps26119/model.h"

namespace ps26119::io {

bool read_mps(const std::string& path, Model& model, std::string& error);

}  // namespace ps26119::io
