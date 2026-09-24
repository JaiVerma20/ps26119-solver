// solve.h — the library entry point: dispatch a Model to an engine.
#pragma once

#include "ps26119/model.h"
#include "ps26119/options.h"
#include "ps26119/solution.h"

namespace ps26119 {

// Validates the model, dispatches to the engine chosen in `options`, and stamps the
// model fingerprint, engine and precision into the result. Never throws for a bad
// model: returns NumericalError / NotSolved with `message` set instead.
Solution solve(const Model& model, const Options& options = {});

}  // namespace ps26119
