// model_report.h — model statistics and algebraic printout for the CLI (see model_report.cpp).
#pragma once

#include "ps26119/model.h"

namespace ps26119::cli {

void print_statistics(const Model& lp);
void print_model(const Model& lp);

}  // namespace ps26119::cli
