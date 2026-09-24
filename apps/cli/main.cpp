// main.cpp — the ps26119 command-line interface (a thin layer over the library).
//
//   ps26119 --version
//   ps26119 solve <file.lpm|file.mps> [options]
#include <cstdio>
#include <cstring>
#include <string>

#include "ps26119/solve.h"
#include "ps26119/version.h"

using namespace ps26119;

namespace {

void usage() {
  std::printf(
      "%s %s\n"
      "usage:\n"
      "  %s --version\n"
      "  %s solve <file.lpm|file.mps>\n",
      kProductName, kVersion, kProductName, kProductName);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  const std::string cmd = argv[1];
  if (cmd == "--version" || cmd == "version") {
    std::printf("%s %s\n", kProductName, kVersion);
    return 0;
  }
  if (cmd == "--help" || cmd == "-h") {
    usage();
    return 0;
  }
  usage();
  return 2;
}
