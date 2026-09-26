// test_framework.hpp - a minimal, dependency-free test harness.
//
//   TEST(name) { EXPECT_NEAR(a, b, 1e-9); ... }
//   TEST_MAIN()
#pragma once

#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace mini_test {

struct Case {
  const char* name;
  void (*fn)();
};

inline std::vector<Case>& registry() {
  static std::vector<Case> cases;
  return cases;
}

inline int& failures() {
  static int count = 0;
  return count;
}

struct Registrar {
  Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

inline int run_all() {
  int failed_cases = 0;
  for (const Case& c : registry()) {
    const int before = failures();
    try {
      c.fn();
    } catch (const std::exception& e) {
      std::printf("  unexpected exception: %s\n", e.what());
      ++failures();
    }
    const bool ok = failures() == before;
    std::printf("[%s] %s\n", ok ? "  OK  " : " FAIL ", c.name);
    if (!ok) ++failed_cases;
  }
  std::printf("\n%zu tests, %d failed\n", registry().size(), failed_cases);
  return failed_cases == 0 ? 0 : 1;
}

}  // namespace mini_test

#define TEST(name)                                                  \
  static void name();                                               \
  static const mini_test::Registrar name##_registrar(#name, name); \
  static void name()

#define EXPECT_TRUE(cond)                                                          \
  do {                                                                             \
    if (!(cond)) {                                                                 \
      std::printf("  %s:%d: EXPECT_TRUE(%s) failed\n", __FILE__, __LINE__, #cond); \
      ++mini_test::failures();                                                     \
    }                                                                              \
  } while (0)

#define EXPECT_EQ(a, b)                                                                \
  do {                                                                                 \
    if (!((a) == (b))) {                                                               \
      std::printf("  %s:%d: EXPECT_EQ(%s, %s) failed\n", __FILE__, __LINE__, #a, #b); \
      ++mini_test::failures();                                                         \
    }                                                                                  \
  } while (0)

#define EXPECT_NEAR(a, b, tol)                                                       \
  do {                                                                               \
    const double va_ = (a), vb_ = (b);                                               \
    if (!(std::fabs(va_ - vb_) <= (tol))) {                                          \
      std::printf("  %s:%d: EXPECT_NEAR(%s, %s) failed: %.12g vs %.12g\n", __FILE__, \
                  __LINE__, #a, #b, va_, vb_);                                       \
      ++mini_test::failures();                                                       \
    }                                                                                \
  } while (0)

#define EXPECT_THROWS(expr)                                                         \
  do {                                                                              \
    bool thrown_ = false;                                                           \
    try {                                                                           \
      (void)(expr);                                                                 \
    } catch (...) {                                                                 \
      thrown_ = true;                                                               \
    }                                                                               \
    if (!thrown_) {                                                                 \
      std::printf("  %s:%d: expected exception from %s\n", __FILE__, __LINE__, #expr); \
      ++mini_test::failures();                                                      \
    }                                                                               \
  } while (0)

#define TEST_MAIN() \
  int main() { return mini_test::run_all(); }
