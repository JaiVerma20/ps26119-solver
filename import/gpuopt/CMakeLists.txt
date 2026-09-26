cmake_minimum_required(VERSION 3.20)
project(gpuopt VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(GPUOPT_BUILD_TESTS "Build the test executables" ON)

# ---------------------------------------------------------------------------
# Core library. No third-party solver code: everything here is our own.
#   core/        problem + result data structures (the M1/M2/M4 contracts)
#   io/          Layer 1 front end: MPS reader
#   linalg/      Layer 2: sparse LU factorization, FTRAN / BTRAN, basis updates
#   simplex/     Layer 3: scaling + bounded revised primal simplex
#   oracle/      dense tableau simplex - the slow, obviously-correct reference
#   validation/  independent optimality certificate checker
# ---------------------------------------------------------------------------
add_library(gpuopt_core STATIC
  src/core/problem.cpp
  src/core/result.cpp
  src/io/mps_reader.cpp
  src/linalg/sparse_lu.cpp
  src/simplex/scaling.cpp
  src/simplex/primal_simplex.cpp
  src/oracle/dense_oracle.cpp
  src/validation/solution_checker.cpp
)
target_include_directories(gpuopt_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(gpuopt_core PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
elseif(MSVC)
  target_compile_options(gpuopt_core PRIVATE /W4)
endif()

# Link the MinGW runtime statically so gpuopt.exe runs outside the toolchain shell.
function(gpuopt_static_runtime target)
  if(MINGW)
    target_link_options(${target} PRIVATE -static)
  endif()
endfunction()

add_executable(gpuopt apps/gpuopt_cli.cpp apps/lu_bench.cpp)
target_link_libraries(gpuopt PRIVATE gpuopt_core)
gpuopt_static_runtime(gpuopt)

# ---------------------------------------------------------------------------
# Tests (plain executables, run through CTest)
# ---------------------------------------------------------------------------
if(GPUOPT_BUILD_TESTS)
  enable_testing()
  foreach(test_name test_mps_reader test_dense_oracle test_oracle_random test_sparse_lu test_simplex)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE gpuopt_core)
    target_compile_definitions(${test_name} PRIVATE
      GPUOPT_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
    gpuopt_static_runtime(${test_name})
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
endif()
