cmake_minimum_required(VERSION 3.20)
project(ps26119 VERSION 0.1.0 LANGUAGES C CXX)

option(PS26119_ENABLE_CUDA "Build the CUDA backend (GPU machines only)" OFF)
option(PS26119_ENABLE_OPENMP "Use OpenMP in CPU kernels" OFF)
option(PS26119_BUILD_TESTS "Build unit tests (GoogleTest via FetchContent)" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(PS26119_ENABLE_CUDA)
  # CMAKE_CUDA_ARCHITECTURES must be set before enable_language(CUDA).
  # "native" (CMake >= 3.24) builds for the GPU in this machine; override with e.g.
  # -DCMAKE_CUDA_ARCHITECTURES="75;86;89" for a portable binary.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
      set(CMAKE_CUDA_ARCHITECTURES native)
    else()
      set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
    endif()
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
endif()

# ---------------------------------------------------------------- core library
set(PS26119_SOURCES
  src/core/model.cpp
  src/core/status.cpp
  src/core/solve.cpp
  src/core/c_api.cpp
  src/io/lpm_reader.cpp
  src/io/solution_writer.cpp
  src/oracle/dense_simplex.cpp
  src/la/csr.cpp
  src/pdhg/scaling.cpp
  src/pdhg/cpu_backend.cpp
  src/pdhg/engine.cpp
  src/pdhg/pdlp.cpp
  src/pdhg/r2hpdhg.cpp
)

# The MPS reader belongs to a teammate (CLAUDE.md §2). Compile it only once it exists.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/io/mps_reader.cpp)
  list(APPEND PS26119_SOURCES src/io/mps_reader.cpp)
  set(PS26119_HAVE_MPS_READER ON)
endif()

if(PS26119_ENABLE_CUDA)
  list(APPEND PS26119_SOURCES src/gpu/cuda_backend.cu)
endif()

add_library(ps26119_core STATIC ${PS26119_SOURCES})
target_include_directories(ps26119_core
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(MSVC)
  target_compile_options(ps26119_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
  target_compile_options(ps26119_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif()
if(PS26119_ENABLE_CUDA)
  target_compile_definitions(ps26119_core PUBLIC PS26119_HAVE_CUDA=1)
  target_link_libraries(ps26119_core PUBLIC CUDA::cudart)
  target_compile_options(ps26119_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)
endif()
if(PS26119_HAVE_MPS_READER)
  target_compile_definitions(ps26119_core PUBLIC PS26119_HAVE_MPS_READER=1)
endif()

if(PS26119_ENABLE_OPENMP)
  if(APPLE)
    # Apple clang has no bundled OpenMP: `brew install libomp`, then point CMake at it.
    execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE _libomp
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_libomp)
      set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${_libomp}/include")
      set(OpenMP_CXX_LIB_NAMES "omp")
      set(OpenMP_omp_LIBRARY "${_libomp}/lib/libomp.dylib")
    endif()
  endif()
  find_package(OpenMP REQUIRED)
  target_link_libraries(ps26119_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(ps26119_core PUBLIC PS26119_HAVE_OPENMP=1)
endif()

# ---------------------------------------------------------------- CLI
add_executable(ps26119_cli apps/cli/main.cpp)
set_target_properties(ps26119_cli PROPERTIES OUTPUT_NAME ps26119)
target_link_libraries(ps26119_cli PRIVATE ps26119_core)
target_include_directories(ps26119_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---------------------------------------------------------------- tests
if(PS26119_BUILD_TESTS)
  enable_testing()
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  add_subdirectory(tests)
endif()
