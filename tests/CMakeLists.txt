include(GoogleTest)

set(PS26119_TEST_SOURCES
  unit/test_model.cpp
  unit/test_lpm.cpp
  unit/test_dd.cpp
  unit/test_la.cpp
  unit/test_oracle.cpp
  unit/test_pdhg.cpp
  unit/test_c_api.cpp
  unit/test_batch.cpp
  unit/test_safe_bound.cpp
  unit/test_presolve.cpp
  unit/test_mip.cpp
)

if(PS26119_ENABLE_CUDA)
  list(APPEND PS26119_TEST_SOURCES unit/test_gpu.cpp)
endif()

add_executable(ps26119_tests ${PS26119_TEST_SOURCES})
target_link_libraries(ps26119_tests PRIVATE ps26119_core GTest::gtest_main)
target_include_directories(ps26119_tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(ps26119_tests PRIVATE PS26119_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
gtest_discover_tests(ps26119_tests DISCOVERY_TIMEOUT 60)

# The C API header must compile as plain C.
add_executable(c_api_smoke c_api_smoke.c)
set_target_properties(c_api_smoke PROPERTIES C_STANDARD 99 LINKER_LANGUAGE CXX)
target_link_libraries(c_api_smoke PRIVATE ps26119_core)
add_test(NAME c_api.smoke COMMAND c_api_smoke)

# Python tooling tests (MPS bridge + verifier) run under ctest when highspy is available.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import highspy, numpy"
                  RESULT_VARIABLE _no_highspy OUTPUT_QUIET ERROR_QUIET)
  if(_no_highspy EQUAL 0)
    add_test(NAME tools.python COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/test_tools.py)
    add_test(NAME cli.contract COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_cli.py $<TARGET_FILE:ps26119_cli>)
    if(TARGET ps26119_shared)
      add_test(NAME python.binding COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/test_python.py)
      set_tests_properties(python.binding PROPERTIES ENVIRONMENT "PS26119_LIB=$<TARGET_FILE:ps26119_shared>")
    endif()
    add_test(NAME bench.generators COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/bench/test_generators.py)
  else()
    set(_py_skip_reason "highspy/numpy not importable by ${Python3_EXECUTABLE}")
  endif()
else()
  set(_py_skip_reason "no Python 3 interpreter found")
endif()
if(DEFINED _py_skip_reason)
  # These tests cover the CLI contract, the independent verifier and the Python binding;
  # silently losing them hides regressions (found in the integration audit: a fresh
  # configure picked a Python without highspy). CI sets PS26119_REQUIRE_PYTHON_TESTS=ON.
  if(PS26119_REQUIRE_PYTHON_TESTS)
    message(FATAL_ERROR "Python tests required but unavailable: ${_py_skip_reason}. "
                        "Pass -DPython3_EXECUTABLE=/path/to/python-with-highspy.")
  else()
    message(WARNING "Python tests SKIPPED (${_py_skip_reason}): CLI contract, verifier and "
                    "Python binding are not tested in this build. "
                    "Pass -DPython3_EXECUTABLE=/path/to/python-with-highspy.")
  endif()
endif()
