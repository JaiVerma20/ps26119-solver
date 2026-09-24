include(GoogleTest)

set(PS26119_TEST_SOURCES
  unit/test_model.cpp
  unit/test_lpm.cpp
  unit/test_dd.cpp
  unit/test_oracle.cpp
  unit/test_pdhg.cpp
)

if(PS26119_ENABLE_CUDA)
  list(APPEND PS26119_TEST_SOURCES unit/test_gpu.cpp)
endif()

add_executable(ps26119_tests ${PS26119_TEST_SOURCES})
target_link_libraries(ps26119_tests PRIVATE ps26119_core GTest::gtest_main)
target_include_directories(ps26119_tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(ps26119_tests PRIVATE PS26119_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
gtest_discover_tests(ps26119_tests DISCOVERY_TIMEOUT 60)

# Python tooling tests (MPS bridge + verifier) run under ctest when highspy is available.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import highspy, numpy"
                  RESULT_VARIABLE _no_highspy OUTPUT_QUIET ERROR_QUIET)
  if(_no_highspy EQUAL 0)
    add_test(NAME tools.python COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/test_tools.py)
    add_test(NAME bench.generators COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/bench/test_generators.py)
  else()
    message(STATUS "highspy/numpy not found: python tooling tests skipped")
  endif()
endif()
