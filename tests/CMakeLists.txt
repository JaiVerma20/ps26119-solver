include(GoogleTest)

set(PS26119_TEST_SOURCES
  unit/test_model.cpp
)

add_executable(ps26119_tests ${PS26119_TEST_SOURCES})
target_link_libraries(ps26119_tests PRIVATE ps26119_core GTest::gtest_main)
target_include_directories(ps26119_tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(ps26119_tests PRIVATE PS26119_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
gtest_discover_tests(ps26119_tests DISCOVERY_TIMEOUT 60)
