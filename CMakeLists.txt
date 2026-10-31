cmake_minimum_required(VERSION 3.21)
project(ztensor LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ZTENSOR_BUILD_TESTS "Build unit tests (needs GTest)" ON)

set(ZTENSOR_WARNINGS
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

add_library(ztensor
  src/tensor.cpp
  src/add.cpp
  src/mul.cpp
  src/dot.cpp
  src/layers.cpp
  src/attention.cpp
  src/mlp.cpp
  src/transformer.cpp
  src/query_model.cpp
  src/loss.cpp
)
target_include_directories(ztensor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
target_compile_options(ztensor PRIVATE ${ZTENSOR_WARNINGS})

# The README example.
add_executable(find_symbol examples/find_symbol.cpp)
target_link_libraries(find_symbol PRIVATE ztensor)
target_compile_options(find_symbol PRIVATE ${ZTENSOR_WARNINGS})

if(ZTENSOR_BUILD_TESTS)
  enable_testing()
  find_package(GTest CONFIG REQUIRED)

  add_executable(ztensor_tests
    tests/tensor_test.cpp
    tests/layers_test.cpp
    tests/mlp_test.cpp
    tests/transformer_test.cpp
    tests/cross_attention_test.cpp
    tests/softmax_exp_test.cpp
    tests/query_model_test.cpp
    tests/loss_test.cpp
  )
  target_include_directories(ztensor_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_link_libraries(ztensor_tests PRIVATE ztensor GTest::gtest_main)

  include(GoogleTest)
  # Enumerate at ctest time: a slow first launch of a large, template-heavy
  # binary can exceed the build-time discovery timeout.
  gtest_discover_tests(ztensor_tests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 60
  )
endif()

# Microbenchmarks for the hot paths; reports timings, not pass/fail.
add_executable(ztensor_bench benchmarks/tensor_bench.cpp)
target_link_libraries(ztensor_bench PRIVATE ztensor)
