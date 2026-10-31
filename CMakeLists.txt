cmake_minimum_required(VERSION 3.21)
project(ztensor LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ZTENSOR_BUILD_TESTS "Build unit tests (needs GTest)" ON)
# Off by default: the binary then runs only on CPUs like the build machine.
# Matters on x86, where the baseline is SSE2; arm64's baseline already has NEON.
option(ZTENSOR_CHECK_FINITE "Throw on NaN/Inf after every block (debug)" OFF)
option(ZTENSOR_NATIVE "Compile for the build machine's CPU (-march=native)" OFF)
# Multi-row float Linear layers through Accelerate's SGEMM. Off elsewhere: the
# inline kernels are the portable path.
if(APPLE)
  option(ZTENSOR_ACCELERATE "Use Apple Accelerate SGEMM for multi-row Linear" ON)
else()
  set(ZTENSOR_ACCELERATE OFF)
endif()

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
  src/blas.cpp
)
target_include_directories(ztensor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
target_compile_options(ztensor PRIVATE ${ZTENSOR_WARNINGS})
if(ZTENSOR_CHECK_FINITE)
  target_compile_definitions(ztensor PUBLIC ZTENSOR_CHECK_FINITE)
endif()
if(ZTENSOR_ACCELERATE)
  # PUBLIC: Linear is a template, so the switch must reach the user's code.
  target_compile_definitions(ztensor PUBLIC ZF_USE_ACCELERATE)
  target_link_libraries(ztensor PUBLIC "-framework Accelerate")
endif()
if(ZTENSOR_NATIVE)
  # PUBLIC: the kernels are templates, compiled in the user's translation units.
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-march=native ZTENSOR_HAS_MARCH_NATIVE)
  if(ZTENSOR_HAS_MARCH_NATIVE)
    target_compile_options(ztensor PUBLIC -march=native)
  elseif(MSVC)
    target_compile_options(ztensor PUBLIC /arch:AVX2)
  endif()
endif()

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
    tests/batch_test.cpp
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
