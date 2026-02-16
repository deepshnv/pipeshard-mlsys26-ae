# No project() here; this file is included from the repo's top-level CMakeLists.txt

# Helper to reduce duplication
function(add_microbench TARGET SRC)
  add_executable(${TARGET} ${SRC})
  target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/ggml/include)
  # Core ggml libs + common (for cpu_get_num_math, etc.)
  target_link_libraries(${TARGET} PRIVATE ggml ggml-base common)
  # Add CUDA backend if the main build enabled it
  if (GGML_CUDA)
    target_link_libraries(${TARGET} PRIVATE ggml-cuda)
    target_compile_definitions(${TARGET} PRIVATE GGML_USE_CUDA=1)
  endif()
  # Nice-to-have defaults
  target_compile_features(${TARGET} PRIVATE cxx_std_17)
  if (MSVC)
    target_compile_options(${TARGET} PRIVATE /W3)
  else()
    target_compile_options(${TARGET} PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
endfunction()

# Roofline micro-benchmark
add_microbench(concurrent_profiler ${CMAKE_CURRENT_SOURCE_DIR}/concurrent_profiler.cpp)
add_microbench(gpu_profiler ${CMAKE_CURRENT_SOURCE_DIR}/gpu_profiler.cpp)
