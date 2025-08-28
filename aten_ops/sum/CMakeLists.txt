cmake_minimum_required(VERSION 3.15)
project(aten_sum_op)

# Find PyTorch installation
find_package(Torch REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add source files
set(SUM_OP_SOURCES
  sum_op.cpp
)

# Create static library for sum operator
add_library(aten_sum_op STATIC ${SUM_OP_SOURCES})

# Set include directories
target_include_directories(aten_sum_op PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../aten/src
  ${CMAKE_CURRENT_SOURCE_DIR}/../../c10
)

# Link against PyTorch libraries
target_link_libraries(aten_sum_op PRIVATE ${TORCH_LIBRARIES})

# Compiler flags
target_compile_definitions(aten_sum_op PRIVATE
  TORCH_ASSERT_ONLY_METHOD_OPERATORS
)

# Test executable
add_executable(test_sum test_sum.cpp)
target_link_libraries(test_sum PRIVATE aten_sum_op ${TORCH_LIBRARIES})
target_include_directories(test_sum PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../aten/src
  ${CMAKE_CURRENT_SOURCE_DIR}/../../c10
)

# Install target
install(TARGETS aten_sum_op
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)