cmake_minimum_required(VERSION 3.15)
project(aten_mm_op)

# Find PyTorch installation
find_package(Torch REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add source files
set(MM_OP_SOURCES
  mm_op.cpp
)

# Create static library for mm operator
add_library(aten_mm_op STATIC ${MM_OP_SOURCES})

# Set include directories
target_include_directories(aten_mm_op PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../aten/src
  ${CMAKE_CURRENT_SOURCE_DIR}/../../c10
)

# Link against PyTorch libraries
target_link_libraries(aten_mm_op PRIVATE ${TORCH_LIBRARIES})

# Compiler flags
target_compile_definitions(aten_mm_op PRIVATE
  TORCH_ASSERT_ONLY_METHOD_OPERATORS
)

# Test executable
add_executable(test_mm test_mm.cpp)
target_link_libraries(test_mm PRIVATE aten_mm_op ${TORCH_LIBRARIES})
target_include_directories(test_mm PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../aten/src
  ${CMAKE_CURRENT_SOURCE_DIR}/../../c10
)

# Install target
install(TARGETS aten_mm_op
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)