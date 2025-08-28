cmake_minimum_required(VERSION 3.15)
project(aten_operators)

# Find PyTorch installation
find_package(Torch REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add subdirectories for individual operators
add_subdirectory(sum)
add_subdirectory(mm)

# Create target to build all operators
add_custom_target(all_operators
  DEPENDS aten_sum_op aten_mm_op
)

# Create individual make targets as custom targets
add_custom_target(sum
  DEPENDS aten_sum_op
  COMMENT "Building sum operator"
)

add_custom_target(mm
  DEPENDS aten_mm_op
  COMMENT "Building mm operator"
)