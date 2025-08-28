# Makefile for individual aten operators
# This allows building operators separately: make aten_ops/sum, make aten_ops/mm

SHELL        := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
CMAKE        ?= cmake
BUILD_DIR    := build
CXX          ?= g++
CXXFLAGS     := -std=c++17 -O2

.PHONY: all sum mm clean simple_sum simple_mm test

all: sum mm

# Simple standalone builds (no PyTorch dependencies)
simple_sum:
	@echo "Building simple sum operator..."
	@mkdir -p $(BUILD_DIR)/simple_sum
	@cd $(BUILD_DIR)/simple_sum && $(CMAKE) -f ../../sum/simple_CMakeLists.txt ../../sum && $(CMAKE) --build .

simple_mm:
	@echo "Building simple mm operator..."  
	@mkdir -p $(BUILD_DIR)/simple_mm
	@cd $(BUILD_DIR)/simple_mm && $(CMAKE) -f ../../mm/simple_CMakeLists.txt ../../mm && $(CMAKE) --build .

# Direct compilation (fastest way to test)
test_simple: 
	@echo "Testing simple operators with direct compilation..."
	@cd sum && $(CXX) $(CXXFLAGS) simple_test.cpp -o simple_sum_test && ./simple_sum_test
	@cd mm && $(CXX) $(CXXFLAGS) simple_test.cpp -o simple_mm_test && ./simple_mm_test

# Configure and build sum operator (full PyTorch integration)
sum: $(BUILD_DIR)/sum/libaten_sum_op.a

$(BUILD_DIR)/sum/libaten_sum_op.a:
	@echo "Building sum operator..."
	@mkdir -p $(BUILD_DIR)/sum
	@cd $(BUILD_DIR)/sum && $(CMAKE) ../../sum && $(CMAKE) --build .

# Configure and build mm operator (full PyTorch integration)
mm: $(BUILD_DIR)/mm/libaten_mm_op.a

$(BUILD_DIR)/mm/libaten_mm_op.a:
	@echo "Building mm operator..."
	@mkdir -p $(BUILD_DIR)/mm
	@cd $(BUILD_DIR)/mm && $(CMAKE) ../../mm && $(CMAKE) --build .

# For compatibility with "make aten_ops/sum" syntax
aten_ops/sum: sum
aten_ops/mm: mm

clean:
	@rm -rf $(BUILD_DIR)
	@rm -f sum/simple_sum_test mm/simple_mm_test