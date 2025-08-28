# Makefile for individual aten operators
# This allows building operators separately: make aten_ops/sum, make aten_ops/mm

SHELL        := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
CMAKE        ?= cmake
BUILD_DIR    := build

.PHONY: all sum mm clean

all: sum mm

# Configure and build sum operator
sum: $(BUILD_DIR)/sum/libaten_sum_op.a

$(BUILD_DIR)/sum/libaten_sum_op.a:
	@echo "Building sum operator..."
	@mkdir -p $(BUILD_DIR)/sum
	@cd $(BUILD_DIR)/sum && $(CMAKE) ../../sum && $(CMAKE) --build .

# Configure and build mm operator  
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