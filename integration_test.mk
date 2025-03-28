# Integration Test Makefile for NVDLA Attention Module

# Directories
BUILD_DIR = build
LOG_DIR = $(BUILD_DIR)/logs

# SystemVerilog and Verilog files for integration test
RTL_FILES = src/rtl/NV_NVDLA_attn_partition.v \
            src/rtl/nvdla_attention_bridge.sv \
            src/rtl/nvdla_attention.sv \
            src/rtl/nvdla_matrix_mult.sv \
            src/rtl/nvdla_softmax.sv

TB_FILES = src/tb/NV_NVDLA_attn_partition_tb.sv

# Simulator settings (Verilator by default)
SIM ?= verilator

# Verilator configuration
VERILATOR = verilator
VERILATOR_FLAGS = --trace --trace-structs -Wall \
                 -Wno-UNUSED -Wno-UNDRIVEN -Wno-UNOPTFLAT -Wno-WIDTH \
                 -Wno-CASEINCOMPLETE -Wno-DECLFILENAME -Wno-PINMISSING \
                 -Wno-PINCONNECTEMPTY -Wno-SYNCASYNCNET --assert \
                 --trace-fst -cc

# VCS configuration (if available)
VCS = vcs
VCS_FLAGS = -full64 -sverilog -debug_access+all -timescale=1ns/1ps \
           +define+ASSERT_ON -assert svaext -notice

# Default target
all: directories sim

# Create build directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(LOG_DIR)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -rf obj_dir

# Simulation target
sim: directories
ifeq ($(SIM), verilator)
	@echo "Running integration simulation with Verilator..."
	$(VERILATOR) $(VERILATOR_FLAGS) \
		--Mdir obj_dir \
		--top-module NV_NVDLA_attn_partition_tb \
		$(RTL_FILES) $(TB_FILES)
	$(MAKE) -C obj_dir -f VNV_NVDLA_attn_partition_tb.mk
	obj_dir/VNV_NVDLA_attn_partition_tb
else ifeq ($(SIM), vcs)
	@echo "Running integration simulation with VCS..."
	$(VCS) $(VCS_FLAGS) \
		-o $(BUILD_DIR)/attn_integration_tb \
		$(RTL_FILES) $(TB_FILES)
	$(BUILD_DIR)/attn_integration_tb
else
	@echo "Unsupported simulator: $(SIM). Use 'verilator' or 'vcs'."
	@exit 1
endif

# Lint target
lint:
	$(VERILATOR) --lint-only -Wall \
	-Wno-UNUSED -Wno-UNDRIVEN -Wno-UNOPTFLAT -Wno-WIDTH \
	-Wno-CASEINCOMPLETE -Wno-DECLFILENAME -Wno-PINMISSING \
	-Wno-PINCONNECTEMPTY \
	--top-module NV_NVDLA_attn_partition \
	$(RTL_FILES)

# Wave viewer (GTKWave)
waves:
	gtkwave $(LOG_DIR)/attn_integration_tb.fst &

.PHONY: all directories clean sim lint waves