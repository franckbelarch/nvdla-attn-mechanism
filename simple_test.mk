VERILATOR = verilator
BUILD_DIR = build
LOG_DIR = $(BUILD_DIR)/logs

RTL_FILES = src/rtl/nvdla_attention.sv src/rtl/nvdla_matrix_mult.sv src/rtl/nvdla_softmax.sv
TB_FILES = src/tb/simple_test.sv
CPP_FILES = src/tb/simple_test_main.cpp

VERILATOR_FLAGS = --trace --trace-structs -Wall -Wno-UNUSED -Wno-UNDRIVEN -Wno-CASEINCOMPLETE -Wno-WIDTH -Wno-TIMESCALEMOD -Wno-COMBDLY -Wno-UNOPTFLAT -Wno-MULTITOP -Wno-DECLFILENAME -Wno-REALCVT --MMD --trace-fst -cc --exe
TOP_MODULE = simple_test

all: directories sim

directories:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(LOG_DIR)

sim: directories
	$(VERILATOR) $(VERILATOR_FLAGS) \
		--Mdir obj_dir \
		--top-module $(TOP_MODULE) \
		$(RTL_FILES) $(TB_FILES) $(CPP_FILES)
	$(MAKE) -C obj_dir -f V$(TOP_MODULE).mk
	obj_dir/V$(TOP_MODULE)

lint:
	$(VERILATOR) --lint-only $(RTL_FILES)

clean:
	rm -rf $(BUILD_DIR)
	rm -rf obj_dir

.PHONY: all sim lint clean directories