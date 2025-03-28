#!/bin/bash

# test_integration.sh - Script to test integration of NVDLA Attention module
# This script performs basic tests to verify that the integration is working

set -e  # Exit on any error

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if NVDLA_ROOT is provided as an argument, otherwise use default
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_nvdla_root>"
    echo "Example: $0 ~/nvdla/hw"
    exit 1
fi

NVDLA_ROOT="$1"

# Check if NVDLA directory exists
if [ ! -d "$NVDLA_ROOT" ]; then
    echo "Error: NVDLA directory not found at $NVDLA_ROOT"
    exit 1
fi

# Test 1: Verify files were copied
echo "Test 1: Verifying file integration..."

PASS=true

# Check RTL files
if [ ! -f "$NVDLA_ROOT/vmod/nvdla/attn/nvdla_attention.sv" ]; then
    echo "Error: RTL files were not properly integrated"
    PASS=false
fi

# Check software files
if [ ! -f "$NVDLA_ROOT/sw/umd/core/include/nvdla_attn/nvdla_attn_interface.h" ]; then
    echo "Error: Software interface files were not properly integrated"
    PASS=false
fi

# Check compiler files
if [ ! -f "$NVDLA_ROOT/sw/compiler/include/priv/AttentionLayer.h" ]; then
    echo "Error: Compiler files were not properly integrated"
    PASS=false
fi

# Test 2: Verify build 
echo "Test 2: Testing build with attention module..."

cd "$NVDLA_ROOT"
make clean || true  # Clean might fail if it's the first build, that's ok

# Run make for small config
make TOOLS_ONLY=0 PROJECT=nv_small || {
    echo "Error: Build failed with attention module"
    PASS=false
}

# Test 3: Run a simple attention example if simulation tools are available
echo "Test 3: Testing simple attention example..."

if command -v verilator &> /dev/null; then
    cd "$SCRIPT_DIR"
    make -f simple_test.mk || {
        echo "Error: Simple test failed"
        PASS=false
    }
else
    echo "Skipping Test 3: Verilator not found"
fi

# Final results
if [ "$PASS" = true ]; then
    echo "✅ All integration tests passed!"
    echo "The attention module has been successfully integrated with NVDLA."
    echo "Please refer to docs/integration_guide.md for how to use the module."
    exit 0
else
    echo "❌ Integration tests failed!"
    echo "Please check the error messages above and refer to docs/integration_checklist.md"
    echo "for troubleshooting steps."
    exit 1
fi