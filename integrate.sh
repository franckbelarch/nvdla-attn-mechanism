#!/bin/bash

# integrate.sh - Script to integrate NVDLA Attention module into NVDLA project
# This script copies the necessary files to integrate the attention module into NVDLA

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

# Create necessary directories if they don't exist
mkdir -p "$NVDLA_ROOT/vmod/nvdla/attn"
mkdir -p "$NVDLA_ROOT/sw/umd/core/include/nvdla_attn"
mkdir -p "$NVDLA_ROOT/sw/umd/core/src/nvdla_attn"
mkdir -p "$NVDLA_ROOT/sw/compiler/include/priv"
mkdir -p "$NVDLA_ROOT/sw/compiler/src/compiler"

echo "Copying RTL files..."
cp "$SCRIPT_DIR/src/rtl"/* "$NVDLA_ROOT/vmod/nvdla/attn/"

echo "Copying software implementation files..."
cp "$SCRIPT_DIR/include/nvdla_attn_interface.h" "$NVDLA_ROOT/sw/umd/core/include/nvdla_attn/"
cp "$SCRIPT_DIR/src/sw"/* "$NVDLA_ROOT/sw/umd/core/src/nvdla_attn/"
cp "$SCRIPT_DIR/src/utils/reference_attention.py" "$NVDLA_ROOT/sw/compiler/src/compiler/AttentionLayer.cpp"
cp "$SCRIPT_DIR/include/nvdla_attn_interface.h" "$NVDLA_ROOT/sw/compiler/include/priv/AttentionLayer.h"

echo "Patching NVDLA top-level files..."

# Add attention module to NV_nvdla.v
echo "Patching top-level NV_nvdla.v..."
if ! grep -q "attn" "$NVDLA_ROOT/vmod/nvdla/top/NV_nvdla.v"; then
    # Backup original file
    cp "$NVDLA_ROOT/vmod/nvdla/top/NV_nvdla.v" "$NVDLA_ROOT/vmod/nvdla/top/NV_nvdla.v.bak"
    
    # Using sed to insert the attention module inclusion
    sed -i '/module NV_nvdla/a \
    // Attention module instance\
    NV_NVDLA_attn_partition u_attn_partition (\
        .nvdla_core_clk(nvdla_core_clk),\
        .nvdla_core_rstn(nvdla_core_rstn),\
        .csb2attn_req_pd(csb2attn_req_pd),\
        .csb2attn_req_valid(csb2attn_req_valid),\
        .csb2attn_req_ready(csb2attn_req_ready),\
        .attn2csb_resp_pd(attn2csb_resp_pd),\
        .attn2csb_resp_valid(attn2csb_resp_valid),\
        .attn2glb_done_intr_pd(attn2glb_done_intr_pd),\
        .attn2glb_done_intr_valid(attn2glb_done_intr_valid),\
        .attn2mcif_rd_req_pd(attn2mcif_rd_req_pd),\
        .attn2mcif_rd_req_valid(attn2mcif_rd_req_valid),\
        .attn2mcif_rd_req_ready(attn2mcif_rd_req_ready),\
        .mcif2attn_rd_rsp_pd(mcif2attn_rd_rsp_pd),\
        .mcif2attn_rd_rsp_valid(mcif2attn_rd_rsp_valid),\
        .mcif2attn_rd_rsp_ready(mcif2attn_rd_rsp_ready),\
        .attn2mcif_wr_req_pd(attn2mcif_wr_req_pd),\
        .attn2mcif_wr_req_valid(attn2mcif_wr_req_valid),\
        .attn2mcif_wr_req_ready(attn2mcif_wr_req_ready),\
        .mcif2attn_wr_rsp_pd(mcif2attn_wr_rsp_pd),\
        .mcif2attn_wr_rsp_valid(mcif2attn_wr_rsp_valid),\
        .mcif2attn_wr_rsp_ready(mcif2attn_wr_rsp_ready)\
    );' "$NVDLA_ROOT/vmod/nvdla/top/NV_nvdla.v"
    
    echo "Patched NV_nvdla.v successfully"
else
    echo "Attention module already found in NV_nvdla.v, skipping patch"
fi

# Modify MCIF files to add attention interfaces
echo "Patching MCIF interface files..."
if ! grep -q "attn" "$NVDLA_ROOT/vmod/nvdla/mcif/NV_NVDLA_mcif.v"; then
    # Backup original file
    cp "$NVDLA_ROOT/vmod/nvdla/mcif/NV_NVDLA_mcif.v" "$NVDLA_ROOT/vmod/nvdla/mcif/NV_NVDLA_mcif.v.bak"
    
    # Add attention interface to MCIF
    sed -i '/module NV_NVDLA_mcif/a \
    // Attention module interface\
    input         attn2mcif_rd_req_valid;\
    output        attn2mcif_rd_req_ready;\
    input  [54:0] attn2mcif_rd_req_pd;\
    output        mcif2attn_rd_rsp_valid;\
    input         mcif2attn_rd_rsp_ready;\
    output [513:0] mcif2attn_rd_rsp_pd;\
    input         attn2mcif_wr_req_valid;\
    output        attn2mcif_wr_req_ready;\
    input  [514:0] attn2mcif_wr_req_pd;\
    output        mcif2attn_wr_rsp_valid;\
    input         mcif2attn_wr_rsp_ready;\
    output [1:0]  mcif2attn_wr_rsp_pd;' "$NVDLA_ROOT/vmod/nvdla/mcif/NV_NVDLA_mcif.v"
    
    echo "Patched NV_NVDLA_mcif.v successfully"
else
    echo "Attention interface already found in NV_NVDLA_mcif.v, skipping patch"
fi

# Add attention to CSB interface
echo "Patching CSB interface files..."
if ! grep -q "attn" "$NVDLA_ROOT/vmod/nvdla/csb/NV_NVDLA_csb.v"; then
    # Backup original file
    cp "$NVDLA_ROOT/vmod/nvdla/csb/NV_NVDLA_csb.v" "$NVDLA_ROOT/vmod/nvdla/csb/NV_NVDLA_csb.v.bak"
    
    # Add attention interface to CSB
    sed -i '/module NV_NVDLA_csb/a \
    // Attention module interface\
    output        csb2attn_req_valid;\
    input         csb2attn_req_ready;\
    output [62:0] csb2attn_req_pd;\
    input         attn2csb_resp_valid;\
    output        attn2csb_resp_ready;\
    input  [33:0] attn2csb_resp_pd;' "$NVDLA_ROOT/vmod/nvdla/csb/NV_NVDLA_csb.v"
    
    echo "Patched NV_NVDLA_csb.v successfully"
else
    echo "Attention interface already found in NV_NVDLA_csb.v, skipping patch"
fi

# Add attention to GLB interface
echo "Patching GLB interface files..."
if ! grep -q "attn" "$NVDLA_ROOT/vmod/nvdla/glb/NV_NVDLA_glb.v"; then
    # Backup original file
    cp "$NVDLA_ROOT/vmod/nvdla/glb/NV_NVDLA_glb.v" "$NVDLA_ROOT/vmod/nvdla/glb/NV_NVDLA_glb.v.bak"
    
    # Add attention interface to GLB
    sed -i '/module NV_NVDLA_glb/a \
    // Attention module interface\
    input         attn2glb_done_intr_valid;\
    input  [1:0]  attn2glb_done_intr_pd;' "$NVDLA_ROOT/vmod/nvdla/glb/NV_NVDLA_glb.v"
    
    echo "Patched NV_NVDLA_glb.v successfully"
else
    echo "Attention interface already found in NV_NVDLA_glb.v, skipping patch"
fi

echo "Integration complete!"
echo "Please refer to docs/integration_guide.md for further instructions on how to use the attention module."