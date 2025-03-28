# NVDLA Attention Module Integration Steps

This document provides the detailed step-by-step process for integrating the attention mechanism into the NVDLA hardware architecture.

## 1. File Placement

Place the attention module files in the NVDLA repository:

```bash
# Create directory for attention module
mkdir -p hw/vmod/nvdla/attn

# Copy SystemVerilog files
cp src/rtl/nvdla_attention.sv hw/vmod/nvdla/attn/
cp src/rtl/nvdla_matrix_mult.sv hw/vmod/nvdla/attn/
cp src/rtl/nvdla_softmax.sv hw/vmod/nvdla/attn/
cp src/rtl/nvdla_attention_bridge.sv hw/vmod/nvdla/attn/

# Copy Verilog integration files
cp src/rtl/NV_NVDLA_attn.v hw/vmod/nvdla/attn/
cp src/rtl/NV_NVDLA_attn_partition.v hw/vmod/nvdla/
```

## 2. Update NVDLA Top-level Integration

Modify the top-level NVDLA module to include the attention module:

### 2.1. Edit `hw/vmod/nvdla/top/NV_NVDLA_partition_o.v`

Add the attention module interfaces to the top-level module:

```verilog
// In port list, add:
// Attention CSB interface
,input         csb2attn_req_pvld
,input  [62:0] csb2attn_req_pd
,output        csb2attn_req_prdy
,output        attn2csb_resp_valid
,output [33:0] attn2csb_resp_pd

// Attention MCIF read interface
,output        attn2mcif_rd_req_valid
,output [78:0] attn2mcif_rd_req_pd
,input         attn2mcif_rd_req_ready
,input         mcif2attn_rd_rsp_valid
,input [513:0] mcif2attn_rd_rsp_pd
,output        mcif2attn_rd_rsp_ready

// Attention MCIF write interface
,output        attn2mcif_wr_req_valid
,output [514:0] attn2mcif_wr_req_pd
,input         attn2mcif_wr_req_ready
,input         mcif2attn_wr_rsp_complete

// Attention interrupt
,output        attn2glb_intr_req
```

Then instantiate the attention module in the same file:

```verilog
// Instantiate attention module
NV_NVDLA_attn_partition u_attn (
   .nvdla_core_clk            (nvdla_core_clk)
  ,.nvdla_core_rstn           (nvdla_core_rstn)
  ,.csb2attn_req_pvld         (csb2attn_req_pvld)
  ,.csb2attn_req_prdy         (csb2attn_req_prdy)
  ,.csb2attn_req_pd           (csb2attn_req_pd)
  ,.attn2csb_resp_valid       (attn2csb_resp_valid)
  ,.attn2csb_resp_pd          (attn2csb_resp_pd)
  ,.attn2mcif_rd_req_valid    (attn2mcif_rd_req_valid)
  ,.attn2mcif_rd_req_ready    (attn2mcif_rd_req_ready)
  ,.attn2mcif_rd_req_pd       (attn2mcif_rd_req_pd)
  ,.mcif2attn_rd_rsp_valid    (mcif2attn_rd_rsp_valid)
  ,.mcif2attn_rd_rsp_ready    (mcif2attn_rd_rsp_ready)
  ,.mcif2attn_rd_rsp_pd       (mcif2attn_rd_rsp_pd)
  ,.attn2mcif_wr_req_valid    (attn2mcif_wr_req_valid)
  ,.attn2mcif_wr_req_ready    (attn2mcif_wr_req_ready)
  ,.attn2mcif_wr_req_pd       (attn2mcif_wr_req_pd)
  ,.mcif2attn_wr_rsp_complete (mcif2attn_wr_rsp_complete)
  ,.attn2glb_intr_req         (attn2glb_intr_req)
);
```

### 2.2. Edit `hw/vmod/nvdla/top/NV_nvdla.v`

Add the attention module interfaces to the top-level NV_nvdla module following the same pattern.

### 2.3. Update MCIF Connections

Modify the MCIF module (`hw/vmod/nvdla/mcif/NV_NVDLA_MCIF.v`) to include the attention client:

```verilog
// Add attention client to MCIF
// In port list, add:
,output         mcif2attn_rd_rsp_valid
,output [513:0] mcif2attn_rd_rsp_pd
,input          mcif2attn_rd_rsp_ready
,input          attn2mcif_rd_req_valid
,input  [78:0]  attn2mcif_rd_req_pd
,output         attn2mcif_rd_req_ready
,input          attn2mcif_wr_req_valid
,input  [514:0] attn2mcif_wr_req_pd
,output         attn2mcif_wr_req_ready
,output         mcif2attn_wr_rsp_complete
```

Then update the MCIF client handling logic to include the attention client.

## 3. Register Space Allocation

Update the NVDLA CSB bus decoder to include the attention module's register space.

### 3.1. Edit `hw/vmod/nvdla/csb/NV_NVDLA_csb.v`

Add the attention module to the CSB block select logic:

```verilog
// In port list, add:
,output        csb2attn_req_pvld
,output [62:0] csb2attn_req_pd 
,input         csb2attn_req_prdy
,input         attn2csb_resp_valid
,input  [33:0] attn2csb_resp_pd
```

Update the address decoding logic to route requests to the attention module:

```verilog
// Define attention module address range
localparam NVDLA_ATTN_BASE_ADDRESS = 16'h7000;
localparam NVDLA_ATTN_END_ADDRESS = 16'h7fff;

// In the address decoder section:
wire is_attn = (req_addr >= NVDLA_ATTN_BASE_ADDRESS) && (req_addr <= NVDLA_ATTN_END_ADDRESS);

// Route requests to attention module
assign csb2attn_req_pvld = req_pvld & is_attn;
assign csb2attn_req_pd = {req_nposted,req_wr,req_wdat,req_addr};
```

Add the attention module to the response mux:

```verilog
// Mux in response from attention module
assign client_resp_valid = sdp2csb_resp_valid | ... | attn2csb_resp_valid;
assign client_resp_pd = ({34{select_sdp}} & sdp2csb_resp_pd) |
                        ... |
                        ({34{select_attn}} & attn2csb_resp_pd);
```

## 4. Interrupt Integration 

### 4.1. Edit `hw/vmod/nvdla/glb/NV_NVDLA_GLB.v`

Add the attention module interrupt:

```verilog
// In port list, add:
,input  attn2glb_intr_req

// Add attention interrupt to GLB_INTR_STATUS register
assign intr_status[XX] = attn2glb_intr_req; // Replace XX with the next available bit
```

## 5. Update Build System

### 5.1. Add attention module to `hw/Makefile`

Add the attention module files to the Makefile's RTL file list:

```makefile
VMOD_VFILES += \
  $(NVDLA_HW)/vmod/nvdla/attn/NV_NVDLA_attn.v \
  $(NVDLA_HW)/vmod/nvdla/attn/NV_NVDLA_attn_partition.v

VMOD_SVFILES += \
  $(NVDLA_HW)/vmod/nvdla/attn/nvdla_attention.sv \
  $(NVDLA_HW)/vmod/nvdla/attn/nvdla_matrix_mult.sv \
  $(NVDLA_HW)/vmod/nvdla/attn/nvdla_softmax.sv \
  $(NVDLA_HW)/vmod/nvdla/attn/nvdla_attention_bridge.sv
```

## 6. Software Integration

### 6.1. Update NVDLA SW Headers

Add attention register definitions to the NVDLA software header files:

```c
// In sw/umd/core/include/nvdla_interface.h
#define NVDLA_ATTN_BASE_ADDRESS     0x7000

// Register offsets
#define NVDLA_ATTN_CONTROL          0x00
#define NVDLA_ATTN_STATUS           0x04
#define NVDLA_ATTN_SEQ_LENGTH       0x08
#define NVDLA_ATTN_HEAD_DIM         0x0C
#define NVDLA_ATTN_NUM_HEADS        0x10
#define NVDLA_ATTN_Q_ADDR           0x14
#define NVDLA_ATTN_K_ADDR           0x18
#define NVDLA_ATTN_V_ADDR           0x1C
#define NVDLA_ATTN_OUT_ADDR         0x20
#define NVDLA_ATTN_PERF_CYCLES      0x24
#define NVDLA_ATTN_PERF_OPS         0x28
```

### 6.2. Add Attention Layer Support to NVDLA Compiler

Update the NVDLA compiler to recognize and map attention operations:

```python
# In sw/compiler/include/priv/EngineAST.h
enum LayerType {
    // ... existing layer types
    kATTENTION,
};

# In sw/compiler/src/compiler/AttentionLayer.cpp
void AttentionLayer::emit() {
    // Configure attention registers
    // Set up memory layout
    // Generate register configurations
}
```

### 6.3. Update Runtime Library

Add attention support to the NVDLA runtime:

```c
// In sw/runtime/src/runtime/attention.c
NvDlaError NvDlaAttentionSubmit(NvDlaDeviceHandle handle, struct NvDlaAttentionParams *params)
{
    // Configure attention registers
    // Submit operation to hardware
    // Wait for completion if needed
    // Return status
}
```

## 7. Verification

### 7.1. Add Attention Module Tests

Create unit tests and integration tests for the attention module:

```
hw/verif/tests/testbench/nvdla_attn_unit_test/
hw/verif/tests/testbench/nvdla_system_attn_test/
```

### 7.2. Update Verification Environment

Update the SystemVerilog testbench environment to include the attention module.

## 8. Documentation

### 8.1. Update Hardware Manual

Add a section on the attention module to the NVDLA Hardware Architecture documentation.

### 8.2. Update Software Manual

Document the attention API in the NVDLA Software Manual.

## 9. Synthesis and Implementation

Update the synthesis and implementation scripts to include the attention module.

## 10. Performance Analysis

Run simulations to measure and document performance metrics:
- Throughput (operations/second)
- Latency
- Energy efficiency
- Area utilization

## Conclusion

After completing these integration steps, the NVDLA hardware will be enhanced with attention mechanism support, enabling efficient execution of transformer-based neural networks.