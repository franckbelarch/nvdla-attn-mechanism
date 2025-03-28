# NVDLA Attention Module Integration Checklist

This checklist provides the final steps needed to complete the integration of the attention module into NVDLA.

## Setup

- [ ] Clone the NVDLA repository: `git clone https://github.com/nvdla/hw.git`
- [ ] Set the environment variable: `export NVDLA_HW_DIR=/path/to/nvdla/hw`
- [ ] Run the integration script: `./integrate.sh`

## Required Manual Modifications

After running the integration script, the following files must be manually modified:

### 1. Top-Level Module Modifications

#### 1.1 Edit `$NVDLA_HW_DIR/vmod/nvdla/partition_o/NV_NVDLA_partition_o.v`:

Add attention module interface signals to port list and instantiate attention module:

```verilog
// In port list, add attention interface signals:
// CSB interface
,input         csb2attn_req_pvld
,input  [62:0] csb2attn_req_pd
,output        csb2attn_req_prdy
,output        attn2csb_resp_valid
,output [33:0] attn2csb_resp_pd

// MCIF interface
,output        attn2mcif_rd_req_valid
,output [78:0] attn2mcif_rd_req_pd
,input         attn2mcif_rd_req_ready
,input         mcif2attn_rd_rsp_valid
,input [513:0] mcif2attn_rd_rsp_pd
,output        mcif2attn_rd_rsp_ready
,output        attn2mcif_wr_req_valid
,output [514:0] attn2mcif_wr_req_pd
,input         attn2mcif_wr_req_ready
,input         mcif2attn_wr_rsp_complete

// Interrupt
,output        attn2glb_intr_req

// Inside module, add instantiation:
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

#### 1.2 Edit `$NVDLA_HW_DIR/vmod/nvdla/top/NV_nvdla.v`:

Add attention module interface signals to the top-level NV_nvdla module, following the pattern established for other modules.

### 2. Memory Interface Updates

#### 2.1 Edit `$NVDLA_HW_DIR/vmod/nvdla/mcif/NV_NVDLA_mcif.v`:

Add attention client interface to the MCIF module:

```verilog
// In port list, add attention client:
// Read client interface
,output         mcif2attn_rd_rsp_valid
,output [513:0] mcif2attn_rd_rsp_pd
,input          mcif2attn_rd_rsp_ready
,input          attn2mcif_rd_req_valid
,input  [78:0]  attn2mcif_rd_req_pd
,output         attn2mcif_rd_req_ready

// Write client interface
,input          attn2mcif_wr_req_valid
,input  [514:0] attn2mcif_wr_req_pd
,output         attn2mcif_wr_req_ready
,output         mcif2attn_wr_rsp_complete

// Inside the client arbitration logic, add attention client handling
```

### 3. CSB Interface Updates

#### 3.1 Edit `$NVDLA_HW_DIR/vmod/nvdla/csb/NV_NVDLA_csb.v`:

Add attention module to CSB address decoding:

```verilog
// In port list, add attention CSB interface:
,output        csb2attn_req_pvld
,output [62:0] csb2attn_req_pd 
,input         csb2attn_req_prdy
,input         attn2csb_resp_valid
,input  [33:0] attn2csb_resp_pd

// Define attention address range:
localparam NVDLA_ATTN_BASE_ADDRESS = 16'h7000;
localparam NVDLA_ATTN_END_ADDRESS = 16'h7fff;

// In address decoder, add attention case:
wire is_attn = (req_addr >= NVDLA_ATTN_BASE_ADDRESS) && (req_addr <= NVDLA_ATTN_END_ADDRESS);

// Route requests to attention module:
assign csb2attn_req_pvld = req_pvld & is_attn;
assign csb2attn_req_pd = {req_nposted,req_wr,req_wdat,req_addr};

// Add attention module to response mux:
assign client_resp_valid = sdp2csb_resp_valid | ... | attn2csb_resp_valid;
assign client_resp_pd = ({34{select_sdp}} & sdp2csb_resp_pd) |
                        ... |
                        ({34{select_attn}} & attn2csb_resp_pd);
```

### 4. Interrupt Handling

#### 4.1 Edit `$NVDLA_HW_DIR/vmod/nvdla/glb/NV_NVDLA_glb.v`:

Add attention module interrupt:

```verilog
// In port list, add:
,input  attn2glb_intr_req

// Inside module, add to interrupt status register:
// Find the next available bit in the intr_status register
assign intr_status[X] = attn2glb_intr_req; // Replace X with the next available bit
```

### 5. Build System Updates

#### 5.1 Apply the build system patch:

```bash
cd $NVDLA_HW_DIR
patch -p1 < attn_build_patch.patch
```

## Software Integration

### 1. Update the NVDLA Compiler

Modify the NVDLA compiler to recognize and map attention operations:

- Edit `$NVDLA_HW_DIR/sw/compiler/include/priv/EngineAST.h` to add support for attention layer type
- Create `$NVDLA_HW_DIR/sw/compiler/src/compiler/AttentionLayer.cpp` to implement attention layer compilation

### 2. Update the Runtime

Update the NVDLA runtime to support attention operations:

- Edit `$NVDLA_HW_DIR/sw/umd/core/include/nvdla_interface.h` to add attention register definitions
- Edit `$NVDLA_HW_DIR/sw/umd/core/src/common/EMUInterface.cpp` to add attention register access

## Verification

### 1. Run Tests

```bash
# Run simple test
make -f simple_test.mk

# Run integration test
make -f integration_test.mk

# Run NVDLA verification
cd $NVDLA_HW_DIR && make verif
```

### 2. Validate the Integration

Follow the steps in `validation_guide.md` to verify the integration.

## Documentation

### 1. Update NVDLA Documentation

- Add attention module documentation to NVDLA hardware manual
- Add attention API documentation to NVDLA software manual
- Create examples demonstrating attention usage

## Final Checklist

- [ ] All RTL files properly copied to NVDLA directory
- [ ] Top-level modules updated to include attention module
- [ ] CSB interface updated for attention register access
- [ ] MCIF updated to include attention client
- [ ] Interrupt handling updated in GLB module
- [ ] Build system updated to compile attention files
- [ ] Software stack updated to support attention operations
- [ ] All tests pass
- [ ] Documentation complete

## Need Assistance?

If you encounter issues during integration, please refer to the following resources:

- Documentation in `docs/` directory
- NVDLA integrator's manual
- Create an issue on our GitHub repository

Once all the integration steps are successfully completed, the NVDLA hardware will be enhanced with attention mechanism support.