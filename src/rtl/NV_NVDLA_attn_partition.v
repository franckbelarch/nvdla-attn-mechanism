// NV_NVDLA_attn_partition.v
// This module implements the NVDLA attention partition

`timescale 1ns/1ps

module NV_NVDLA_attn_partition (
    // Clock and reset
    input                  nvdla_core_clk,   // NVDLA core clock
    input                  nvdla_core_rstn,  // NVDLA core reset (active low)
    
    // CSB master interface
    input                  csb2attn_req_pvld,     // CSB request valid
    output                 csb2attn_req_prdy,     // CSB request ready
    input           [62:0] csb2attn_req_pd,       // CSB request payload
    output                 attn2csb_resp_valid,   // CSB response valid
    output          [33:0] attn2csb_resp_pd,      // CSB response payload
    
    // Data memory interface - read path
    output                 attn2mcif_rd_req_valid,  // Memory read request valid
    input                  attn2mcif_rd_req_ready,  // Memory read request ready
    output          [78:0] attn2mcif_rd_req_pd,     // Memory read request payload
    input                  mcif2attn_rd_rsp_valid,  // Memory read response valid
    output                 mcif2attn_rd_rsp_ready,  // Memory read response ready
    input          [513:0] mcif2attn_rd_rsp_pd,     // Memory read response payload
    
    // Data memory interface - write path
    output                 attn2mcif_wr_req_valid,  // Memory write request valid
    input                  attn2mcif_wr_req_ready,  // Memory write request ready
    output         [514:0] attn2mcif_wr_req_pd,     // Memory write request payload
    input                  mcif2attn_wr_rsp_complete, // Memory write complete
    
    // Interrupt
    output                 attn2glb_intr_req        // Interrupt request
);

    // CSB request decoding
    wire          csb_req_valid;
    wire          csb_req_ready;
    wire   [16:0] csb_req_addr;
    wire   [31:0] csb_req_wdat;
    wire          csb_req_write;
    wire    [1:0] csb_req_nposted;
    
    // CSB request payload decoding
    assign csb_req_valid = csb2attn_req_pvld;
    assign csb2attn_req_prdy = csb_req_ready;
    assign csb_req_addr = csb2attn_req_pd[21:5];
    assign csb_req_wdat = csb2attn_req_pd[53:22];
    assign csb_req_write = csb2attn_req_pd[54];
    assign csb_req_nposted = csb2attn_req_pd[56:55];
    
    // CSB response signals
    wire          csb_resp_valid;
    wire   [31:0] csb_resp_rdat;
    
    // CSB response encoding
    assign attn2csb_resp_valid = csb_resp_valid;
    assign attn2csb_resp_pd = {1'b0, csb_resp_rdat};
    
    // Instantiate the attention module
    nvdla_attention_bridge u_attention_bridge (
        // Clock and reset
        .nvdla_core_clk       (nvdla_core_clk),
        .nvdla_core_rstn      (nvdla_core_rstn),
        
        // CSB interface
        .csb_req_valid        (csb_req_valid),
        .csb_req_ready        (csb_req_ready),
        .csb_req_addr         (csb_req_addr),
        .csb_req_wdat         (csb_req_wdat),
        .csb_req_write        (csb_req_write),
        .csb_req_nposted      (csb_req_nposted),
        .csb_resp_valid       (csb_resp_valid),
        .csb_resp_rdat        (csb_resp_rdat),
        
        // Memory interface - read path
        .dma_rd_req_valid     (attn2mcif_rd_req_valid),
        .dma_rd_req_ready     (attn2mcif_rd_req_ready),
        .dma_rd_req_pd        (attn2mcif_rd_req_pd),
        .dma_rd_rsp_valid     (mcif2attn_rd_rsp_valid),
        .dma_rd_rsp_ready     (mcif2attn_rd_rsp_ready),
        .dma_rd_rsp_pd        (mcif2attn_rd_rsp_pd),
        
        // Memory interface - write path
        .dma_wr_req_valid     (attn2mcif_wr_req_valid),
        .dma_wr_req_ready     (attn2mcif_wr_req_ready),
        .dma_wr_req_pd        (attn2mcif_wr_req_pd),
        .dma_wr_rsp_complete  (mcif2attn_wr_rsp_complete),
        
        // Interrupt
        .intr_req             (attn2glb_intr_req)
    );
    
endmodule // NV_NVDLA_attn_partition