// NV_NVDLA_attn.v - Verilog wrapper for NVDLA attention module
// Follows NVDLA naming conventions and interface style

module NV_NVDLA_attn (
    // Clock and reset
    input         nvdla_core_clk,     // NVDLA core clock
    input         nvdla_core_rstn,    // NVDLA core reset (active low)
    
    // CSB interface
    input         csb_req_pvld,      // CSB request valid
    output        csb_req_prdy,      // CSB request ready
    input  [62:0] csb_req_pd,        // CSB request payload
    output        csb_resp_valid,    // CSB response valid
    output [33:0] csb_resp_pd,       // CSB response payload
    
    // CBUF interface
    output        cbuf_rd_en,        // CBUF read enable
    output [14:0] cbuf_rd_addr,      // CBUF read address
    input  [511:0] cbuf_rd_data,     // CBUF read data
    input         cbuf_rd_valid,     // CBUF read valid
    output        cbuf_wr_en,        // CBUF write enable
    output [14:0] cbuf_wr_addr,      // CBUF write address
    output [511:0] cbuf_wr_data,     // CBUF write data
    input         cbuf_wr_ready,     // CBUF write ready
    
    // DBBIF interface
    output        dma_rd_req_vld,    // DMA read request valid
    input         dma_rd_req_rdy,    // DMA read request ready
    output [78:0] dma_rd_req_pd,     // DMA read request payload
    input         dma_rd_rsp_vld,    // DMA read response valid
    output        dma_rd_rsp_rdy,    // DMA read response ready
    input [513:0] dma_rd_rsp_pd,     // DMA read response payload
    
    output        dma_wr_req_vld,    // DMA write request valid
    input         dma_wr_req_rdy,    // DMA write request ready
    output [514:0] dma_wr_req_pd,    // DMA write request payload
    input         dma_wr_rsp_complete, // DMA write complete
    
    // Global control signals
    input         attn_op_en,        // Attention operation enable
    output        attn_op_done,      // Attention operation done
    output [31:0] attn_op_status     // Attention operation status
);

    // CSB Register interface for configuration
    wire          csb_valid;
    wire          csb_ready;
    wire   [15:0] csb_addr;
    wire   [31:0] csb_wdata;
    wire          csb_write;
    wire    [1:0] csb_nposted;
    wire   [31:0] csb_rdata;
    
    // CSB request decoding
    assign csb_valid    = csb_req_pvld;
    assign csb_req_prdy = csb_ready;
    assign csb_addr     = csb_req_pd[21:6];
    assign csb_wdata    = csb_req_pd[53:22];
    assign csb_write    = csb_req_pd[54];
    assign csb_nposted  = csb_req_pd[56:55];
    
    // CSB response encoding
    assign csb_resp_valid = csb_valid && csb_ready;
    assign csb_resp_pd    = {1'b0, csb_rdata};
    
    // Register file for attention configuration
    reg [31:0] reg_attn_mode;       // 0x0000 - Mode configuration
    reg [31:0] reg_attn_seq_len;    // 0x0004 - Sequence length
    reg [31:0] reg_attn_head_dim;   // 0x0008 - Head dimension
    reg [31:0] reg_attn_num_heads;  // 0x000C - Number of heads
    reg [31:0] reg_attn_q_addr;     // 0x0010 - Q matrix base address
    reg [31:0] reg_attn_k_addr;     // 0x0014 - K matrix base address
    reg [31:0] reg_attn_v_addr;     // 0x0018 - V matrix base address
    reg [31:0] reg_attn_out_addr;   // 0x001C - Output base address
    reg [31:0] reg_attn_status;     // 0x0020 - Status register
    
    // CSB Register Read/Write
    always @(posedge nvdla_core_clk or negedge nvdla_core_rstn) begin
        if (!nvdla_core_rstn) begin
            reg_attn_mode      <= 32'h0;
            reg_attn_seq_len   <= 32'h0;
            reg_attn_head_dim  <= 32'h0;
            reg_attn_num_heads <= 32'h0;
            reg_attn_q_addr    <= 32'h0;
            reg_attn_k_addr    <= 32'h0;
            reg_attn_v_addr    <= 32'h0;
            reg_attn_out_addr  <= 32'h0;
        end else begin
            if (csb_valid && csb_ready && csb_write) begin
                case (csb_addr)
                    16'h0000: reg_attn_mode      <= csb_wdata;
                    16'h0004: reg_attn_seq_len   <= csb_wdata;
                    16'h0008: reg_attn_head_dim  <= csb_wdata;
                    16'h000C: reg_attn_num_heads <= csb_wdata;
                    16'h0010: reg_attn_q_addr    <= csb_wdata;
                    16'h0014: reg_attn_k_addr    <= csb_wdata;
                    16'h0018: reg_attn_v_addr    <= csb_wdata;
                    16'h001C: reg_attn_out_addr  <= csb_wdata;
                endcase
            end
        end
    end
    
    // CSB read data
    reg [31:0] csb_rdata_reg;
    always @(posedge nvdla_core_clk or negedge nvdla_core_rstn) begin
        if (!nvdla_core_rstn) begin
            csb_rdata_reg <= 32'h0;
        end else if (csb_valid && !csb_write) begin
            case (csb_addr)
                16'h0000: csb_rdata_reg <= reg_attn_mode;
                16'h0004: csb_rdata_reg <= reg_attn_seq_len;
                16'h0008: csb_rdata_reg <= reg_attn_head_dim;
                16'h000C: csb_rdata_reg <= reg_attn_num_heads;
                16'h0010: csb_rdata_reg <= reg_attn_q_addr;
                16'h0014: csb_rdata_reg <= reg_attn_k_addr;
                16'h0018: csb_rdata_reg <= reg_attn_v_addr;
                16'h001C: csb_rdata_reg <= reg_attn_out_addr;
                16'h0020: csb_rdata_reg <= reg_attn_status;
                default:  csb_rdata_reg <= 32'h0;
            endcase
        end
    end
    assign csb_rdata = csb_rdata_reg;
    assign csb_ready = 1'b1; // Always ready to accept CSB requests
    
    // Control signals for attention module
    wire         attention_enable;
    wire [31:0]  seq_length;
    wire [31:0]  head_dim;
    wire [31:0]  num_heads;
    wire         mask_enable;
    
    // Memory interface signals for Q
    wire         q_valid;
    wire [31:0]  q_addr;
    wire [127:0] q_data;
    wire         q_ready;
    
    // Memory interface signals for K
    wire         k_valid;
    wire [31:0]  k_addr;
    wire [127:0] k_data;
    wire         k_ready;
    
    // Memory interface signals for V
    wire         v_valid;
    wire [31:0]  v_addr;
    wire [127:0] v_data;
    wire         v_ready;
    
    // Memory interface signals for Mask
    wire         mask_valid;
    wire [31:0]  mask_addr;
    wire [127:0] mask_data;
    wire         mask_ready;
    
    // Output memory interface
    wire         out_valid;
    wire [31:0]  out_addr;
    wire [127:0] out_data;
    wire         out_ready;
    
    // Control logic for attention operation
    assign attention_enable = attn_op_en || (reg_attn_mode[0] && reg_attn_mode[1:0] != 2'b00);
    assign seq_length = reg_attn_seq_len;
    assign head_dim = reg_attn_head_dim;
    assign num_heads = reg_attn_num_heads;
    assign mask_enable = reg_attn_mode[1];
    
    // Status signals
    wire        attention_done;
    wire        attention_error;
    wire [31:0] attention_status;
    
    // Connect to output status
    assign attn_op_done = attention_done;
    assign attn_op_status = attention_status;
    
    // Update status register
    always @(posedge nvdla_core_clk or negedge nvdla_core_rstn) begin
        if (!nvdla_core_rstn) begin
            reg_attn_status <= 32'h0;
        end else begin
            reg_attn_status <= attention_status;
        end
    end
    
    // Simplified memory interface adaptation (would be more complex in real implementation)
    // In a real implementation, this would include proper memory controller logic
    
    // Memory conversion from 512-bit CBUF/DMA to 128-bit attention module
    // This is simplified and would need to be expanded for actual implementation
    reg [3:0] data_chunk_sel;
    
    // CBUF read mux to attention module
    always @(posedge nvdla_core_clk or negedge nvdla_core_rstn) begin
        if (!nvdla_core_rstn) begin
            data_chunk_sel <= 4'h0;
        end else if (cbuf_rd_valid) begin
            data_chunk_sel <= data_chunk_sel + 1;
        end
    end
    
    // Simple data routing - would be more complex in actual implementation
    assign q_data = cbuf_rd_data[data_chunk_sel*128 +: 128];
    assign k_data = cbuf_rd_data[data_chunk_sel*128 +: 128];
    assign v_data = cbuf_rd_data[data_chunk_sel*128 +: 128];
    
    // Instantiate the SystemVerilog attention module
    nvdla_attention attention_core (
        .clk               (nvdla_core_clk),
        .rst_n             (nvdla_core_rstn),
        
        .attention_enable  (attention_enable),
        .seq_length        (seq_length),
        .head_dim          (head_dim),
        .num_heads         (num_heads),
        .mask_enable       (mask_enable),
        
        .q_valid           (q_valid),
        .q_addr            (q_addr),
        .q_data            (q_data),
        .q_ready           (q_ready),
        
        .k_valid           (k_valid),
        .k_addr            (k_addr),
        .k_data            (k_data),
        .k_ready           (k_ready),
        
        .v_valid           (v_valid),
        .v_addr            (v_addr),
        .v_data            (v_data),
        .v_ready           (v_ready),
        
        .mask_valid        (mask_valid),
        .mask_addr         (mask_addr),
        .mask_data         (mask_data),
        .mask_ready        (mask_ready),
        
        .out_valid         (out_valid),
        .out_addr          (out_addr),
        .out_data          (out_data),
        .out_ready         (out_ready),
        
        .attention_done    (attention_done),
        .attention_error   (attention_error),
        .attention_status  (attention_status)
    );

endmodule