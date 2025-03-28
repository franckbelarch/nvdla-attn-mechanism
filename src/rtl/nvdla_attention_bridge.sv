`timescale 1ns/1ps

// NVDLA Attention Bridge Module
// This module interfaces between NVDLA's memory system and the attention module

module nvdla_attention_bridge (
    // Clock and reset
    input  logic        nvdla_core_clk,    // NVDLA clock
    input  logic        nvdla_core_rstn,   // NVDLA reset (active low)
    
    // NVDLA CSB interface for register configuration
    input  logic        csb_req_valid,
    output logic        csb_req_ready,
    input  logic [16:0] csb_req_addr,      // Register address
    input  logic [31:0] csb_req_wdat,      // Write data
    input  logic        csb_req_write,     // 1: Write, 0: Read
    input  logic [1:0]  csb_req_nposted,   // Non-posted operation
    output logic        csb_resp_valid,
    output logic [31:0] csb_resp_rdat,     // Read data
    
    // NVDLA data memory interface - read path
    output logic        dma_rd_req_valid,
    input  logic        dma_rd_req_ready,
    output logic [78:0] dma_rd_req_pd,     // Address, size, etc.
    input  logic        dma_rd_rsp_valid,
    output logic        dma_rd_rsp_ready,
    input  logic [513:0] dma_rd_rsp_pd,    // Data and flags
    
    // NVDLA data memory interface - write path
    output logic        dma_wr_req_valid,
    input  logic        dma_wr_req_ready,
    output logic [514:0] dma_wr_req_pd,    // Address, data, size, etc.
    input  logic        dma_wr_rsp_complete,
    
    // Interrupt and status
    output logic        intr_req
);

    //====================================================================
    // Register definition
    //====================================================================
    // Control register (0x00)
    // [0]    - Attention enable
    // [1]    - Mask enable
    // [2]    - Interrupt enable
    // [31:16] - Reserved
    logic [31:0] reg_control;
    
    // Status register (0x04)
    // [0]    - Operation done
    // [1]    - Error flag
    // [7:4]  - State
    // [31:8] - Reserved
    logic [31:0] reg_status;
    
    // Sequence Length register (0x08)
    logic [31:0] reg_seq_length;
    
    // Head dimension register (0x0C)
    logic [31:0] reg_head_dim;
    
    // Number of heads register (0x10)
    logic [31:0] reg_num_heads;
    
    // Q matrix base address register (0x14)
    logic [31:0] reg_q_addr;
    
    // K matrix base address register (0x18)
    logic [31:0] reg_k_addr;
    
    // V matrix base address register (0x1C)
    logic [31:0] reg_v_addr;
    
    // Output base address register (0x20)
    logic [31:0] reg_out_addr;
    
    // Performance counter - cycles (0x24)
    logic [31:0] reg_perf_cycles;
    
    // Performance counter - operations (0x28)
    logic [31:0] reg_perf_ops;
    
    //====================================================================
    // Register interface
    //====================================================================
    // Register access signals
    logic reg_access;
    logic reg_write;
    logic [16:0] reg_addr;
    logic [31:0] reg_wdata;
    logic [31:0] reg_rdata;
    
    // Process register access
    assign reg_access = csb_req_valid;
    assign reg_write = csb_req_write;
    assign reg_addr = csb_req_addr;
    assign reg_wdata = csb_req_wdat;
    
    // Register write logic
    always_ff @(posedge nvdla_core_clk or negedge nvdla_core_rstn) begin
        if (!nvdla_core_rstn) begin
            // Initialize registers to default values
            reg_control <= 32'h0;
            reg_seq_length <= 32'h0;
            reg_head_dim <= 32'h0;
            reg_num_heads <= 32'h0;
            reg_q_addr <= 32'h0;
            reg_k_addr <= 32'h0;
            reg_v_addr <= 32'h0;
            reg_out_addr <= 32'h0;
        end else if (reg_access && reg_write) begin
            // Register write
            case (reg_addr[7:0])
                8'h00: reg_control <= reg_wdata;
                8'h08: reg_seq_length <= reg_wdata;
                8'h0C: reg_head_dim <= reg_wdata;
                8'h10: reg_num_heads <= reg_wdata;
                8'h14: reg_q_addr <= reg_wdata;
                8'h18: reg_k_addr <= reg_wdata;
                8'h1C: reg_v_addr <= reg_wdata;
                8'h20: reg_out_addr <= reg_wdata;
                // Read-only registers cannot be written:
                // 8'h04: reg_status
                // 8'h24: reg_perf_cycles
                // 8'h28: reg_perf_ops
                default: begin
                    // No change for other addresses
                end
            endcase
        end
    end
    
    // Register read logic
    always_comb begin
        // Default read data
        reg_rdata = 32'h0;
        
        if (reg_access && !reg_write) begin
            // Register read
            case (reg_addr[7:0])
                8'h00: reg_rdata = reg_control;
                8'h04: reg_rdata = reg_status;
                8'h08: reg_rdata = reg_seq_length;
                8'h0C: reg_rdata = reg_head_dim;
                8'h10: reg_rdata = reg_num_heads;
                8'h14: reg_rdata = reg_q_addr;
                8'h18: reg_rdata = reg_k_addr;
                8'h1C: reg_rdata = reg_v_addr;
                8'h20: reg_rdata = reg_out_addr;
                8'h24: reg_rdata = reg_perf_cycles;
                8'h28: reg_rdata = reg_perf_ops;
                default: reg_rdata = 32'h0;
            endcase
        end
    end
    
    // CSB response
    assign csb_req_ready = 1'b1; // Always ready to accept requests
    assign csb_resp_valid = reg_access; // Valid response after a request
    assign csb_resp_rdat = reg_rdata;
    
    //====================================================================
    // Attention module interface
    //====================================================================
    // Control signals
    logic attention_enable;
    logic mask_enable;
    
    // Status signals
    logic attention_done;
    logic attention_error;
    logic [31:0] attention_status;
    
    // Memory interface signals for Q, K, V
    logic q_valid, k_valid, v_valid;
    logic [31:0] q_addr, k_addr, v_addr;
    logic [127:0] q_data, k_data, v_data;
    logic q_ready, k_ready, v_ready;
    
    // Mask signals
    logic mask_valid;
    logic [31:0] mask_addr;
    logic [127:0] mask_data;
    logic mask_ready;
    
    // Output signals
    logic out_valid;
    logic [31:0] out_addr;
    logic [127:0] out_data;
    logic out_ready;
    
    // Control signals derived from registers
    assign attention_enable = reg_control[0];
    assign mask_enable = reg_control[1];
    
    // DMA state machine for memory transfers
    typedef enum logic [2:0] {
        DMA_IDLE,
        DMA_READ_Q,
        DMA_READ_K,
        DMA_READ_V,
        DMA_WRITE_OUT
    } dma_state_t;
    
    dma_state_t dma_state, dma_next_state;
    
    // DMA counters and flags
    logic [31:0] dma_addr;
    logic [31:0] dma_size;
    logic [31:0] dma_count;
    logic dma_req_in_progress;
    
    // DMA state machine
    always_ff @(posedge nvdla_core_clk or negedge nvdla_core_rstn) begin
        if (!nvdla_core_rstn) begin
            dma_state <= DMA_IDLE;
            dma_addr <= '0;
            dma_size <= '0;
            dma_count <= '0;
            dma_req_in_progress <= 1'b0;
        end else begin
            dma_state <= dma_next_state;
            
            // Handle DMA request initialization
            if (dma_state != dma_next_state) begin
                case (dma_next_state)
                    DMA_READ_Q: begin
                        dma_addr <= reg_q_addr;
                        dma_size <= reg_seq_length * reg_head_dim * 2; // 16-bit values
                        dma_count <= '0;
                        dma_req_in_progress <= 1'b0;
                    end
                    
                    DMA_READ_K: begin
                        dma_addr <= reg_k_addr;
                        dma_size <= reg_seq_length * reg_head_dim * 2; // 16-bit values
                        dma_count <= '0;
                        dma_req_in_progress <= 1'b0;
                    end
                    
                    DMA_READ_V: begin
                        dma_addr <= reg_v_addr;
                        dma_size <= reg_seq_length * reg_head_dim * 2; // 16-bit values
                        dma_count <= '0;
                        dma_req_in_progress <= 1'b0;
                    end
                    
                    DMA_WRITE_OUT: begin
                        dma_addr <= reg_out_addr;
                        dma_size <= reg_seq_length * reg_head_dim * 2; // 16-bit values
                        dma_count <= '0;
                        dma_req_in_progress <= 1'b0;
                    end
                    
                    default: begin
                        // No change
                    end
                endcase
            end
            
            // Update DMA request progress
            if (dma_rd_req_valid && dma_rd_req_ready) begin
                dma_req_in_progress <= 1'b1;
                dma_addr <= dma_addr + 64; // Assume 64-byte request
                dma_count <= dma_count + 64;
            end
            
            if (dma_wr_req_valid && dma_wr_req_ready) begin
                dma_req_in_progress <= 1'b1;
                dma_addr <= dma_addr + 64; // Assume 64-byte request
                dma_count <= dma_count + 64;
            end
            
            // Clear in-progress flag when response received
            if (dma_rd_rsp_valid && dma_rd_rsp_ready) begin
                dma_req_in_progress <= 1'b0;
            end
            
            if (dma_wr_rsp_complete) begin
                dma_req_in_progress <= 1'b0;
            end
        end
    end
    
    // DMA next state logic
    always_comb begin
        dma_next_state = dma_state;
        
        case (dma_state)
            DMA_IDLE: begin
                if (attention_enable && !(reg_status[0])) begin
                    dma_next_state = DMA_READ_Q;
                end
            end
            
            DMA_READ_Q: begin
                if (dma_count >= dma_size && !dma_req_in_progress) begin
                    dma_next_state = DMA_READ_K;
                end
            end
            
            DMA_READ_K: begin
                if (dma_count >= dma_size && !dma_req_in_progress) begin
                    dma_next_state = DMA_READ_V;
                end
            end
            
            DMA_READ_V: begin
                if (dma_count >= dma_size && !dma_req_in_progress) begin
                    // All data loaded, wait for computation
                end
            end
            
            DMA_WRITE_OUT: begin
                if (dma_count >= dma_size && !dma_req_in_progress) begin
                    dma_next_state = DMA_IDLE;
                end
            end
            
            default: begin
                // Default case to handle other values
                dma_next_state = DMA_IDLE;
            end
        endcase
        
        // Transition to output writing when attention is done
        if (dma_state == DMA_READ_V && attention_done) begin
            dma_next_state = DMA_WRITE_OUT;
        end
    end
    
    // DMA read request generation
    assign dma_rd_req_valid = (dma_state == DMA_READ_Q || dma_state == DMA_READ_K || 
                               dma_state == DMA_READ_V) && !dma_req_in_progress && 
                               (dma_count < dma_size);
                               
    // Simplified DMA read request payload (actual implementation would be more complex)
    assign dma_rd_req_pd = {dma_addr, 8'd64, 3'd0}; // addr, size, cmd
    
    // DMA read response handling
    assign dma_rd_rsp_ready = 1'b1; // Always ready for responses
    
    // DMA write request generation
    assign dma_wr_req_valid = (dma_state == DMA_WRITE_OUT) && !dma_req_in_progress && 
                              (dma_count < dma_size) && out_valid;
    
    // Simplified DMA write request payload (actual implementation would be more complex)
    assign dma_wr_req_pd = {dma_addr, out_data, 8'd64, 3'd0}; // addr, data, size, cmd
    
    // Status register update
    always_ff @(posedge nvdla_core_clk or negedge nvdla_core_rstn) begin
        if (!nvdla_core_rstn) begin
            reg_status <= 32'h0;
            reg_perf_cycles <= 32'h0;
            reg_perf_ops <= 32'h0;
            intr_req <= 1'b0;
        end else begin
            // Done flag
            if (attention_done) begin
                reg_status[0] <= 1'b1;
                // Interrupt generation
                if (reg_control[2]) begin
                    intr_req <= 1'b1;
                end
            end else if (reg_control[0] && dma_state == DMA_IDLE) begin
                reg_status[0] <= 1'b0;
                intr_req <= 1'b0;
            end
            
            // Error flag
            reg_status[1] <= attention_error;
            
            // Current state
            reg_status[7:4] <= attention_status[3:0];
            
            // Update performance counters
            if (attention_enable && !reg_status[0]) begin
                reg_perf_cycles <= reg_perf_cycles + 1;
            end else if (!attention_enable) begin
                reg_perf_cycles <= 32'h0;
            end
            
            // In a real implementation, we would track and count MAC operations
            if (attention_done) begin
                reg_perf_ops <= 32'h0; // Reset after completion
            end
        end
    end
    
    // Instantiate the attention module
    nvdla_attention attention_core (
        .clk(nvdla_core_clk),
        .rst_n(nvdla_core_rstn),
        
        .attention_enable(attention_enable),
        .seq_length(reg_seq_length),
        .head_dim(reg_head_dim),
        .num_heads(reg_num_heads),
        .mask_enable(mask_enable),
        
        .q_valid(q_valid),
        .q_addr(q_addr),
        .q_data(q_data),
        .q_ready(q_ready),
        
        .k_valid(k_valid),
        .k_addr(k_addr),
        .k_data(k_data),
        .k_ready(k_ready),
        
        .v_valid(v_valid),
        .v_addr(v_addr),
        .v_data(v_data),
        .v_ready(v_ready),
        
        .mask_valid(mask_valid),
        .mask_addr(mask_addr),
        .mask_data(mask_data),
        .mask_ready(mask_ready),
        
        .out_valid(out_valid),
        .out_addr(out_addr),
        .out_data(out_data),
        .out_ready(out_ready),
        
        .attention_done(attention_done),
        .attention_error(attention_error),
        .attention_status(attention_status)
    );
    
    // Memory interface conversion between DMA and attention module
    // In a real implementation, this would include proper buffering
    // and data format conversion
    always_comb begin
        // Connect DMA read responses to attention module inputs
        if (dma_rd_rsp_valid) begin
            case (dma_state)
                DMA_READ_Q: begin
                    q_valid = 1'b1;
                    q_data = dma_rd_rsp_pd[127:0];
                    q_addr = dma_count[31:0];
                end
                
                DMA_READ_K: begin
                    k_valid = 1'b1;
                    k_data = dma_rd_rsp_pd[127:0];
                    k_addr = dma_count[31:0];
                end
                
                DMA_READ_V: begin
                    v_valid = 1'b1;
                    v_data = dma_rd_rsp_pd[127:0];
                    v_addr = dma_count[31:0];
                end
                
                default: begin
                    q_valid = 1'b0;
                    k_valid = 1'b0;
                    v_valid = 1'b0;
                end
            endcase
        end else begin
            q_valid = 1'b0;
            k_valid = 1'b0;
            v_valid = 1'b0;
        end
        
        // Connect attention module outputs to DMA write requests
        out_ready = (dma_state == DMA_WRITE_OUT) && dma_wr_req_ready;
    end
    
endmodule : nvdla_attention_bridge