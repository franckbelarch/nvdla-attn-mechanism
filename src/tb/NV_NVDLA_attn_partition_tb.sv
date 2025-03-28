`timescale 1ns/1ps

// Testbench for NV_NVDLA_attn_partition
module NV_NVDLA_attn_partition_tb;
    // Clock and reset
    logic nvdla_core_clk;
    logic nvdla_core_rstn;
    
    // CSB master interface
    logic csb2attn_req_pvld;
    logic csb2attn_req_prdy;
    logic [62:0] csb2attn_req_pd;
    logic attn2csb_resp_valid;
    logic [33:0] attn2csb_resp_pd;
    
    // Data memory interface - read path
    logic attn2mcif_rd_req_valid;
    logic attn2mcif_rd_req_ready;
    logic [78:0] attn2mcif_rd_req_pd;
    logic mcif2attn_rd_rsp_valid;
    logic mcif2attn_rd_rsp_ready;
    logic [513:0] mcif2attn_rd_rsp_pd;
    
    // Data memory interface - write path
    logic attn2mcif_wr_req_valid;
    logic attn2mcif_wr_req_ready;
    logic [514:0] attn2mcif_wr_req_pd;
    logic mcif2attn_wr_rsp_complete;
    
    // Interrupt
    logic attn2glb_intr_req;
    
    // Memory model
    typedef struct {
        logic [31:0] addr;
        logic [511:0] data;
    } mem_entry_t;
    
    mem_entry_t mem_array[1024];
    
    // Sequential register access counter
    int reg_sequence;
    
    // Test configuration
    parameter SEQ_LEN = 16;
    parameter HEAD_DIM = 64;
    parameter NUM_HEADS = 2;
    parameter MEM_BASE_ADDR = 32'h8000_0000;
    
    // DUT instantiation
    NV_NVDLA_attn_partition DUT (
        .nvdla_core_clk(nvdla_core_clk),
        .nvdla_core_rstn(nvdla_core_rstn),
        
        .csb2attn_req_pvld(csb2attn_req_pvld),
        .csb2attn_req_prdy(csb2attn_req_prdy),
        .csb2attn_req_pd(csb2attn_req_pd),
        .attn2csb_resp_valid(attn2csb_resp_valid),
        .attn2csb_resp_pd(attn2csb_resp_pd),
        
        .attn2mcif_rd_req_valid(attn2mcif_rd_req_valid),
        .attn2mcif_rd_req_ready(attn2mcif_rd_req_ready),
        .attn2mcif_rd_req_pd(attn2mcif_rd_req_pd),
        .mcif2attn_rd_rsp_valid(mcif2attn_rd_rsp_valid),
        .mcif2attn_rd_rsp_ready(mcif2attn_rd_rsp_ready),
        .mcif2attn_rd_rsp_pd(mcif2attn_rd_rsp_pd),
        
        .attn2mcif_wr_req_valid(attn2mcif_wr_req_valid),
        .attn2mcif_wr_req_ready(attn2mcif_wr_req_ready),
        .attn2mcif_wr_req_pd(attn2mcif_wr_req_pd),
        .mcif2attn_wr_rsp_complete(mcif2attn_wr_rsp_complete),
        
        .attn2glb_intr_req(attn2glb_intr_req)
    );
    
    // Clock generation
    always begin
        #5 nvdla_core_clk = ~nvdla_core_clk;
    end
    
    // Memory model functions
    function void mem_write(input logic [31:0] addr, input logic [511:0] data);
        mem_array[addr[13:4]] = '{addr, data};
    endfunction
    
    function logic [511:0] mem_read(input logic [31:0] addr);
        return mem_array[addr[13:4]].data;
    endfunction
    
    // CSB Register access task
    task csb_write(input [16:0] addr, input [31:0] data);
        csb2attn_req_pvld = 1'b1;
        csb2attn_req_pd = {2'b00, 1'b1, data, addr, 5'b0};
        @(posedge nvdla_core_clk);
        while (!csb2attn_req_prdy) @(posedge nvdla_core_clk);
        csb2attn_req_pvld = 1'b0;
        @(posedge nvdla_core_clk);
    endtask
    
    task csb_read(input [16:0] addr, output [31:0] data);
        csb2attn_req_pvld = 1'b1;
        csb2attn_req_pd = {2'b00, 1'b0, 32'h0, addr, 5'b0};
        @(posedge nvdla_core_clk);
        while (!csb2attn_req_prdy) @(posedge nvdla_core_clk);
        csb2attn_req_pvld = 1'b0;
        
        @(posedge nvdla_core_clk);
        while (!attn2csb_resp_valid) @(posedge nvdla_core_clk);
        data = attn2csb_resp_pd[31:0];
        @(posedge nvdla_core_clk);
    endtask
    
    // Initialize test data
    task initialize_test_data();
        // Initialize Q, K, V matrices with test pattern
        // In this example, we're using identity matrices for simplicity
        for (int i = 0; i < SEQ_LEN; i++) begin
            for (int j = 0; j < HEAD_DIM; j += 32) { // 32 16-bit values per 512-bit memory entry
                logic [511:0] q_data = '0;
                logic [511:0] k_data = '0;
                logic [511:0] v_data = '0;
                
                for (int k = 0; k < 32; k++) begin
                    if (j + k < HEAD_DIM) begin
                        if (i == j + k) begin
                            // Identity matrix: 1.0 (0x100 in fixed point) on diagonal
                            q_data[k*16 +: 16] = 16'h0100;
                            k_data[k*16 +: 16] = 16'h0100;
                            v_data[k*16 +: 16] = 16'h0100;
                        end else begin
                            // Zeros elsewhere
                            q_data[k*16 +: 16] = 16'h0000;
                            k_data[k*16 +: 16] = 16'h0000;
                            v_data[k*16 +: 16] = 16'h0000;
                        end
                    end
                end
                
                // Write to memory model
                logic [31:0] q_addr = MEM_BASE_ADDR + (i * HEAD_DIM + j) * 2; // 16-bit values
                logic [31:0] k_addr = MEM_BASE_ADDR + SEQ_LEN * HEAD_DIM * 2 + (i * HEAD_DIM + j) * 2;
                logic [31:0] v_addr = MEM_BASE_ADDR + 2 * SEQ_LEN * HEAD_DIM * 2 + (i * HEAD_DIM + j) * 2;
                
                mem_write(q_addr, q_data);
                mem_write(k_addr, k_data);
                mem_write(v_addr, v_data);
            end
        end
    endtask
    
    // Memory response generation based on DMA requests
    always @(posedge nvdla_core_clk) begin
        if (attn2mcif_rd_req_valid && attn2mcif_rd_req_ready) begin
            logic [31:0] addr = attn2mcif_rd_req_pd[78:32]; // Extract address
            logic [511:0] data = mem_read(addr);
            
            // Simulate memory latency of a few cycles
            repeat (3) @(posedge nvdla_core_clk);
            
            mcif2attn_rd_rsp_valid = 1'b1;
            mcif2attn_rd_rsp_pd = {1'b0, 1'b0, data}; // Format response with data
            
            @(posedge nvdla_core_clk);
            while (!mcif2attn_rd_rsp_ready) @(posedge nvdla_core_clk);
            mcif2attn_rd_rsp_valid = 1'b0;
        end
    end
    
    // Memory write handling
    always @(posedge nvdla_core_clk) begin
        if (attn2mcif_wr_req_valid && attn2mcif_wr_req_ready) begin
            logic [31:0] addr = attn2mcif_wr_req_pd[514:483]; // Extract address
            logic [511:0] data = attn2mcif_wr_req_pd[482:3]; // Extract data
            
            // Write to memory model
            mem_write(addr, data);
            
            // Simulate memory write latency
            repeat (5) @(posedge nvdla_core_clk);
            
            // Signal completion
            mcif2attn_wr_rsp_complete = 1'b1;
            @(posedge nvdla_core_clk);
            mcif2attn_wr_rsp_complete = 1'b0;
        end
    end
    
    // Test sequence
    initial begin
        // Initialize signals
        nvdla_core_clk = 0;
        nvdla_core_rstn = 0;
        csb2attn_req_pvld = 0;
        csb2attn_req_pd = 0;
        attn2mcif_rd_req_ready = 1;
        mcif2attn_rd_rsp_valid = 0;
        mcif2attn_rd_rsp_pd = 0;
        attn2mcif_wr_req_ready = 1;
        mcif2attn_wr_rsp_complete = 0;
        reg_sequence = 0;
        
        // Initialize memory with test data
        initialize_test_data();
        
        // Reset sequence
        repeat (10) @(posedge nvdla_core_clk);
        nvdla_core_rstn = 1;
        repeat (5) @(posedge nvdla_core_clk);
        
        // Test sequence
        $display("Starting attention module test...");
        
        // 1. Configure attention registers
        $display("Configuring attention registers...");
        csb_write(17'h7000, 32'h0000_0000);   // ATTN_CONTROL: off
        csb_write(17'h7008, SEQ_LEN);         // ATTN_SEQ_LENGTH
        csb_write(17'h700C, HEAD_DIM);        // ATTN_HEAD_DIM
        csb_write(17'h7010, NUM_HEADS);       // ATTN_NUM_HEADS
        csb_write(17'h7014, MEM_BASE_ADDR);                      // ATTN_Q_ADDR
        csb_write(17'h7018, MEM_BASE_ADDR + SEQ_LEN*HEAD_DIM*2); // ATTN_K_ADDR
        csb_write(17'h701C, MEM_BASE_ADDR + 2*SEQ_LEN*HEAD_DIM*2); // ATTN_V_ADDR
        csb_write(17'h7020, MEM_BASE_ADDR + 3*SEQ_LEN*HEAD_DIM*2); // ATTN_OUT_ADDR
        
        // 2. Start attention operation
        $display("Starting attention operation...");
        csb_write(17'h7000, 32'h0000_0007);   // ATTN_CONTROL: enable + interrupt enable + mask enable
        
        // 3. Wait for completion (interrupt)
        $display("Waiting for attention calculation to complete...");
        while (!attn2glb_intr_req) @(posedge nvdla_core_clk);
        
        // 4. Read status register
        logic [31:0] status;
        csb_read(17'h7004, status);
        $display("Attention status: %h", status);
        
        // 5. Check results
        $display("Checking results...");
        // In a real test, we would read the output buffer through the memory model
        // and validate the results against a software reference model
        
        // 6. Read performance counters
        logic [31:0] perf_cycles, perf_ops;
        csb_read(17'h7024, perf_cycles);
        csb_read(17'h7028, perf_ops);
        $display("Performance: %d cycles, %d operations", perf_cycles, perf_ops);
        
        // End test
        $display("Test completed!");
        repeat (10) @(posedge nvdla_core_clk);
        $finish;
    end
    
    // Timeout
    initial begin
        #1000000; // 1ms timeout
        $display("ERROR: Test timed out!");
        $finish;
    end
    
endmodule