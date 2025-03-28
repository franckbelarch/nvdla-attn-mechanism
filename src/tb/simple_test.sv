`timescale 1ns/1ps

// Simple testbench for Verilator
module simple_test (
    // Inputs for Verilator
    input wire clk_i  // Clock input
);
    // Internal signals
    logic rst_n;
    
    // Test parameters
    parameter SEQ_LEN = 4;  // Small sequence length for quick simulation
    parameter HEAD_DIM = 4;
    parameter NUM_HEADS = 1;
    
    // Control signals
    logic attention_enable;
    
    // Memory interface signals for Q, K, V
    logic q_valid, k_valid, v_valid;
    logic [31:0] q_addr, k_addr, v_addr;
    logic [127:0] q_data, k_data, v_data;
    logic q_ready, k_ready, v_ready;
    
    // Mask signals
    logic mask_enable;
    logic mask_valid;
    logic [31:0] mask_addr;
    logic [127:0] mask_data;
    logic mask_ready;
    
    // Output signals
    logic out_valid;
    logic [31:0] out_addr;
    logic [127:0] out_data;
    logic out_ready;
    
    // Status signals
    logic attention_done;
    logic attention_error;
    logic [31:0] attention_status;
    
    // Test data
    logic [15:0] q_memory [SEQ_LEN][HEAD_DIM];
    logic [15:0] k_memory [SEQ_LEN][HEAD_DIM];
    logic [15:0] v_memory [SEQ_LEN][HEAD_DIM];
    logic [15:0] out_memory [SEQ_LEN][HEAD_DIM];
    
    // Internal clock (derived from input)
    logic clk;
    assign clk = clk_i;
    
    // Test state machine for Verilator
    enum logic [3:0] {
        RESET,
        INIT,
        LOAD_Q,
        LOAD_K,
        LOAD_V,
        WAIT_DONE,
        COLLECT_RESULTS,
        FINISH,
        IDLE
    } test_state;
    
    // Counters for loading data
    logic [31:0] row_counter;
    logic [31:0] col_counter;
    
    // Instantiate the attention module
    nvdla_attention dut (
        .clk(clk),
        .rst_n(rst_n),
        .attention_enable(attention_enable),
        .seq_length(32'(SEQ_LEN)),
        .head_dim(32'(HEAD_DIM)),
        .num_heads(32'(NUM_HEADS)),
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
    
    // Initialize test data
    initial begin
        // Identity matrices for simple test
        for (int i = 0; i < SEQ_LEN; i++) begin
            for (int j = 0; j < HEAD_DIM; j++) begin
                if (i == j) begin
                    q_memory[i][j] = 16'h0100; // Fixed-point 1.0
                    k_memory[i][j] = 16'h0100; // Fixed-point 1.0
                    v_memory[i][j] = 16'h0100; // Fixed-point 1.0
                end else begin
                    q_memory[i][j] = 16'h0000; // Fixed-point 0.0
                    k_memory[i][j] = 16'h0000; // Fixed-point 0.0
                    v_memory[i][j] = 16'h0000; // Fixed-point 0.0
                end
            end
        end
        
        // Initialize state
        test_state = RESET;
        row_counter = 0;
        col_counter = 0;
        
        // Initialize signals
        rst_n = 0;
        attention_enable = 0;
        mask_enable = 0;
        
        q_valid = 0;
        k_valid = 0;
        v_valid = 0;
        mask_valid = 0;
        out_ready = 1;
        
        $display("Test initialization complete");
    end
    
    // Main test state machine (Verilator-friendly)
    always @(posedge clk) begin
        case (test_state)
            RESET: begin
                // Hold reset for a few cycles
                if (row_counter < 5) begin
                    rst_n <= 0;
                    row_counter <= row_counter + 1;
                end else begin
                    rst_n <= 1;
                    row_counter <= 0;
                    test_state <= INIT;
                    $display("Reset complete, starting test");
                end
            end
            
            INIT: begin
                // Start attention calculation
                attention_enable <= 1;
                test_state <= LOAD_Q;
                $display("Starting attention calculation...");
            end
            
            LOAD_Q: begin
                // Load Q matrix
                q_valid <= 1;
                q_addr <= row_counter * HEAD_DIM;
                
                for (int j = 0; j < HEAD_DIM; j++) begin
                    q_data[j*16 +: 16] <= q_memory[row_counter][j];
                end
                
                if (q_ready) begin
                    if (row_counter >= SEQ_LEN - 1) begin
                        row_counter <= 0;
                        q_valid <= 0;
                        test_state <= LOAD_K;
                        $display("Q matrix loaded, loading K matrix");
                    end else begin
                        row_counter <= row_counter + 1;
                    end
                end
            end
            
            LOAD_K: begin
                // Load K matrix
                k_valid <= 1;
                k_addr <= row_counter * HEAD_DIM;
                
                for (int j = 0; j < HEAD_DIM; j++) begin
                    k_data[j*16 +: 16] <= k_memory[row_counter][j];
                end
                
                if (k_ready) begin
                    if (row_counter >= SEQ_LEN - 1) begin
                        row_counter <= 0;
                        k_valid <= 0;
                        test_state <= LOAD_V;
                        $display("K matrix loaded, loading V matrix");
                    end else begin
                        row_counter <= row_counter + 1;
                    end
                end
            end
            
            LOAD_V: begin
                // Load V matrix
                v_valid <= 1;
                v_addr <= row_counter * HEAD_DIM;
                
                for (int j = 0; j < HEAD_DIM; j++) begin
                    v_data[j*16 +: 16] <= v_memory[row_counter][j];
                end
                
                if (v_ready) begin
                    if (row_counter >= SEQ_LEN - 1) begin
                        row_counter <= 0;
                        v_valid <= 0;
                        test_state <= WAIT_DONE;
                        $display("V matrix loaded, waiting for attention to complete");
                    end else begin
                        row_counter <= row_counter + 1;
                    end
                end
            end
            
            WAIT_DONE: begin
                // Wait for attention calculation to complete
                if (attention_done) begin
                    test_state <= COLLECT_RESULTS;
                    $display("Attention calculation completed!");
                    $display("Status: %s", attention_error ? "ERROR" : "SUCCESS");
                end
            end
            
            COLLECT_RESULTS: begin
                // Collect results from output
                if (out_valid) begin
                    for (int j = 0; j < HEAD_DIM; j++) begin
                        out_memory[row_counter][j] <= out_data[j*16 +: 16];
                    end
                    $display("Output data at addr %0d: %h", out_addr, out_data);
                    
                    if (row_counter >= SEQ_LEN - 1) begin
                        test_state <= FINISH;
                    end else begin
                        row_counter <= row_counter + 1;
                    end
                end
            end
            
            FINISH: begin
                // Display final results
                $display("Test completed!");
                
                if (!attention_error) begin
                    $display("Output matrix:");
                    for (int i = 0; i < SEQ_LEN; i++) begin
                        $write("Row %0d: ", i);
                        for (int j = 0; j < HEAD_DIM; j++) begin
                            $write("%h ", out_memory[i][j]);
                        end
                        $write("\n");
                    end
                end
                
                test_state <= IDLE;
                $finish;
            end
            
            IDLE: begin
                // Do nothing in idle state
            end
        endcase
    end
    
endmodule