// nvdla_attention_tb.sv - Testbench for the NVDLA attention module

`timescale 1ns/1ps

module nvdla_attention_tb;
    // Clock and reset signals
    logic clk;
    logic rst_n;
    
    // Control signals
    logic attention_enable;
    logic [31:0] seq_length;
    logic [31:0] head_dim;
    logic [31:0] num_heads;
    logic mask_enable;
    
    // Memory interface - Q
    logic q_valid;
    logic [31:0] q_addr;
    logic [127:0] q_data;
    logic q_ready;
    
    // Memory interface - K
    logic k_valid;
    logic [31:0] k_addr;
    logic [127:0] k_data;
    logic k_ready;
    
    // Memory interface - V
    logic v_valid;
    logic [31:0] v_addr;
    logic [127:0] v_data;
    logic v_ready;
    
    // Memory interface - Mask (optional)
    logic mask_valid;
    logic [31:0] mask_addr;
    logic [127:0] mask_data;
    logic mask_ready;
    
    // Output memory interface
    logic out_valid;
    logic [31:0] out_addr;
    logic [127:0] out_data;
    logic out_ready;
    
    // Status outputs
    logic attention_done;
    logic attention_error;
    logic [31:0] attention_status;
    
    // Test data memory
    parameter SEQ_LEN = 16;
    parameter HEAD_DIM = 64;
    parameter NUM_HEADS = 4;
    
    // These would be initialized with test data in a real test
    logic [15:0] q_memory [SEQ_LEN-1:0][HEAD_DIM-1:0];
    logic [15:0] k_memory [SEQ_LEN-1:0][HEAD_DIM-1:0];
    logic [15:0] v_memory [SEQ_LEN-1:0][HEAD_DIM-1:0];
    logic [15:0] out_memory [SEQ_LEN-1:0][HEAD_DIM-1:0];
    
    // Instantiate the attention module
    nvdla_attention dut (
        .clk(clk),
        .rst_n(rst_n),
        .attention_enable(attention_enable),
        .seq_length(seq_length),
        .head_dim(head_dim),
        .num_heads(num_heads),
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
    
    // Clock generation
    always begin
        #5 clk = ~clk;
    end
    
    // Test sequence
    initial begin
        // Initialize signals
        clk = 0;
        rst_n = 0;
        attention_enable = 0;
        seq_length = SEQ_LEN;
        head_dim = HEAD_DIM;
        num_heads = NUM_HEADS;
        mask_enable = 0;
        
        q_valid = 0;
        q_addr = 0;
        q_data = 0;
        
        k_valid = 0;
        k_addr = 0;
        k_data = 0;
        
        v_valid = 0;
        v_addr = 0;
        v_data = 0;
        
        mask_valid = 0;
        mask_addr = 0;
        mask_data = 0;
        
        out_ready = 1;
        
        // Reset sequence
        #20 rst_n = 1;
        
        // Wait for a few clock cycles
        #30;
        
        // Start the attention calculation
        attention_enable = 1;
        
        // Simulate loading Q, K, V matrices
        // In a real test, this would load actual test data
        load_qkv_matrices();
        
        // Wait for attention calculation to complete
        wait(attention_done);
        
        // Check results
        check_results();
        
        // End simulation
        #100 $finish;
    end
    
    // Task to load Q, K, V matrices
    task load_qkv_matrices();
        integer i, j;
        
        // Initialize test data with some patterns
        for (i = 0; i < SEQ_LEN; i++) begin
            for (j = 0; j < HEAD_DIM; j++) begin
                // Initialize with some test patterns
                // In a real test, this would use data from a reference model
                q_memory[i][j] = 16'h0100 + i*HEAD_DIM + j; // Simple pattern
                k_memory[i][j] = 16'h0200 + i*HEAD_DIM + j;
                v_memory[i][j] = 16'h0300 + i*HEAD_DIM + j;
            end
        end
        
        // Simulate memory transfers to the attention module
        // Q matrix
        for (i = 0; i < SEQ_LEN; i++) begin
            q_valid = 1;
            q_addr = i * HEAD_DIM;
            
            // Pack 8 values into the 128-bit data bus
            for (j = 0; j < HEAD_DIM; j += 8) begin
                q_data = {
                    q_memory[i][j+0], q_memory[i][j+1],
                    q_memory[i][j+2], q_memory[i][j+3],
                    q_memory[i][j+4], q_memory[i][j+5],
                    q_memory[i][j+6], q_memory[i][j+7]
                };
                
                // Wait for ready signal
                @(posedge clk);
                while (!q_ready) @(posedge clk);
                
                // Move to next chunk
                q_addr = q_addr + 8;
            end
        end
        q_valid = 0;
        
        // K matrix (similar approach)
        for (i = 0; i < SEQ_LEN; i++) begin
            k_valid = 1;
            k_addr = i * HEAD_DIM;
            
            for (j = 0; j < HEAD_DIM; j += 8) begin
                k_data = {
                    k_memory[i][j+0], k_memory[i][j+1],
                    k_memory[i][j+2], k_memory[i][j+3],
                    k_memory[i][j+4], k_memory[i][j+5],
                    k_memory[i][j+6], k_memory[i][j+7]
                };
                
                @(posedge clk);
                while (!k_ready) @(posedge clk);
                
                k_addr = k_addr + 8;
            end
        end
        k_valid = 0;
        
        // V matrix (similar approach)
        for (i = 0; i < SEQ_LEN; i++) begin
            v_valid = 1;
            v_addr = i * HEAD_DIM;
            
            for (j = 0; j < HEAD_DIM; j += 8) begin
                v_data = {
                    v_memory[i][j+0], v_memory[i][j+1],
                    v_memory[i][j+2], v_memory[i][j+3],
                    v_memory[i][j+4], v_memory[i][j+5],
                    v_memory[i][j+6], v_memory[i][j+7]
                };
                
                @(posedge clk);
                while (!v_ready) @(posedge clk);
                
                v_addr = v_addr + 8;
            end
        end
        v_valid = 0;
    endtask
    
    // Task to check results
    task check_results();
        integer i, j;
        logic [15:0] expected_output [SEQ_LEN-1:0][HEAD_DIM-1:0];
        
        // In a real test, expected_output would be calculated by a reference model
        // Here we're just checking if the DUT produced any output
        
        $display("Checking results...");
        
        // Collect output data
        fork
            begin
                i = 0;
                j = 0;
                
                while (i < SEQ_LEN) begin
                    @(posedge clk);
                    
                    if (out_valid) begin
                        // Unpack data from the 128-bit bus
                        out_memory[i][j+0] = out_data[15:0];
                        out_memory[i][j+1] = out_data[31:16];
                        out_memory[i][j+2] = out_data[47:32];
                        out_memory[i][j+3] = out_data[63:48];
                        out_memory[i][j+4] = out_data[79:64];
                        out_memory[i][j+5] = out_data[95:80];
                        out_memory[i][j+6] = out_data[111:96];
                        out_memory[i][j+7] = out_data[127:112];
                        
                        j = j + 8;
                        if (j >= HEAD_DIM) begin
                            j = 0;
                            i = i + 1;
                        end
                    end
                end
            end
        join_none
        
        // Wait until data collection is complete
        wait(attention_done);
        
        // Compare with expected results
        // In a real test, this would compare with reference model outputs
        
        // Report test results
        if (!attention_error) begin
            $display("Test PASSED!");
        end else begin
            $display("Test FAILED! Error status: %h", attention_status);
        end
    endtask
    
endmodule : nvdla_attention_tb