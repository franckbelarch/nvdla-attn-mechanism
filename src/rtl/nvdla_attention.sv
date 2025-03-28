// nvdla_attention.sv - Attention module for NVDLA
// This module implements the attention mechanism for transformer architectures

`timescale 1ns/1ps

module nvdla_attention (
    // Clock and reset
    input  logic        clk,
    input  logic        rst_n,
    
    // Control signals
    input  logic        attention_enable,
    input  logic [31:0] seq_length,
    input  logic [31:0] head_dim,
    input  logic [31:0] num_heads,
    input  logic        mask_enable,
    
    // Memory interface - Q
    input  logic        q_valid,
    input  logic [31:0] q_addr,
    input  logic [127:0] q_data,
    output logic        q_ready,
    
    // Memory interface - K
    input  logic        k_valid,
    input  logic [31:0] k_addr,
    input  logic [127:0] k_data,
    output logic        k_ready,
    
    // Memory interface - V
    input  logic        v_valid,
    input  logic [31:0] v_addr,
    input  logic [127:0] v_data,
    output logic        v_ready,
    
    // Memory interface - Mask (optional)
    input  logic        mask_valid,
    input  logic [31:0] mask_addr,
    input  logic [127:0] mask_data,
    output logic        mask_ready,
    
    // Output memory interface
    output logic        out_valid,
    output logic [31:0] out_addr,
    output logic [127:0] out_data,
    input  logic        out_ready,
    
    // Status outputs
    output logic        attention_done,
    output logic        attention_error,
    output logic [31:0] attention_status
);

    // State machine definitions
    typedef enum logic [3:0] {
        IDLE,
        LOAD_QKV,
        COMPUTE_QK,
        SCALE_QK,
        SOFTMAX_FIND_MAX,
        SOFTMAX_COMPUTE_EXP,
        SOFTMAX_NORMALIZE,
        COMPUTE_ATTN_OUTPUT,
        STORE_OUTPUT,
        ERROR_STATE
    } attention_state_t;
    
    attention_state_t current_state, next_state;
    
    // Configuration registers
    logic [31:0] seq_length_reg;
    logic [31:0] head_dim_reg;
    logic [31:0] num_heads_reg;
    logic        mask_enable_reg;
    
    // Processing counters
    logic [31:0] q_row_counter;
    logic [31:0] k_col_counter;
    logic [31:0] v_col_counter;
    logic [31:0] head_counter;
    
    // Debug counters
    logic [31:0] q_data_loaded_count;
    logic [31:0] k_data_loaded_count;
    logic [31:0] v_data_loaded_count;
    logic [31:0] output_data_stored_count;
    logic [31:0] cycle_counter;
    
    // Buffer management
    logic        qkv_buffers_loaded;
    logic        qk_computation_done;
    logic        softmax_done;
    logic        output_computation_done;
    
    // Error indicators
    logic        size_mismatch_error;
    logic        memory_access_error;
    logic        timeout_error;
    
    // Scale factor constant (1/sqrt(head_dim))
    logic [15:0] scale_factor; // Fixed-point representation
    
    // Main FSM sequential logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
            head_counter <= '0;
        end else begin
            current_state <= next_state;
            
            // Update head counter in sequential logic
            if (current_state == STORE_OUTPUT && out_ready && next_state == LOAD_QKV) begin
                head_counter <= head_counter + 1;
            end else if (current_state == IDLE) begin
                head_counter <= '0;
            end
        end
    end
    
    // FSM next state logic
    always_comb begin
        next_state = current_state; // Default: stay in current state
        
        case (current_state)
            IDLE: begin
                if (attention_enable) begin
                    next_state = LOAD_QKV;
                end
            end
            
            LOAD_QKV: begin
                if (qkv_buffers_loaded) begin
                    next_state = COMPUTE_QK;
                end else if (memory_access_error || timeout_error) begin
                    next_state = ERROR_STATE;
                end
                
                // Set qkv_buffers_loaded flag when all data is loaded is handled in sequential logic
            end
            
            COMPUTE_QK: begin
                if (qk_computation_done) begin
                    next_state = SCALE_QK;
                end
            end
            
            SCALE_QK: begin
                next_state = SOFTMAX_FIND_MAX;
            end
            
            SOFTMAX_FIND_MAX: begin
                next_state = SOFTMAX_COMPUTE_EXP;
            end
            
            SOFTMAX_COMPUTE_EXP: begin
                next_state = SOFTMAX_NORMALIZE;
            end
            
            SOFTMAX_NORMALIZE: begin
                if (softmax_done) begin
                    next_state = COMPUTE_ATTN_OUTPUT;
                end
            end
            
            COMPUTE_ATTN_OUTPUT: begin
                if (output_computation_done) begin
                    next_state = STORE_OUTPUT;
                end
            end
            
            STORE_OUTPUT: begin
                if (out_ready) begin
                    // Check if more processing needed (for multi-head attention)
                    if (head_counter < num_heads_reg - 1) begin
                        // This should be in a sequential block, not combinational
                        next_state = LOAD_QKV;
                    end else begin
                        next_state = IDLE;
                    end
                end
            end
            
            ERROR_STATE: begin
                // Stay in error state until reset
                next_state = ERROR_STATE;
            end
            
            default: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // Configuration registers and counters initialization
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Configuration registers
            seq_length_reg <= '0;
            head_dim_reg <= '0;
            num_heads_reg <= '0;
            mask_enable_reg <= 1'b0;
            
            // Reset counters
            q_row_counter <= '0;
            k_col_counter <= '0;
            v_col_counter <= '0;
            head_counter <= '0;
            
            // Reset debug counters
            q_data_loaded_count <= '0;
            k_data_loaded_count <= '0;
            v_data_loaded_count <= '0;
            output_data_stored_count <= '0;
            cycle_counter <= '0;
            
            // Reset buffer management flags
            qkv_buffers_loaded <= 1'b0;
            qk_computation_done <= 1'b0;
            softmax_done <= 1'b0;
            output_computation_done <= 1'b0;
            
            // Reset error flags
            size_mismatch_error <= 1'b0;
            memory_access_error <= 1'b0;
            timeout_error <= 1'b0;
            
        end else begin
            // Update buffer loading flag
            if (current_state == LOAD_QKV) begin
                if (q_data_loaded_count >= seq_length_reg && 
                    k_data_loaded_count >= seq_length_reg && 
                    v_data_loaded_count >= seq_length_reg) begin
                    qkv_buffers_loaded <= 1'b1;
                end
            end else if (current_state == IDLE) begin
                qkv_buffers_loaded <= 1'b0;
            end
            // Increment cycle counter (for debugging and timeout detection)
            cycle_counter <= cycle_counter + 1'b1;
            
            // Configuration update when starting a new operation
            if (current_state == IDLE && attention_enable) begin
                seq_length_reg <= seq_length;
                head_dim_reg <= head_dim;
                num_heads_reg <= num_heads;
                mask_enable_reg <= mask_enable;
                
                // Reset counters for new operation
                q_data_loaded_count <= '0;
                k_data_loaded_count <= '0;
                v_data_loaded_count <= '0;
                output_data_stored_count <= '0;
                cycle_counter <= '0;
                
                // Calculate scale factor (1/sqrt(head_dim))
                // This would be implemented with a lookup table or approximation in hardware
                scale_factor <= calculate_scale_factor(head_dim);
            end
            
            // Update data loading counters for debugging
            if (q_valid && q_ready) q_data_loaded_count <= q_data_loaded_count + 1'b1;
            if (k_valid && k_ready) k_data_loaded_count <= k_data_loaded_count + 1'b1;
            if (v_valid && v_ready) v_data_loaded_count <= v_data_loaded_count + 1'b1;
            if (out_valid && out_ready) output_data_stored_count <= output_data_stored_count + 1'b1;
            
            // Timeout detection (example: 100000 cycles without completion)
            if (cycle_counter > 32'd100000 && current_state != IDLE && !attention_done) begin
                timeout_error <= 1'b1;
            end
        end
    end
    
    // Function to calculate scale factor (1/sqrt(head_dim))
    // In real hardware, this would be implemented with a lookup table or approximation
    function logic [15:0] calculate_scale_factor(input logic [31:0] dim);
        logic [31:0] sqrt_dim;
        logic [15:0] result;
        
        // Simplified approximation for simulation
        // In real hardware, this would use a more sophisticated approach
        sqrt_dim = 32'(dim) >> 2; // Very crude approximation for demo
        result = 16'h0100 / sqrt_dim[15:0]; // Fixed point division
        
        return result;
    endfunction
    
    // Status outputs
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            attention_done <= 1'b0;
            attention_error <= 1'b0;
            attention_status <= '0;
        end else begin
            // Fix completion detection logic
            if (current_state == STORE_OUTPUT && next_state == IDLE) begin
                attention_done <= 1'b1;
            end else if (current_state == IDLE && attention_enable) begin
                attention_done <= 1'b0;
            end
            
            attention_error <= (current_state == ERROR_STATE);
            
            if (current_state == ERROR_STATE) begin
                if (size_mismatch_error)
                    attention_status <= 32'h0000_0001;
                else if (memory_access_error)
                    attention_status <= 32'h0000_0002;
                else
                    attention_status <= 32'h0000_FFFF; // Unknown error
            end else begin
                attention_status <= {28'h0, current_state};
            end
        end
    end
    
    // Memory interface logic would be implemented here
    // For a complete implementation, you would need:
    // 1. Memory controllers for Q, K, V matrices
    // 2. Matrix multiplication units (potentially reusing NVDLA CMAC)
    // 3. Softmax implementation (with approximation logic)
    // 4. Output buffering and control
    
    // The following blocks are placeholders for the actual implementation
    
    // Matrix multiplication unit for Q×K^T
    // In a real implementation, this would use systolic arrays or other efficient hardware
    // Use parameter for max dimensions to avoid dynamic arrays
    parameter MAX_SEQ_LEN = 256;
    parameter MAX_HEAD_DIM = 128;
    
    logic [15:0] qk_matrix [MAX_SEQ_LEN-1:0][MAX_SEQ_LEN-1:0];
    logic qk_compute_enable;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            qk_compute_enable <= 1'b0;
        end else begin
            qk_compute_enable <= (current_state == COMPUTE_QK);
        end
    end
    
    // Softmax implementation
    // In a real implementation, this would use piece-wise linear approximation or LUTs
    logic [15:0] softmax_output [MAX_SEQ_LEN-1:0][MAX_SEQ_LEN-1:0];
    logic softmax_enable;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            softmax_enable <= 1'b0;
        end else begin
            softmax_enable <= (current_state == SOFTMAX_COMPUTE_EXP) || 
                             (current_state == SOFTMAX_NORMALIZE);
        end
    end
    
    // Final output calculation (softmax×V)
    logic [15:0] attention_output [MAX_SEQ_LEN-1:0][MAX_HEAD_DIM-1:0];
    logic output_compute_enable;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_compute_enable <= 1'b0;
        end else begin
            output_compute_enable <= (current_state == COMPUTE_ATTN_OUTPUT);
        end
    end
    
    // Memory interface control logic
    logic q_data_received;
    logic k_data_received;
    logic v_data_received;
    logic output_data_sent;
    
    // Memory access control with data tracking
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q_data_received <= 1'b0;
            k_data_received <= 1'b0;
            v_data_received <= 1'b0;
            output_data_sent <= 1'b0;
        end else begin
            // Track when data is received
            if (q_valid && q_ready) q_data_received <= 1'b1;
            if (k_valid && k_ready) k_data_received <= 1'b1;
            if (v_valid && v_ready) v_data_received <= 1'b1;
            
            // Reset data received flags when moving to next state
            if (current_state != next_state) begin
                q_data_received <= 1'b0;
                k_data_received <= 1'b0;
                v_data_received <= 1'b0;
                output_data_sent <= 1'b0;
            end
            
            // Track output data sent
            if (out_valid && out_ready) output_data_sent <= 1'b1;
        end
    end
    
    // Memory read/write control logic with handshaking
    assign q_ready = (current_state == LOAD_QKV) && !q_data_received;
    assign k_ready = (current_state == LOAD_QKV) && !k_data_received;
    assign v_ready = (current_state == LOAD_QKV) && !v_data_received;
    assign mask_ready = mask_enable_reg && (current_state == LOAD_QKV);
    
    // Output generation with handshaking
    assign out_valid = (current_state == STORE_OUTPUT) && !output_data_sent;
    
    // Detailed implementation of memory interfaces, computation units, and datapath
    // would be added here in a complete implementation
    
endmodule : nvdla_attention

// Note: These modules are placeholders that would be implemented in separate files
// in a real implementation. They are kept here as documentation only.
/*
// Matrix multiplication unit - would be implemented in nvdla_matrix_mult.sv
module nvdla_matrix_mult_unit (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        enable,
    input  logic [31:0] rows_a,
    input  logic [31:0] cols_a,
    input  logic [31:0] cols_b,
    input  logic [15:0] a_matrix [],
    input  logic [15:0] b_matrix [],
    output logic [15:0] c_matrix [],
    output logic        done
);
    // Implementation would go here
    // In real hardware, this would use systolic arrays or other efficient architectures
endmodule : nvdla_matrix_mult_unit

// Softmax implementation - would be implemented in nvdla_softmax.sv
module nvdla_softmax_unit (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        enable,
    input  logic [31:0] vector_length,
    input  logic [15:0] input_vector [],
    output logic [15:0] output_vector [],
    output logic        done
);
    // Implementation would go here
    // In real hardware, this would use piece-wise linear approximation or LUTs
endmodule : nvdla_softmax_unit
*/