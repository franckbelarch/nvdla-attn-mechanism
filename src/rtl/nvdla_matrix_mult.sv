// nvdla_matrix_mult.sv - Matrix multiplication unit for NVDLA attention mechanism
// Performs A × B matrix multiplication with configurable dimensions

`timescale 1ns/1ps

module nvdla_matrix_mult (
    // Clock and reset
    input  logic        clk,
    input  logic        rst_n,
    
    // Control signals
    input  logic        enable,
    input  logic [31:0] rows_a,
    input  logic [31:0] cols_a,
    input  logic [31:0] cols_b,
    input  logic [15:0] scale_factor, // Optional scaling factor for QK^T operation
    input  logic        apply_scale,  // Whether to apply scaling
    
    // Memory interface - A matrix
    input  logic        a_valid,
    input  logic [31:0] a_addr,
    input  logic [127:0] a_data,  // 8 x 16-bit values
    output logic        a_ready,
    
    // Memory interface - B matrix
    input  logic        b_valid,
    input  logic [31:0] b_addr,
    input  logic [127:0] b_data,  // 8 x 16-bit values
    output logic        b_ready,
    
    // Output memory interface
    output logic        c_valid,
    output logic [31:0] c_addr,
    output logic [127:0] c_data,  // 8 x 16-bit values
    input  logic        c_ready,
    
    // Status signals
    output logic        mult_done,
    output logic        mult_error
);

    // Parameters for processing
    parameter MAX_DIM = 256;  // Maximum matrix dimension
    parameter FRAC_BITS = 8;  // Number of fractional bits in fixed-point representation
    
    // Internal state machine
    typedef enum logic [2:0] {
        IDLE,
        LOAD_MATRICES,
        COMPUTE_MULT,
        STORE_RESULT,
        DONE,
        ERROR
    } matmult_state_t;
    
    matmult_state_t current_state, next_state;
    
    // Internal registers
    logic [31:0] a_row_counter;
    logic [31:0] a_col_counter;
    logic [31:0] b_col_counter;
    logic [31:0] compute_row;
    logic [31:0] compute_col;
    logic [31:0] compute_k;
    
    // Debug and performance counters
    logic [31:0] cycle_counter;
    logic [31:0] mac_operations_counter; // Count number of multiply-accumulate operations
    logic [31:0] data_transfers_counter; // Count number of data transfers
    
    // Timeout detection
    logic timeout_detected;
    parameter TIMEOUT_CYCLES = 100000; // Adjust as needed for real implementation
    
    // Buffers to hold matrix data
    logic [15:0] a_buffer [0:MAX_DIM-1][0:MAX_DIM-1];
    logic [15:0] b_buffer [0:MAX_DIM-1][0:MAX_DIM-1];
    logic [15:0] c_buffer [0:MAX_DIM-1][0:MAX_DIM-1];
    
    // Loading status
    logic a_loaded;
    logic b_loaded;
    
    // Data validation flags
    logic a_data_valid;
    logic b_data_valid;
    logic result_valid;
    
    // FSM sequential logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
        end else begin
            current_state <= next_state;
        end
    end
    
    // FSM next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            IDLE: begin
                if (enable) begin
                    next_state = LOAD_MATRICES;
                end
            end
            
            LOAD_MATRICES: begin
                if (a_loaded && b_loaded) begin
                    next_state = COMPUTE_MULT;
                end
            end
            
            COMPUTE_MULT: begin
                // After computing all elements of C matrix
                if (compute_row >= rows_a) begin
                    next_state = STORE_RESULT;
                end
            end
            
            STORE_RESULT: begin
                // After storing all results
                if (a_row_counter >= rows_a) begin
                    next_state = DONE;
                end
            end
            
            DONE: begin
                next_state = IDLE;
            end
            
            ERROR: begin
                // Stay in error state until reset
                next_state = ERROR;
            end
            
            default: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // Matrix loading logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset counters
            a_row_counter <= 0;
            a_col_counter <= 0;
            b_col_counter <= 0;
            compute_row <= 0;
            compute_col <= 0;
            compute_k <= 0;
            
            // Reset performance counters
            cycle_counter <= 0;
            mac_operations_counter <= 0;
            data_transfers_counter <= 0;
            
            // Reset status flags
            a_loaded <= 0;
            b_loaded <= 0;
            a_data_valid <= 0;
            b_data_valid <= 0;
            result_valid <= 0;
            timeout_detected <= 0;
            
            // Reset interface signals
            a_ready <= 0;
            b_ready <= 0;
            c_valid <= 0;
            mult_done <= 0;
            mult_error <= 0;
        end else begin
            // Increment cycle counter for timeout detection
            cycle_counter <= cycle_counter + 1;
            
            // Timeout detection
            if (cycle_counter > TIMEOUT_CYCLES && current_state != IDLE) begin
                timeout_detected <= 1'b1;
                mult_error <= 1'b1;
            end
            case (current_state)
                IDLE: begin
                    a_row_counter <= 0;
                    a_col_counter <= 0;
                    b_col_counter <= 0;
                    compute_row <= 0;
                    compute_col <= 0;
                    compute_k <= 0;
                    a_loaded <= 0;
                    b_loaded <= 0;
                    a_ready <= enable;
                    b_ready <= enable;
                    c_valid <= 0;
                    mult_done <= 0;
                    mult_error <= 0;
                end
                
                LOAD_MATRICES: begin
                    // Ready to receive A and B matrices
                    a_ready <= !a_loaded;
                    b_ready <= !b_loaded;
                    
                    // Load A matrix
                    if (a_valid && !a_loaded) begin
                        // Process 8 elements at a time from the 128-bit bus
                        for (int i = 0; i < 8; i++) begin
                            if (a_col_counter + i < cols_a) begin
                                a_buffer[a_row_counter][a_col_counter + i] = a_data[i*16 +: 16];
                            end
                        end
                        
                        // Update counters
                        if (a_col_counter + 8 >= cols_a) begin
                            a_col_counter <= 0;
                            if (a_row_counter + 1 >= rows_a) begin
                                a_loaded <= 1;
                            end else begin
                                a_row_counter <= a_row_counter + 1;
                            end
                        end else begin
                            a_col_counter <= a_col_counter + 8;
                        end
                    end
                    
                    // Load B matrix (transposed for easier multiplication)
                    if (b_valid && !b_loaded) begin
                        // Process 8 elements at a time from the 128-bit bus
                        // Note: B is assumed to be in row-major format but loaded in a transposed manner
                        for (int i = 0; i < 8; i++) begin
                            if (b_col_counter + i < cols_b) begin
                                // For B transpose: b_buffer[col][row] = b_matrix[row][col]
                                b_buffer[b_col_counter + i][a_col_counter] = b_data[i*16 +: 16];
                            end
                        end
                        
                        // Update counters for B
                        if (b_col_counter + 8 >= cols_b) begin
                            b_col_counter <= 0;
                            if (a_col_counter + 1 >= cols_a) begin
                                b_loaded <= 1;
                            end
                        end else begin
                            b_col_counter <= b_col_counter + 8;
                        end
                    end
                end
                
                COMPUTE_MULT: begin
                    // Compute matrix multiplication C = A×B
                    // In a real implementation, this would be pipelined and use hardware MAC units
                    
                    // Reset cycle counter for new computation
                    if (compute_row == 0 && compute_col == 0 && compute_k == 0) begin
                        cycle_counter <= 0;
                        mac_operations_counter <= 0;
                    end
                    
                    // Initialize accumulator for current element
                    if (compute_k == 0) begin
                        c_buffer[compute_row][compute_col] = 0;
                    end
                    
                    // Accumulate dot product
                    c_buffer[compute_row][compute_col] = c_buffer[compute_row][compute_col] + 
                        multiply_fixed_point(a_buffer[compute_row][compute_k], b_buffer[compute_col][compute_k]);
                    
                    // Increment MAC operations counter for performance tracking
                    mac_operations_counter <= mac_operations_counter + 1;
                    
                    // Update compute counters
                    if (compute_k + 1 >= cols_a) begin
                        // Apply scaling if needed (for attention QK^T operation)
                        if (apply_scale && compute_k + 1 == cols_a) begin
                            c_buffer[compute_row][compute_col] = 
                                multiply_fixed_point(c_buffer[compute_row][compute_col], scale_factor);
                        end
                        
                        compute_k <= 0;
                        if (compute_col + 1 >= cols_b) begin
                            compute_col <= 0;
                            compute_row <= compute_row + 1;
                            
                            // Mark computation as done when we've processed all rows
                            if (compute_row + 1 >= rows_a) begin
                                result_valid <= 1'b1;
                            end
                        end else begin
                            compute_col <= compute_col + 1;
                        end
                    end else begin
                        compute_k <= compute_k + 1;
                    end
                end
                
                STORE_RESULT: begin
                    // Output computed results
                    c_valid <= 1;
                    
                    if (c_ready) begin
                        // Pack 8 values into the 128-bit data bus
                        for (int i = 0; i < 8; i++) begin
                            if (a_col_counter + i < cols_b) begin
                                c_data[i*16 +: 16] = c_buffer[a_row_counter][a_col_counter + i];
                            end else begin
                                c_data[i*16 +: 16] = 0;
                            end
                        end
                        
                        c_addr <= (a_row_counter * cols_b) + a_col_counter;
                        
                        // Update counters
                        if (a_col_counter + 8 >= cols_b) begin
                            a_col_counter <= 0;
                            a_row_counter <= a_row_counter + 1;
                        end else begin
                            a_col_counter <= a_col_counter + 8;
                        end
                    end
                end
                
                DONE: begin
                    c_valid <= 0;
                    mult_done <= 1;
                end
                
                ERROR: begin
                    mult_error <= 1;
                end
                
                default: begin
                    // Default case to avoid incomplete case warning
                    // No action needed
                end
            endcase
        end
    end
    
    // Helper function for fixed-point multiplication
    function logic [15:0] multiply_fixed_point(input logic [15:0] a, input logic [15:0] b);
        logic [31:0] result;
        
        // Perform multiplication (a × b)
        result = a * b;
        
        // Adjust for fixed-point format
        result = result >> FRAC_BITS;
        
        // Saturate to 16 bits
        if (result > 16'hFFFF) begin
            return 16'hFFFF;
        end else begin
            return result[15:0];
        end
    endfunction

endmodule : nvdla_matrix_mult