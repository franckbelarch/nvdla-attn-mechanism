// nvdla_softmax.sv - Softmax implementation for NVDLA Attention mechanism
// Implements hardware-efficient softmax approximation

`timescale 1ns/1ps

module nvdla_softmax (
    // Clock and reset
    input  logic        clk,
    input  logic        rst_n,
    
    // Control signals
    input  logic        enable,
    input  logic [31:0] vector_length,
    
    // Input interface
    input  logic        data_valid,
    input  logic [31:0] data_addr,
    input  logic [127:0] data_in,  // 8 x 16-bit values
    output logic        data_ready,
    
    // Output interface
    output logic        out_valid,
    output logic [31:0] out_addr,
    output logic [127:0] out_data, // 8 x 16-bit values
    input  logic        out_ready,
    
    // Status signals
    output logic        softmax_done,
    output logic        softmax_error
);

    // Parameters for fixed-point representation
    parameter FRAC_BITS = 8;  // Number of fractional bits
    parameter EXP_LUT_SIZE = 32; // Size of exponential lookup table
    
    // Internal state machine
    typedef enum logic [2:0] {
        IDLE,
        FIND_MAX,
        SUBTRACT_MAX,
        COMPUTE_EXP,
        COMPUTE_SUM,
        NORMALIZE,
        DONE,
        ERROR
    } softmax_state_t;
    
    softmax_state_t current_state, next_state;
    
    // Internal registers
    logic [15:0] max_value; // Maximum value in the input vector
    logic [15:0] exp_sum;   // Sum of exponentials
    logic [31:0] data_counter; // Tracks position in vector
    logic [31:0] phase_counter; // Multi-cycle operation counter
    logic [31:0] cycle_counter; // Counts cycles for timeout detection
    logic        timeout_detected; // Timeout detection flag
    
    // Buffers for processing
    logic [15:0] input_buffer [0:255]; // Buffer for input data (up to 256 elements)
    logic [15:0] exp_buffer [0:255];   // Buffer for exp results
    
    // Exponential LUT (piece-wise linear approximation)
    // Contains pre-computed values for exp(x) in the range [-8, 8]
    // In real implementation, this would be a more sophisticated LUT
    logic [15:0] exp_lut [0:EXP_LUT_SIZE-1];
    
    // Pre-computed exponential LUT with more accurate values
    // In a real implementation, this would be loaded from a parameter file or 
    // generated during synthesis for maximum accuracy
    function void initialize_exp_lut();
        // Use a better approximation for exponential function
        // e^x ≈ 1 + x + x^2/2! + x^3/3! + x^4/4! for small x
        for (int i = 0; i < EXP_LUT_SIZE; i++) begin
            // Map index to range [-8, 8]
            real x = -8.0 + i * (16.0 / (EXP_LUT_SIZE - 1));
            real ex;
            
            // More accurate piecewise approximation
            if (x <= -4.0) begin
                // Very small values (near zero)
                ex = 0.01;
            end else if (x <= 0) begin
                // For negative values, use Taylor series approximation
                ex = 1.0 + x + x*x/2.0 + x*x*x/6.0 + x*x*x*x/24.0;
            end else if (x <= 4.0) begin
                // For small positive values, use Taylor series approximation
                ex = 1.0 + x + x*x/2.0 + x*x*x/6.0 + x*x*x*x/24.0;
            end else begin
                // For large positive values, use a simple quadratic approximation
                // This is still an approximation - in real hardware, we'd use a more
                // accurate approximation or pre-computed values
                ex = 60.0 + 10.0*(x-4.0) + 20.0*(x-4.0)*(x-4.0);
            end
            
            // Scale and convert to fixed-point representation
            exp_lut[i] = 16'((ex * (1 << FRAC_BITS)) % (1 << 16));
        end
    endfunction
    
    // Initialize the exponential LUT
    initial begin
        initialize_exp_lut();
    end
    
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
                    next_state = FIND_MAX;
                end
            end
            
            FIND_MAX: begin
                if (data_counter >= vector_length) begin
                    next_state = SUBTRACT_MAX;
                end
            end
            
            SUBTRACT_MAX: begin
                // After processing all inputs
                if (data_counter >= vector_length) begin
                    next_state = COMPUTE_EXP;
                end
            end
            
            COMPUTE_EXP: begin
                // After computing exp for all values
                if (data_counter >= vector_length) begin
                    next_state = COMPUTE_SUM;
                end
            end
            
            COMPUTE_SUM: begin
                // After summing all exp values
                if (data_counter >= vector_length) begin
                    next_state = NORMALIZE;
                end
            end
            
            NORMALIZE: begin
                // After normalizing all values
                if (data_counter >= vector_length) begin
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
    
    // Data processing logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_counter <= 0;
            phase_counter <= 0;
            cycle_counter <= 0;
            max_value <= 16'h8000; // Smallest possible 16-bit value
            exp_sum <= 0;
            softmax_done <= 0;
            softmax_error <= 0;
            timeout_detected <= 0;
            data_ready <= 0;
            out_valid <= 0;
        end else begin
            // Increment cycle counter for timeout detection
            cycle_counter <= cycle_counter + 1;
            
            // Timeout detection (50000 cycles is arbitrary - adjust based on expected workload)
            if (cycle_counter > 32'd50000 && current_state != IDLE && !softmax_done) begin
                timeout_detected <= 1'b1;
                softmax_error <= 1'b1;
            end
            case (current_state)
                IDLE: begin
                    data_counter <= 0;
                    phase_counter <= 0;
                    cycle_counter <= 0; // Reset cycle counter when idle
                    timeout_detected <= 0; // Reset timeout flag
                    max_value <= 16'h8000; // Smallest possible 16-bit value
                    exp_sum <= 0;
                    softmax_done <= 0;
                    softmax_error <= 0;
                    data_ready <= enable;
                    out_valid <= 0;
                end
                
                FIND_MAX: begin
                    // Read input values and find maximum
                    data_ready <= 1;
                    
                    if (data_valid) begin
                        // Store input data
                        for (int i = 0; i < 8; i++) begin
                            if (data_counter + i < vector_length) begin
                                input_buffer[data_counter + i] = data_in[i*16 +: 16];
                                
                                // Update max value
                                if (data_in[i*16 +: 16] > max_value) begin
                                    max_value <= data_in[i*16 +: 16];
                                end
                            end
                        end
                        
                        data_counter <= data_counter + 8;
                    end
                end
                
                SUBTRACT_MAX: begin
                    // Subtract max from all inputs for numerical stability
                    data_ready <= 0;
                    
                    if (phase_counter < (vector_length + 7) / 8) begin
                        for (int i = 0; i < 8; i++) begin
                            if (phase_counter * 8 + i < vector_length) begin
                                input_buffer[phase_counter * 8 + i] <= 
                                    input_buffer[phase_counter * 8 + i] - max_value;
                            end
                        end
                        phase_counter <= phase_counter + 1;
                    end else begin
                        phase_counter <= 0;
                        data_counter <= 0;
                    end
                end
                
                COMPUTE_EXP: begin
                    // Compute exponential for each value
                    if (phase_counter < (vector_length + 7) / 8) begin
                        for (int i = 0; i < 8; i++) begin
                            if (phase_counter * 8 + i < vector_length) begin
                                // Compute exp using LUT or piece-wise linear approximation
                                exp_buffer[phase_counter * 8 + i] <= 
                                    compute_exp(input_buffer[phase_counter * 8 + i]);
                            end
                        end
                        phase_counter <= phase_counter + 1;
                    end else begin
                        phase_counter <= 0;
                        data_counter <= 0;
                    end
                end
                
                COMPUTE_SUM: begin
                    // Sum all exponential values
                    if (data_counter < vector_length) begin
                        exp_sum <= exp_sum + exp_buffer[data_counter];
                        data_counter <= data_counter + 1;
                    end else begin
                        data_counter <= 0;
                    end
                end
                
                NORMALIZE: begin
                    // Normalize by dividing each exp value by the sum
                    out_valid <= 1;
                    
                    if (out_ready) begin
                        // Prepare output data (8 values per cycle)
                        for (int i = 0; i < 8; i++) begin
                            if (data_counter + i < vector_length) begin
                                // Fixed-point division
                                out_data[i*16 +: 16] <= divide_fixed_point(
                                    exp_buffer[data_counter + i], exp_sum);
                            end else begin
                                out_data[i*16 +: 16] <= 0;
                            end
                        end
                        
                        out_addr <= data_counter;
                        data_counter <= data_counter + 8;
                    end
                end
                
                DONE: begin
                    out_valid <= 0;
                    softmax_done <= 1;
                end
                
                ERROR: begin
                    softmax_error <= 1;
                end
            endcase
        end
    end
    
    // Helper function to compute exponential using LUT
    function logic [15:0] compute_exp(input logic [15:0] x);
        logic [15:0] result;
        logic [31:0] idx;
        
        // Clamp input range to [-8, 8] in fixed-point representation
        logic [15:0] clamped_x;
        if (x[15]) begin // Negative
            clamped_x = (x < -8 * (1 << FRAC_BITS)) ? -8 * (1 << FRAC_BITS) : x;
        end else begin // Positive
            clamped_x = (x > 8 * (1 << FRAC_BITS)) ? 8 * (1 << FRAC_BITS) : x;
        end
        
        // Map x from [-8, 8] to [0, EXP_LUT_SIZE-1]
        idx = ((clamped_x + 8 * (1 << FRAC_BITS)) * (EXP_LUT_SIZE - 1)) / (16 * (1 << FRAC_BITS));
        
        // Ensure index is within bounds
        if (idx >= EXP_LUT_SIZE) begin
            idx = EXP_LUT_SIZE - 1;
        end
        
        // Lookup result
        result = exp_lut[idx];
        
        return result;
    endfunction
    
    // Helper function for fixed-point division
    function logic [15:0] divide_fixed_point(input logic [15:0] numerator, input logic [15:0] denominator);
        logic [31:0] temp;
        logic [15:0] result;
        
        // Scale numerator for fixed-point division
        temp = numerator;
        temp = temp << FRAC_BITS;
        
        // Perform division
        if (denominator == 0) begin
            // Handle division by zero
            result = 16'hFFFF; // Return maximum value
        end else begin
            result = temp / denominator;
        end
        
        return result;
    endfunction

endmodule : nvdla_softmax