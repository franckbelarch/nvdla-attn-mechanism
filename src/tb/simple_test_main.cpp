#include <iostream>
#include <verilated.h>
#include <verilated_fst_c.h>

// Include model header, generated from Verilating the design
#include "Vsimple_test.h"

// VCD/FST file for waveform
VerilatedFstC* tfp = nullptr;

// Current simulation time
vluint64_t main_time = 0;

// Called by $time in Verilog
double sc_time_stamp() {
    return main_time;
}

int main(int argc, char** argv) {
    // Initialize Verilators variables
    Verilated::commandArgs(argc, argv);
    
    // Create logs/ directory in case it doesn't exist
    Verilated::mkdir("logs");
    
    // Create an instance of the model
    Vsimple_test* tb = new Vsimple_test();
    
    // Setup waveform tracing
    Verilated::traceEverOn(true);
    tfp = new VerilatedFstC;
    tb->trace(tfp, 99);
    tfp->open("logs/simple_test.fst");
    
    // Run simulation for 5000 clock cycles or until $finish
    std::cout << "Starting simulation..." << std::endl;
    
    // Clock signal
    bool clk = false;
    
    while (!Verilated::gotFinish() && main_time < 10000) {
        // Toggle clock
        clk = !clk;
        tb->clk_i = clk;
        
        // Evaluate model
        tb->eval();
        
        // Dump trace data
        if (tfp) tfp->dump(main_time);
        
        // Advance time
        main_time++;
    }
    
    // Cleanup
    tfp->close();
    delete tb;
    delete tfp;

    // Return good completion status
    std::cout << "Simulation completed at " << main_time << " ticks" << std::endl;
    return 0;
}