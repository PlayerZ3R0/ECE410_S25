/* 
Fixed-Point Arithmetic
To implement the leak factor (λ) in hardware, we need to use fixed-point arithmetic, as Verilog does not natively handle floating-point numbers in a synthesizable way. We'll represent our numbers as 16-bit values with 8 integer bits and 8 fractional bits (an 8.8 format).

A leak factor λ=0.75 would be represented as 0.75 * 2^8 = 192.
A multiplication P * \lambda becomes (P * 192) >> 8.
1. Verilog Implementation of the LIF Neuron
This module implements the binary LIF neuron as described. It has a single input I_in and outputs for the current membrane potential P_out and the spiking state S_out.
*/

`timescale 1ns / 1ps

module lif_neuron_tb;

    // Parameters for the neuron instance
    localparam DATA_WIDTH  = 16;
    localparam FRAC_WIDTH  = 8;
    localparam THRESHOLD   = 10 << FRAC_WIDTH;
    localparam LEAK_FACTOR = 243; // ~0.95 in Q8.8 format (less leaky)
    localparam RESET_VAL   = 0;

    // Testbench signals
    logic                 clk;
    logic                 rst_n;
    logic [DATA_WIDTH-1:0] I_in;
    logic [DATA_WIDTH-1:0] P_out;
    logic                 S_out;

    // Instantiate the Device Under Test (DUT)
    lif_neuron #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_WIDTH(FRAC_WIDTH),
        .THRESHOLD(THRESHOLD),
        .LEAK_FACTOR(LEAK_FACTOR),
        .RESET_VAL(RESET_VAL)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .I_in(I_in),
        .P_out(P_out),
        .S_out(S_out)
    );

    // Clock generator
    always #5 clk = ~clk;

    // Monitoring task
    task monitor_neuron;
        $display("Time=%0t | Input I=%d | Potential P=%.3f | Spike S=%b",
                 $time, I_in, (real'(P_out) / (2**FRAC_WIDTH)), S_out);
    endtask

    // Main test sequence
    initial begin
        clk = 0;
        rst_n = 0;
        I_in = 0;
        #20;
        rst_n = 1;
        $display("--- Test Start ---");

        // --- Scenario 1: Constant input below threshold ---
        $display("\n--- Scenario 1: Constant input (1.0) below threshold ---");
        I_in = 1 << FRAC_WIDTH; // Input of 1.0
        repeat (5) @(posedge clk) monitor_neuron();


        // --- Scenario 2: Input that accumulates until threshold ---
        $display("\n--- Scenario 2: Accumulating input (3.0) until spike ---");
        rst_n = 0; #20; rst_n = 1; // Reset neuron
        I_in = 3 << FRAC_WIDTH; // Input of 3.0
        repeat (5) @(posedge clk) monitor_neuron();
        
        
        // --- Scenario 3: Leakage with no input ---
        $display("\n--- Scenario 3: Leakage with no input ---");
        // Potential is already high from the previous test. Set input to 0.
        I_in = 0;
        repeat (5) @(posedge clk) monitor_neuron();
        

        // --- Scenario 4: Strong input causing immediate spiking ---
        $display("\n--- Scenario 4: Strong input (12.0) for immediate spike ---");
        rst_n = 0; #20; rst_n = 1; // Reset neuron
        I_in = 12 << FRAC_WIDTH; // Input of 12.0, which is > THRESHOLD
        @(posedge clk) monitor_neuron();
        
        $display("\n--- Test End ---");
        $finish;
    end

endmodule
