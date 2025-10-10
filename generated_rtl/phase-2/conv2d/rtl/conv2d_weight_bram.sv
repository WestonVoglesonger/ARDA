`timescale 1ns/1ps
`include "conv2d_params.svh"

module conv2d_weight_bram (
    input  logic                 clk,
    input  logic                 rst_n,
    input  logic [7:0]           addr,
    input  logic                 enable,

    output logic [WEIGHTS_PER_CH_W-1:0] weights, // 216 bits
    output logic [BIAS_WIDTH-1:0]        bias     // 16 bits
);

// Simple synchronous ROM implemented with registers. For synthesis, this maps to BRAM/ROM.
// Weight memory: NUM_OUT_CHANNELS entries, each WEIGHTS_PER_CH_W bits + bias.

// Using arrays to model memory
logic [WEIGHTS_PER_CH_W-1:0] mem_weights [NUM_OUT_CHANNELS-1:0];
logic [BIAS_WIDTH-1:0]       mem_bias    [NUM_OUT_CHANNELS-1:0];

// initialize to zeros. In real flow, Xilinx COE or init file will populate.
initial begin
    for (int i=0; i<NUM_OUT_CHANNELS; i++) begin
        mem_weights[i] = '0;
        mem_bias[i]    = '0;
    end
end

// Read port
logic [WEIGHTS_PER_CH_W-1:0] weights_r;
logic [BIAS_WIDTH-1:0]       bias_r;

always_ff @(posedge clk) begin
    if (!rst_n) begin
        weights_r <= '0;
        bias_r <= '0;
    end else begin
        if (enable) begin
            if (addr < NUM_OUT_CHANNELS) begin
                weights_r <= mem_weights[addr];
                bias_r    <= mem_bias[addr];
            end else begin
                weights_r <= '0;
                bias_r    <= '0;
            end
        end
    end
end

assign weights = weights_r;
assign bias = bias_r;

endmodule
