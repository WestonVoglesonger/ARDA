`timescale 1ns/1ps
`include "complex_adaptive_kalman_params.svh"

module input_buffer (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     in_valid,
    input  logic                     in_ready,
    input  logic signed [FXP_WIDTH-1:0] normalized_sample,

    output logic                     out_valid,
    output logic                     out_ready,
    output logic [TAP_BUS_WIDTH-1:0] buffer_out // packed parallel taps
);

// Internal shift register for taps
logic signed [FXP_WIDTH-1:0] taps [0:FILTER_LENGTH-1];
integer i;

// Simple handshake: capture when in_valid & in_ready
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i=0;i<FILTER_LENGTH;i=i+1) taps[i] <= '0;
        out_valid <= 1'b0;
        out_ready <= 1'b1;
    end else begin
        if (in_valid & in_ready) begin
            // shift right: newest at taps[0]
            for (i=FILTER_LENGTH-1;i>0;i=i-1) taps[i] <= taps[i-1];
            taps[0] <= normalized_sample;
            out_valid <= 1'b1;
        end else if (out_valid & out_ready) begin
            out_valid <= 1'b0;
        end

        // simple flow control: always ready unless out_valid asserted
        if (!out_valid) out_ready <= 1'b1; else out_ready <= 1'b0;
    end
end

// Pack taps into parallel bus combinationally
always_comb begin
    for (i=0;i<FILTER_LENGTH;i=i+1) begin
        buffer_out[(i+1)*FXP_WIDTH-1 -: FXP_WIDTH] = taps[i];
    end
end

endmodule
