`timescale 1ns/1ps
`include "complex_adaptive_kalman_params.svh"

module output_buffer (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic signed [FXP_WIDTH-1:0] in_sample,
    input  logic                     in_valid,

    output logic [TAP_BUS_WIDTH-1:0] buffer_out
);

// Shift register of last FILTER_LENGTH outputs
logic signed [FXP_WIDTH-1:0] out_hist [0:FILTER_LENGTH-1];
integer m;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (m=0;m<FILTER_LENGTH;m=m+1) out_hist[m] <= '0;
    end else begin
        if (in_valid) begin
            for (m=FILTER_LENGTH-1;m>0;m=m-1) out_hist[m] <= out_hist[m-1];
            out_hist[0] <= in_sample;
        end
    end
end

always_comb begin
    for (m=0;m<FILTER_LENGTH;m=m+1) buffer_out[(m+1)*FXP_WIDTH-1 -: FXP_WIDTH] = out_hist[m];
end

endmodule
