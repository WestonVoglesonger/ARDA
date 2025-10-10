`timescale 1ns/1ps
`include "conv2d_params.svh"

module conv2d_axi_interface #(
    parameter int DATA_WIDTH = 64,
    parameter int ADDR_WIDTH = 10
) (
    input  logic                 clk,
    input  logic                 rst_n,
    input  logic [DATA_WIDTH-1:0] in_data,
    input  logic                 in_valid,
    input  logic                 out_ready,
    input  logic [128-1:0]       out_px_vec_q,

    output logic [128-1:0]       out_data,
    output logic                 out_valid,

    // Local adaptor outputs to internal datapath
    output logic [PACKED_PIXEL_W-1:0] out_pixel, // to line buffer
    output logic                      out_pixel_valid
);

// Simple adaptor: forward incoming AXI input as pixel data when valid. Assume in_data holds one packed pixel in lower bits.
assign out_pixel = in_data[PACKED_PIXEL_W-1:0];
assign out_pixel_valid = in_valid;

// Output adaptor: present out_px_vec_q as out_data when out_ready asserted by downstream
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        out_data <= '0;
        out_valid <= 1'b0;
    end else begin
        if (out_ready) begin
            out_data <= out_px_vec_q;
            out_valid <= 1'b1;
        end else begin
            out_valid <= 1'b0;
        end
    end
end

endmodule
