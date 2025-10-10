`timescale 1ns/1ps
`include "conv2d_params.svh"

module conv2d_output_buffer #(
    parameter int DEPTH = 32
) (
    input  logic [NUM_OUT_CHANNELS-1:0][7:0] in_px_vec_q,
    input  logic                      in_valid,
    input  logic                      clk,
    input  logic                      out_ready,

    output logic [128-1:0]            out_data,
    output logic                      out_valid
);

// Simple FIFO buffer implemented as ring buffer storing 128-bit entries (16x8bits)
logic [127:0] fifo_mem [DEPTH-1:0];
logic [$clog2(DEPTH)-1:0] wr_ptr, rd_ptr;
logic [($bits(integer))-1:0] count;

// pack input vector into 128-bit word
logic [127:0] packed_in;
always_comb begin
    packed_in = '0;
    for (int i=0;i<NUM_OUT_CHANNELS;i++) begin
        int off = (NUM_OUT_CHANNELS-1 - i)*8;
        packed_in[off +: 8] = in_px_vec_q[i];
    end
end

always_ff @(posedge clk) begin
    if (in_valid && (count < DEPTH)) begin
        fifo_mem[wr_ptr] <= packed_in;
        wr_ptr <= wr_ptr + 1;
        count <= count + 1;
    end

    if (out_ready && (count > 0)) begin
        out_data <= fifo_mem[rd_ptr];
        rd_ptr <= rd_ptr + 1;
        count <= count - 1;
        out_valid <= 1'b1;
    end else begin
        out_valid <= 1'b0;
    end
end

endmodule
