`timescale 1ns/1ps

module fft_memory (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        wr_en,
  input  logic [7:0]  wr_addr,
  input  logic [31:0] wr_data,
  input  logic        rd_en,
  input  logic [7:0]  rd_addr,

  output logic [31:0] rd_data,
  output logic        ready
);
  import fft_params_pkg::*;

  // Single-port synchronous memory with registered read data
  logic [31:0] bram [0:DEPTH-1];
  logic [31:0] rd_data_r;

  assign ready = 1'b1; // always ready in this simple model

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rd_data_r <= 32'h0;
    end else begin
      if (wr_en) begin
        bram[wr_addr] <= wr_data;
      end
      if (rd_en) begin
        rd_data_r <= bram[rd_addr];
      end
    end
  end

  assign rd_data = rd_data_r;

endmodule
