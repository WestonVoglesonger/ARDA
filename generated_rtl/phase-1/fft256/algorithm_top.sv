/* algorithm_top.sv - Top-level wrapper for the FFT streaming core
   - Exposes ready/valid handshake interface
   - Accepts interleaved real/imag input (DATA_WIDTH each)
   - Produces interleaved real/imag output (OUTPUT_WIDTH each)
   - Instantiates algorithm_core
*/

`timescale 1ns/1ps
`include "params.svh"

module algorithm_top (
  input  logic            clk,
  input  logic            rst_n,

  // Host-facing streaming interface (ready/valid)
  input  logic            s_axis_valid,
  output logic            s_axis_ready,
  input  logic signed [DATA_WIDTH*2-1:0] s_axis_data, // {real, imag}

  output logic            m_axis_valid,
  input  logic            m_axis_ready,
  output logic signed [OUTPUT_WIDTH*2-1:0] m_axis_data
);

  // Instantiate algorithm_core
  algorithm_core core_i (
    .clk      (clk),
    .rst_n    (rst_n),
    .in_valid (s_axis_valid),
    .in_ready (s_axis_ready),
    .in_data  (s_axis_data),
    .out_valid(m_axis_valid),
    .out_ready(m_axis_ready),
    .out_data (m_axis_data)
  );

endmodule
