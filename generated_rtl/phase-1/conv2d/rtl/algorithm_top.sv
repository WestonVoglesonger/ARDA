`timescale 1ns/1ps

// algorithm_top.sv
// Top-level wrapper for the FIR/convolution core with ready/valid handshake.
// This module exposes the same ready/valid streaming interface and instantiates
// algorithm_core which implements the fixed-point pipelined accumulation and ReLU.

module algorithm_top (
  input  logic                    clk,
  input  logic                    rst_n,

  // Streaming interface (host <-> algorithm)
  input  logic                    s_axis_valid,
  output logic                    s_axis_ready,
  input  logic signed [DATA_WIDTH-1:0] s_axis_data,

  output logic                    m_axis_valid,
  input  logic                    m_axis_ready,
  output logic signed [DATA_WIDTH-1:0] m_axis_data
);

  // Import parameters and types
  import params_pkg::*;

  // Simple pass-through to algorithm_core
  algorithm_core u_core (
    .clk      (clk),
    .rst_n    (rst_n),
    .in_valid (s_axis_valid),
    .in_ready (s_axis_ready),
    .in_data  (s_axis_data),
    .out_valid(m_axis_valid),
    .out_ready(m_axis_ready),
    .out_data (m_axis_data)
  );

  // Note: Additional top-level responsibilities (weight loading, BRAM management,
  // multi-channel mapping for 2D conv, etc.) are not included here; this top wraps
  // the compute core and exposes a ready/valid streaming interface compatible with
  // the microarchitecture handshake specified.

endmodule : algorithm_top
