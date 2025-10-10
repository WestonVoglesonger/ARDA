// Top-level wrapper module. Implements ready/valid handshake and instantiates algorithm_core.
`timescale 1ns/1ps

module algorithm_top (
    input  logic                  clk,
    input  logic                  rst_n,

    // Ready/Valid streaming interface (single stream, sample-per-cycle)
    input  logic                  s_axis_valid,
    output logic                  s_axis_ready,
    input  logic signed [15:0]    s_axis_data, // DATA_WIDTH = 16

    output logic                  m_axis_valid,
    input  logic                  m_axis_ready,
    output logic signed [15:0]   m_axis_data
);

// Import parameters
import params_pkg::*;

// Instantiate core
algorithm_core core_inst (
    .clk        (clk),
    .rst_n      (rst_n),
    .in_valid   (s_axis_valid),
    .in_ready   (s_axis_ready),
    .in_data    (s_axis_data),

    .out_valid  (m_axis_valid),
    .out_ready  (m_axis_ready),
    .out_data   (m_axis_data)
);

endmodule
