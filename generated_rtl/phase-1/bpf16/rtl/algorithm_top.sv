`include "params.svh"

module algorithm_top (
  input  logic                 clk,
  input  logic                 rst_n,

  // Streaming input interface (ready/valid)
  input  logic                 s_valid,
  output logic                 s_ready,
  input  params_pkg::in_t      s_data,

  // Streaming output interface (ready/valid)
  output logic                 m_valid,
  input  logic                 m_ready,
  output params_pkg::out_t     m_data
);

  import params_pkg::*;

  // Instantiate the algorithm core
  algorithm_core u_core (
    .clk      (clk),
    .rst_n    (rst_n),

    .s_valid  (s_valid),
    .s_ready  (s_ready),
    .s_data   (s_data),

    .m_valid  (m_valid),
    .m_ready  (m_ready),
    .m_data   (m_data)
  );

  // Top-level can be extended with optional register slices, AXI-Stream adapters,
  // or a batch processor wrapper. For this deliverable the simple passthrough
  // wrapper that instantiates the pipelined FIR core is provided.

endmodule : algorithm_top
