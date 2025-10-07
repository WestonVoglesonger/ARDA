// AXI-Stream interface module for BPF16 FIR filter
// Synthesizable SystemVerilog
// 1 input/output sample per clock, fixed-point, 16-tap, fully pipelined

module BPF16_axi_stream_if #(
    parameter IN_WIDTH = 12,
    parameter IN_FRAC  = 11,
    parameter OUT_WIDTH = 16,
    parameter OUT_FRAC  = 14
) (
    input  logic                   aclk,
    input  logic                   aresetn,
    // AXI-Stream slave input (s_axis)
    input  logic [IN_WIDTH-1:0]    s_axis_tdata,
    input  logic                   s_axis_tvalid,
    output logic                   s_axis_tready,
    // AXI-Stream master output (m_axis)
    output logic [OUT_WIDTH-1:0]   m_axis_tdata,
    output logic                   m_axis_tvalid,
    input  logic                   m_axis_tready
);

    // Internal handshake
    logic                           pipe_valid;
    logic                           pipe_ready;
    logic [IN_WIDTH-1:0]           pipe_data;
    
    // Pipeline register for input handshake
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            pipe_valid <= 1'b0;
        end else if (s_axis_tvalid && s_axis_tready) begin
            pipe_data  <= s_axis_tdata;
            pipe_valid <= 1'b1;
        end else if (pipe_ready && pipe_valid) begin // downstream consumed
            pipe_valid <= 1'b0;
        end
    end
    assign s_axis_tready = (!pipe_valid) || (pipe_ready && pipe_valid);

    // Wire output to core filter module
    logic [OUT_WIDTH-1:0]          core_out;
    logic                          core_valid, core_ready;
    
    BPF16_core #(
        .IN_WIDTH(IN_WIDTH),
        .IN_FRAC(IN_FRAC),
        .OUT_WIDTH(OUT_WIDTH),
        .OUT_FRAC(OUT_FRAC)
    ) uut (
        .clk        (aclk),
        .rstn       (aresetn),
        .din        (pipe_data),
        .din_valid  (pipe_valid),
        .din_ready  (pipe_ready),
        .dout       (core_out),
        .dout_valid (core_valid),
        .dout_ready (core_ready)
    );

    // Connect AXI-Stream output to core
    assign core_ready   = m_axis_tready;
    assign m_axis_tvalid = core_valid;
    assign m_axis_tdata  = core_out;
endmodule
