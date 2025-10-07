// AXI-Stream Wrapper for BPF16 Core
// Connects AXI-Stream to datapath and coefficient ROM
module bpf16_axi_wrapper (
    input  logic clk,
    input  logic rst_n,
    // AXI-Stream Slave In
    input  logic s_axis_tvalid,
    output logic s_axis_tready,
    input  logic signed [11:0] s_axis_tdata,
    // AXI-Stream Master Out
    output logic m_axis_tvalid,
    input  logic m_axis_tready,
    output logic signed [15:0] m_axis_tdata
);
    // Instantiate Coefficient ROM (connect to core via vector assignment)
    logic signed [11:0] coeffs [0:15];
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : coeff_blk
            coeff_rom_bpf16 coeff_inst(
                .addr  (i[$clog2(16)-1:0]),
                .data  (coeffs[i])
            );
        end
    endgenerate

    // Instantiate Core
    bpf16_core #(
        .N_TAPS(16),
        .IN_WIDTH(12),
        .OUT_WIDTH(16),
        .FXP_IN_FRAC(11),
        .FXP_OUT_FRAC(14)
    ) core_inst (
        .clk              (clk),
        .rst_n            (rst_n),
        .s_axis_tvalid    (s_axis_tvalid),
        .s_axis_tready    (s_axis_tready),
        .s_axis_tdata     (s_axis_tdata),
        .m_axis_tvalid    (m_axis_tvalid),
        .m_axis_tready    (m_axis_tready),
        .m_axis_tdata     (m_axis_tdata)
    );
endmodule