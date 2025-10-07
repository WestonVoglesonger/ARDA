// -----------------------------------------------------------------------------
// BPF16 - 16-tap Band-Pass FIR Filter with AXI-Stream Interface
// - 12-bit fixed-point input, 16-bit fixed-point output
// - AXI-Stream TVALID/TREADY handshake, 1 sample in, 1 sample out
// - 4-stage pipelining for timing, 16-cycle delay line
// -----------------------------------------------------------------------------

module bpf16_axi_stream #(
    parameter integer INPUT_WIDTH        = 12,
    parameter integer INPUT_FRAC_BITS    = 11,
    parameter integer COEFF_WIDTH        = 12,
    parameter integer COEFF_FRAC_BITS    = 11,
    parameter integer OUTPUT_WIDTH       = 16,
    parameter integer OUTPUT_FRAC_BITS   = 14,
    parameter integer TAP_NUM            = 16
) (
    input  wire                        clk,
    input  wire                        rst_n,

    // AXI-Stream input
    input  wire                        s_axis_tvalid,
    output wire                        s_axis_tready,
    input  wire signed [INPUT_WIDTH-1:0] s_axis_tdata,

    // AXI-Stream output
    output reg                         m_axis_tvalid,
    input  wire                        m_axis_tready,
    output reg  signed [OUTPUT_WIDTH-1:0] m_axis_tdata
);

    // Internal pipeline and handshake logic
    reg [3:0]                           pipe_val;
    wire                                pipeline_ready;
    assign pipeline_ready = m_axis_tready;

    // TVALID/TREADY handshake for input
    assign s_axis_tready = pipeline_ready;

    // Delay line for input samples (shift register)
    reg signed [INPUT_WIDTH-1:0] x_delay [0:TAP_NUM-1];
    integer i;

    // Coefficient ROM
    wire signed [COEFF_WIDTH-1:0] coeff_rom [0:TAP_NUM-1];
    `include "bpf16_coeff_rom.vh"

    // Pipeline registers (4 stages)
    reg signed [OUTPUT_WIDTH+4:0] pipe_reg [0:3]; // Extra bits for accum intermediate
    reg [3:0]                    pipe_data_valid;

    // Accumulator and control
    reg signed [OUTPUT_WIDTH+4:0] acc;
    reg [3:0]                    sample_ptr;
    reg                          calc_en;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < TAP_NUM; i = i + 1) begin
                x_delay[i] <= '0;
            end
            acc                <= '0;
            m_axis_tdata       <= '0;
            m_axis_tvalid      <= 1'b0;
            sample_ptr         <= '0;
            calc_en            <= 1'b0;
            pipe_data_valid    <= 4'b0;
            for (i = 0; i < 4; i = i + 1)
                pipe_reg[i] <= '0;
        end else begin
            // Handshake
            if (s_axis_tvalid && s_axis_tready) begin
                // Shift delay line
                for (i = TAP_NUM-1; i > 0; i = i - 1)
                    x_delay[i] <= x_delay[i-1];
                x_delay[0] <= s_axis_tdata;
            end

            // FIR computation (fully parallel multiply-accumulate)
            // Start calculation when new input valid
            if (s_axis_tvalid && s_axis_tready) begin
                acc <= '0;
                for (i = 0; i < TAP_NUM; i = i + 1)
                    acc <= acc + x_delay[i] * coeff_rom[i];

                calc_en         <= 1'b1;
            end else begin
                calc_en         <= 1'b0;
            end

            // Pipeline (to meet timing at 200 MHz)
            pipe_reg[0]       <= acc;
            pipe_data_valid[0]<= calc_en;
            for (i = 1; i < 4; i = i + 1) begin
                pipe_reg[i]        <= pipe_reg[i-1];
                pipe_data_valid[i] <= pipe_data_valid[i-1];
            end

            // Output assignment
            m_axis_tvalid    <= pipe_data_valid[3];
            if (pipe_data_valid[3]) begin
                // Output rounding with shift according to fixed-point format
                m_axis_tdata <= pipe_reg[3][OUTPUT_FRAC_BITS+COEFF_FRAC_BITS+INPUT_FRAC_BITS-1 -: OUTPUT_WIDTH];
            end
        end
    end

endmodule
