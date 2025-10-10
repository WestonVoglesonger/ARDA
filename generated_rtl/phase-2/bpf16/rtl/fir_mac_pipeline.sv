// fir_mac_pipeline.sv
// Parallel multiply-accumulate pipeline. Accepts 16 samples and 16 coeffs and produces a pipelined 32-bit accumulator output.
// PIPELINE_DEPTH parameter controls number of pipeline registers for the accumulator path.

`include "fir_params.svh"

module fir_mac_pipeline #(
    parameter int PIPELINE_DEPTH = 4
)(
    input  logic                         clk,
    input  logic                         rst,
    input  logic                         in_valid,
    input  logic signed [N_TAPS*IN_WIDTH-1:0] samples, // packed newest first
    input  logic signed [N_TAPS*COEFF_WIDTH-1:0] coeffs,

    output logic signed [ACC_WIDTH-1:0] mac_result,
    output logic                         out_valid
);

    // Unpack samples and coeffs
    logic signed [IN_WIDTH-1:0]  s_arr [0:N_TAPS-1];
    logic signed [COEFF_WIDTH-1:0] c_arr [0:N_TAPS-1];
    integer i;

    always_comb begin
        for (i = 0; i < N_TAPS; i = i + 1) begin
            s_arr[i] = samples[(N_TAPS-i)*IN_WIDTH-1 -: IN_WIDTH];
            c_arr[i] = coeffs[(N_TAPS-i)*COEFF_WIDTH-1 -: COEFF_WIDTH];
        end
    end

    // Compute products and accumulate combinationally then pipeline the result
    // product width = IN_WIDTH + COEFF_WIDTH
    localparam int PROD_WIDTH = IN_WIDTH + COEFF_WIDTH; // e.g., 12+16=28

    logic signed [PROD_WIDTH-1:0] products [0:N_TAPS-1];
    logic signed [ACC_WIDTH-1:0]  acc_comb;

    always_comb begin
        acc_comb = '0;
        for (i = 0; i < N_TAPS; i = i + 1) begin
            products[i] = $signed(s_arr[i]) * $signed(c_arr[i]);
            // sign-extend product to accumulator width before summation
            acc_comb = acc_comb + $signed({{(ACC_WIDTH-PROD_WIDTH){products[i][PROD_WIDTH-1]}}, products[i]});
        end
    end

    // Pipeline registers for accumulator and valid signal
    logic signed [ACC_WIDTH-1:0] acc_pipe [0:PIPELINE_DEPTH-1];
    logic valid_pipe [0:PIPELINE_DEPTH-1];

    // Shift pipeline
    always_ff @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < PIPELINE_DEPTH; i = i + 1) begin
                acc_pipe[i] <= '0;
                valid_pipe[i] <= 1'b0;
            end
            mac_result <= '0;
            out_valid <= 1'b0;
        end else begin
            // stage 0 captures combinational accumulator
            acc_pipe[0] <= acc_comb;
            valid_pipe[0] <= in_valid;
            // subsequent stages
            for (i = 1; i < PIPELINE_DEPTH; i = i + 1) begin
                acc_pipe[i] <= acc_pipe[i-1];
                valid_pipe[i] <= valid_pipe[i-1];
            end
            mac_result <= acc_pipe[PIPELINE_DEPTH-1];
            out_valid <= valid_pipe[PIPELINE_DEPTH-1];
        end
    end

endmodule
