// fir_control_fsm.sv
// Simple controller for ready/valid handshake and pipeline start/reset logic.
// It accepts upstream in_valid/in_ready and downstream out_ready and generates sample_accept and output_valid.

module fir_control_fsm (
    input  logic clk,
    input  logic rst,
    input  logic in_valid,   // upstream valid
    input  logic in_ready,   // upstream ready (external) - treated as permission to accept
    input  logic out_ready,  // downstream ready

    output logic sample_accept, // assert when we accept an input into the pipeline
    output logic output_valid   // assert when top-level output is valid
);

    // Pipeline depth must match other modules. We hardcode 4 (from architecture)
    localparam int PIPELINE_DEPTH = 4;
    // simple shift register to track when accepted samples produce outputs
    logic valid_pipe [0:PIPELINE_DEPTH-1];
    integer i;

    always_ff @(posedge clk) begin
        if (rst) begin
            sample_accept <= 1'b0;
            output_valid <= 1'b0;
            for (i = 0; i < PIPELINE_DEPTH; i = i + 1) valid_pipe[i] <= 1'b0;
        end else begin
            // Accept sample when upstream asserts valid and indicates ready (handshake)
            sample_accept <= in_valid & in_ready;
            // shift the valid pipeline
            valid_pipe[0] <= sample_accept;
            for (i = 1; i < PIPELINE_DEPTH; i = i + 1) valid_pipe[i] <= valid_pipe[i-1];
            // Output valid is asserted when the last stage becomes valid and downstream is ready
            // We still present valid; downstream must latch/consume based on out_ready
            output_valid <= valid_pipe[PIPELINE_DEPTH-1];
        end
    end

endmodule
