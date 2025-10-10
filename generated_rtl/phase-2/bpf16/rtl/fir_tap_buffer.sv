// fir_tap_buffer.sv
// Shift-register tap buffer holding the most recent N_TAPS samples.
// Interface follows ready/valid semantics (in_valid accepted when sample_accept asserted by controller).

`include "fir_params.svh"

module fir_tap_buffer (
    input  logic                 clk,
    input  logic                 rst,        // active-high synchronous reset
    input  logic                 in_valid,   // when asserted, a new sample is offered
    input  logic                 in_ready,   // (not used internally much) indicates upstream ready - left for compatibility
    input  logic signed [IN_WIDTH-1:0] sample_in,

    output logic signed [N_TAPS*IN_WIDTH-1:0] taps_out, // newest sample first
    output logic                 out_valid   // asserted when taps_out contains N_TAPS valid samples
);

    // Internal circular buffer implemented as shift-register
    logic signed [IN_WIDTH-1:0] buf [0:N_TAPS-1];
    // Count how many valid samples have been loaded
    logic [$clog2(N_TAPS+1)-1:0] fill_cnt;

    integer i;

    // Synchronous shift on accepted sample (in_valid asserted)
    always_ff @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < N_TAPS; i = i + 1) buf[i] <= '0;
            fill_cnt <= '0;
            out_valid <= 1'b0;
        end else begin
            if (in_valid) begin
                // shift toward older indices
                for (i = N_TAPS-1; i >= 1; i = i - 1) begin
                    buf[i] <= buf[i-1];
                end
                buf[0] <= sample_in;
                if (fill_cnt < N_TAPS) fill_cnt <= fill_cnt + 1;
            end

            // Output valid once we've loaded at least N_TAPS samples
            out_valid <= (fill_cnt >= N_TAPS) && in_valid;
        end
    end

    // Pack taps_out as newest first (buf[0] is newest)
    always_comb begin
        for (i = 0; i < N_TAPS; i = i + 1) begin
            taps_out[(N_TAPS-i)*IN_WIDTH-1 -: IN_WIDTH] = buf[i];
        end
    end

endmodule
