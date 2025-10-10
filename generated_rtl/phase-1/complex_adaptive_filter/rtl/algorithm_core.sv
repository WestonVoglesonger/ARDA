// Core computation module: FIR filter with adaptive coefficients, simple Kalman-like state update,
// nonlinear post-processing (softsign approximation), and pipelined outputs.

`timescale 1ns/1ps

module algorithm_core (
    input  logic              clk,
    input  logic              rst_n,

    // Ready/Valid handshake (throughput: 1 sample/cycle)
    input  logic              in_valid,
    output logic              in_ready,
    input  logic signed [DATA_WIDTH-1:0]  in_data,

    output logic              out_valid,
    input  logic              out_ready,
    output logic signed [DATA_WIDTH-1:0] out_data
);

// Import parameters and types
import params_pkg::*;

// Internal storage
// Sample shift register (most recent sample at index 0)
logic signed [DATA_WIDTH-1:0] samples [0:TAP_COUNT-1];
// Coefficients (adaptive)
logic signed [COEFF_WIDTH-1:0] coeffs [0:TAP_COUNT-1];
// Simple state vector for Kalman-like updates (small dimension)
localparam int STATE_DIM = 8;
logic signed [DATA_WIDTH-1:0] state_vec [0:STATE_DIM-1];

// Pipeline registers for output (depth PIPELINE_DEPTH)
logic                         pipe_valid [0:PIPELINE_DEPTH-1];
logic signed [DATA_WIDTH-1:0] pipe_data  [0:PIPELINE_DEPTH-1];

// Combinational wires for products and accumulation
wire signed [ACC_WIDTH-1:0] products [0:TAP_COUNT-1];
acc_t mac_sum;

// Initialize coeffs from ROM on reset
integer i;
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < TAP_COUNT; i++) begin
            coeffs[i] <= COEFFS[i];
        end
        // Clear sample buffer and state
        for (i = 0; i < TAP_COUNT; i++) samples[i] <= '0;
        for (i = 0; i < STATE_DIM; i++) state_vec[i] <= '0;
        // Clear pipeline
        for (i = 0; i < PIPELINE_DEPTH; i++) begin
            pipe_valid[i] <= 1'b0;
            pipe_data[i] <= '0;
        end
    end else begin
        // Default in_ready: accept samples every cycle if not back-pressured by output ready
        // We implement simple flow-control: when downstream is not ready and pipeline full, we deassert in_ready.
        // The pipe depth matching and back-pressure handled in combinational assignment below.

        // Shift pipeline registers (advance if downstream allows)
        // We always shift data forward; out_valid will be pipe_valid[PIPELINE_DEPTH-1]
        for (i = PIPELINE_DEPTH-1; i > 0; i--) begin
            pipe_valid[i] <= pipe_valid[i-1];
            pipe_data[i]  <= pipe_data[i-1];
        end

        // If a new sample was accepted, feed computed final value into pipeline stage 0 below (assigned later)

        // Simple state decay to maintain numerical stability (small leak)
        for (i = 0; i < STATE_DIM; i++) begin
            // state_vec[i] *= 0.999 approx by subtracting right-shifted value (small decay)
            state_vec[i] <= state_vec[i] - (state_vec[i] >>> 10);
        end

    end
end

// Combinational product generation (parallel multipliers)
generate
    genvar g;
    for (g = 0; g < TAP_COUNT; g = g + 1) begin : GEN_PROD
        // Multiply sample (Q1.DATA_FRAC) by coeff (Q1.COEFF_FRAC) => product Q2.(DATA_FRAC+COEFF_FRAC)
        assign products[g] = $signed(coeffs[g]) * $signed(samples[g]);
    end
endgenerate

// MAC and nonlinear processing performed combinationally each cycle for throughput=1
always_comb begin
    // Default values
    mac_sum = '0;
    // Accumulate products
    for (i = 0; i < TAP_COUNT; i++) begin
        mac_sum = mac_sum + products[i];
    end

    // mac_sum is in Q( (DATA_FRAC+COEFF_FRAC) ) (here DATA_FRAC+COEFF_FRAC = 30)
    // Convert back to Q1.DATA_FRAC by arithmetic right shift of COEFF_FRAC bits
    // We will shift by COEFF_FRAC (15)
    acc_t acc_shifted = mac_sum >>> COEFF_FRAC; // Now in Q1.DATA_FRAC (aligned with input)

    // Apply nonlinear post-processing: softsign approximation tanh-like: y = x / (1 + |x|)
    // All values are Q1.DATA_FRAC. To compute x/(1+|x|) in fixed point safely:
    // y = (x << DATA_FRAC) / ( (1<<DATA_FRAC) + abs(x) )  => result in Q1.DATA_FRAC
    logic signed [DATA_WIDTH-1:0] acc_q;
    acc_q = acc_shifted[DATA_WIDTH-1:0];
    logic signed [DATA_WIDTH-1:0] abs_acc_q;
    if (acc_q[DATA_WIDTH-1]) abs_acc_q = -acc_q; else abs_acc_q = acc_q;

    // denominator = 1.0 + |x| in Q1.DATA_FRAC
    logic signed [DATA_WIDTH-1:0] denom_q;
    denom_q = (16'sd1 << DATA_FRAC) + abs_acc_q; // 1.0 in Q format is (1<<DATA_FRAC)

    // Compute numerator = x << DATA_FRAC to keep precision for division
    logic signed [ (DATA_WIDTH*2)-1:0 ] numer_ext;
    numer_ext = $signed({{DATA_WIDTH{acc_q[DATA_WIDTH-1]}}, acc_q}) <<< DATA_FRAC; // extend sign then shift

    // Perform division: y = numer_ext / denom_q (result ~ Q1.DATA_FRAC)
    // Use integer division; synthesis tools usually implement this using DSP/logic
    logic signed [DATA_WIDTH-1:0] nonlin_q;
    if (denom_q != 0)
        nonlin_q = $signed(numer_ext / denom_q);
    else
        nonlin_q = acc_q;

    // State estimation update (Kalman-like simplified): innovation = nonlinearity - state_vec[0]
    logic signed [DATA_WIDTH-1:0] innovation;
    innovation = nonlin_q - state_vec[0];

    // Simple approximate Kalman update: state[0] += innovation * gain; choose gain = 1/16 -> >>4
    logic signed [DATA_WIDTH-1:0] state0_update;
    state0_update = state_vec[0] + (innovation >>> 4);

    // Propagate to other state dimensions with decaying coupling
    logic signed [DATA_WIDTH-1:0] state_contrib_sum;
    state_contrib_sum = '0;
    for (i = 0; i < 4; i++) begin
        // compute temp contribution (propagate portion of state)
        state_contrib_sum = state_contrib_sum + state_vec[i];
    end
    // average of first 4 states: >>2
    state_contrib_sum = state_contrib_sum >>> 2;

    // Final output generation: combine nonlinearity with state contribution
    logic signed [DATA_WIDTH-1:0] final_out_q;
    final_out_q = nonlin_q + state_contrib_sum;

    // Prepare pipeline stage 0 inputs
    pipe_data[0]  = final_out_q;
    pipe_valid[0] = in_valid; // when a new input is provided we pipe the output (combinationally)
end

// Update sample buffer, states and adaptive coefficients on accepting input (in_valid && in_ready)
// in_ready is asserted when we have room in the pipeline (i.e., when the final stage isn't holding data while downstream is not ready)
logic pipeline_full;
assign pipeline_full = pipe_valid[PIPELINE_DEPTH-1] && !out_ready; // cannot advance if last stage held and downstream not ready
assign in_ready = !pipeline_full;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // handled above in reset block
    end else begin
        if (in_valid && in_ready) begin
            // Shift sample buffer
            for (i = TAP_COUNT-1; i > 0; i--) samples[i] <= samples[i-1];
            samples[0] <= in_data;

            // Commit state update computed combinationally above
            state_vec[0] <= state_vec[0] + (( (nonblocking_tie_zero()) , ( /* placeholder to keep linter happy */ 0)) );
            // The actual state update is done below to avoid mixed combinational/clocked reuse
        end

        // Now perform clocked state update and adaptation logic after samples updated
        if (in_valid && in_ready) begin
            // Recompute innovation and state update in clocked context
            logic signed [DATA_WIDTH-1:0] acc_q_c;
            acc_q_c = pipe_data[0]; // this value was computed combinationally earlier as pipe_data[0]

            logic signed [DATA_WIDTH-1:0] innovation_c;
            innovation_c = acc_q_c - state_vec[0];
            // State update
            state_vec[0] <= state_vec[0] + (innovation_c >>> 4);
            // Propagate to other dimensions with small scaled contribution
            for (i = 1; i < STATE_DIM; i++) begin
                // coupling_factor approximated by shifting (smaller for larger i)
                state_vec[i] <= state_vec[i] + (innovation_c >>> (4 + (i/2)));
            end

            // Adaptive coefficient update (simple sign-LMS approximation to limit complexity)
            logic signed [DATA_WIDTH-1:0] err_q;
            err_q = in_data - acc_q_c; // error in Q format
            logic signed [DATA_WIDTH-1:0] abs_err;
            if (err_q[DATA_WIDTH-1]) abs_err = -err_q; else abs_err = err_q;

            if (abs_err > ADAPT_THRESHOLD_Q) begin
                // Use a conservative sign-LMS style update: coeff += sign(error) * (sample >> 6)
                for (i = 0; i < TAP_COUNT; i++) begin
                    // compute small delta from sample
                    logic signed [COEFF_WIDTH-1:0] delta;
                    // sample >> 6 reduces magnitude (learning rate ~1/64)
                    delta = samples[i] >>> 6; // keeps sign
                    if (!err_q[DATA_WIDTH-1])
                        coeffs[i] <= coeffs[i] + { { (COEFF_WIDTH-DATA_WIDTH){delta[DATA_WIDTH-1]} }, delta };
                    else
                        coeffs[i] <= coeffs[i] - { { (COEFF_WIDTH-DATA_WIDTH){delta[DATA_WIDTH-1]} }, delta };

                    // Constrain coefficients to avoid runaway (saturate at +/- (1.0 - small epsilon))
                    // Max coefficient magnitude approx (1.0 in Q1.COEFF_FRAC = (1<<COEFF_FRAC)-1)
                    if (coeffs[i] > ( (1 << (COEFF_FRAC)) - 1 )) coeffs[i] <= (1 << (COEFF_FRAC)) - 1;
                    if (coeffs[i] < - (1 << (COEFF_FRAC)) ) coeffs[i] <= - (1 << (COEFF_FRAC));
                end
            end
        end

        // Advance pipeline registers (clocked already in reset block), but ensure the new stage0 values are kept
        // pipe_data[0] and pipe_valid[0] are assigned combinationally; they have been latched in the earlier always_ff block

    end
end

// Provide outputs from the last pipeline stage
assign out_valid = pipe_valid[PIPELINE_DEPTH-1];
assign out_data  = pipe_data[PIPELINE_DEPTH-1];

// Helper: a tiny function to avoid mixed-signal warnings when generating code above
function automatic int nonblocking_tie_zero();
    nonblocking_tie_zero = 0;
endfunction

endmodule
