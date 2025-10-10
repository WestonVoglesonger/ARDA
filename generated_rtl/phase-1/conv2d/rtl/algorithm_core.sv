`timescale 1ns/1ps

// algorithm_core.sv
// 16-tap FIR-style convolution core (fixed-point) with ready/valid handshake.
// - Uses coefficients from params_pkg (quantized)
// - Fixed-point arithmetic Q6 for inputs and coefficients
// - Pipelined output registers = PIPELINE_DEPTH
// - ReLU activation applied at output stage

module algorithm_core (
  input  logic                    clk,
  input  logic                    rst_n,

  // Input (master -> core)
  input  logic                    in_valid,
  output logic                    in_ready,
  input  logic signed [DATA_WIDTH-1:0] in_data,

  // Output (core -> master)
  output logic                    out_valid,
  input  logic                    out_ready,
  output logic signed [DATA_WIDTH-1:0] out_data
);

  // Import parameters and types
  import params_pkg::*;

  // Internal shift register to hold last COEFF_COUNT samples
  fxp_t sample_sr [0:COEFF_COUNT-1];

  // Combinational accumulator (sum of products). Each product = DATA_WIDTH + COEFF_WIDTH -> keep ACC_WIDTH
  acc_t comb_acc;

  // Pipeline registers for accumulator
  acc_t acc_pipe [0:PIPELINE_DEPTH-1];

  // Valid pipeline to track when outputs are valid
  logic valid_pipe [0:PIPELINE_DEPTH-1];

  // Simple in-flight counter to provide backpressure
  logic [$clog2(PIPELINE_DEPTH+1)-1:0] inflight_count;

  // Default assignments
  assign in_ready = (inflight_count < PIPELINE_DEPTH);

  // Compute combinational sum of products (synchronously depending on shift reg content)
  // product_i = sample_sr[i] * COEFFS[i]  (both signed)
  // Note: sample_sr is Q6, COEFFS is Q6 => product Q12 (we keep full ACC_WIDTH for headroom)
  integer i;
  always_comb begin
    comb_acc = '0;
    for (i = 0; i < COEFF_COUNT; i = i + 1) begin
      // Extend operands to signed representations for multiplication
      // Cast to signed explicitly to avoid unsigned arithmetic
      comb_acc = comb_acc + acc_t($signed(sample_sr[i]) * $signed(COEFFS[i]));
    end
  end

  // Shift register update and pipelines
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // Reset shift register and pipelines
      for (i = 0; i < COEFF_COUNT; i = i + 1)
        sample_sr[i] <= '0;

      for (i = 0; i < PIPELINE_DEPTH; i = i + 1) begin
        acc_pipe[i]   <= '0;
        valid_pipe[i] <= 1'b0;
      end

      inflight_count <= '0;
    end else begin
      // Accept new input when in_valid & in_ready
      logic accept = in_valid & in_ready;

      if (accept) begin
        // Shift samples toward higher indices: sample_sr[COEFF_COUNT-1] <= ... <= sample_sr[0]
        // New sample goes into sample_sr[0]
        for (i = COEFF_COUNT-1; i > 0; i = i - 1)
          sample_sr[i] <= sample_sr[i-1];
        sample_sr[0] <= in_data;
      end

      // Pipeline the combinational accumulator into acc_pipe
      // We register comb_acc every cycle so that each accepted sample will propagate
      // through the pipeline stages and emerge after PIPELINE_DEPTH cycles.
      acc_pipe[0] <= comb_acc;
      valid_pipe[0] <= accept;
      for (i = 1; i < PIPELINE_DEPTH; i = i + 1) begin
        acc_pipe[i]   <= acc_pipe[i-1];
        valid_pipe[i] <= valid_pipe[i-1];
      end

      // Update inflight_count: increment on accept, decrement on successful output handshake
      logic out_consumed = valid_pipe[PIPELINE_DEPTH-1] & out_ready;

      // Note: perform arithmetic updates in non-blocking style
      if (accept && !out_consumed)
        inflight_count <= inflight_count + 1;
      else if (!accept && out_consumed)
        inflight_count <= inflight_count - 1;
      else if (accept && out_consumed)
        inflight_count <= inflight_count; // +1 -1 => unchanged
      else
        inflight_count <= inflight_count; // no change
    end
  end

  // Output generation: take last pipeline stage accumulator, scale from Q12 back to Q6
  // Then apply ReLU and saturate to DATA_WIDTH signed range
  acc_t acc_final;
  logic signed [DATA_WIDTH-1:0] scaled_out;
  logic out_valid_reg;

  always_comb begin
    acc_final = acc_pipe[PIPELINE_DEPTH-1];

    // Arithmetic right shift by COEFF_FRAC to convert Q12 -> Q6
    // Use signed shift
    logic signed [ACC_WIDTH-1:0] shifted;
    shifted = acc_final >>> COEFF_FRAC; // preserves sign (arithmetic shift)

    // Apply ReLU: if negative, output zero
    if (shifted < 0)
      scaled_out = '0;
    else begin
      // Saturate to DATA_WIDTH signed range
      logic signed [DATA_WIDTH-1:0] sat_max;
      logic signed [DATA_WIDTH-1:0] sat_min;
      sat_max = (1 <<< (DATA_WIDTH-1)) - 1;
      sat_min = - (1 <<< (DATA_WIDTH-1));

      if (shifted > sat_max)
        scaled_out = sat_max;
      else if (shifted < sat_min)
        scaled_out = sat_min;
      else
        scaled_out = shifted[DATA_WIDTH-1:0];
    end

    out_valid_reg = valid_pipe[PIPELINE_DEPTH-1];
  end

  // Drive outputs (registered behavior: valid holds until handshake)
  assign out_valid = out_valid_reg;
  assign out_data  = scaled_out;

endmodule : algorithm_core
