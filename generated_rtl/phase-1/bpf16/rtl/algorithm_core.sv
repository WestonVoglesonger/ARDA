`include "params.svh"

module algorithm_core (
  input  logic                 clk,
  input  logic                 rst_n,

  // Ready/Valid streaming input (one sample in per accepted beat)
  input  logic                 s_valid,
  output logic                 s_ready,
  input  params_pkg::in_t      s_data,

  // Ready/Valid streaming output (one sample out per beat)
  output logic                 m_valid,
  input  logic                 m_ready,
  output params_pkg::out_t     m_data
);

  import params_pkg::*;

  // Simple back-pressure model: this core is fully pipelined and can accept
  // one sample per cycle. For simplicity we always present ready=1.
  // Downstream backpressure (m_ready) is not used to stall acceptance in
  // this simple streaming implementation. In practice this could be
  // augmented to track pipeline fullness and assert s_ready accordingly.
  assign s_ready = 1'b1;

  // Sample shift register (state). samples[0] is newest sample.
  logic signed [DATA_WIDTH-1:0] samples [0:NUM_TAPS-1];

  // Valid pipeline control: shift the input valid through the pipeline
  logic valid_pipe [0:PIPELINE_DEPTH];

  // --- Multiplier stage (combinational) producing products ---
  logic signed [PROD_WIDTH-1:0] prods_comb [0:NUM_TAPS-1];

  // Extended products into accumulator width for safe additions
  acc_t prod_ext_comb [0:NUM_TAPS-1];

  // Pipeline registers for the adder tree. We'll implement a 5-stage
  // pipelined adder-tree: products -> sum8 -> sum4 -> sum2 -> final
  // Each stage stores acc_t wide partial sums.
  acc_t stage_prod_reg   [0:NUM_TAPS-1]; // after product register
  acc_t stage_sum8_reg   [0:8-1];       // 8 sums
  acc_t stage_sum4_reg   [0:4-1];       // 4 sums
  acc_t stage_sum2_reg   [0:2-1];       // 2 sums
  acc_t stage_final_reg;                // final accumulated value

  // Compute combinational products based on current sample state and coeffs
  genvar i;
  generate
    for (i = 0; i < NUM_TAPS; i++) begin : PROD_COMB
      // Signed multiply (sample * coeff)
      // The multiplication yields PROD_WIDTH bits. Use explicit cast to signed.
      always_comb begin
        prods_comb[i] = $signed(samples[i]) * $signed(COEFF_ROM[i]);
        // sign-extend to accumulator width combinationally
        prod_ext_comb[i] = acc_t'({{(ACC_WIDTH-PROD_WIDTH){prods_comb[i][PROD_WIDTH-1]}}, prods_comb[i]});
      end
    end
  endgenerate

  // Sequential logic: shift samples on accepted input, register pipeline stages
  integer idx;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // reset samples
      for (idx = 0; idx < NUM_TAPS; idx = idx + 1) begin
        samples[idx] <= '0;
      end
      // reset pipeline valid signals
      for (idx = 0; idx <= PIPELINE_DEPTH; idx = idx + 1) begin
        valid_pipe[idx] <= 1'b0;
      end
      // clear pipeline registers
      for (idx = 0; idx < NUM_TAPS; idx = idx + 1) begin
        stage_prod_reg[idx] <= '0;
      end
      for (idx = 0; idx < 8; idx = idx + 1) begin
        stage_sum8_reg[idx] <= '0;
      end
      for (idx = 0; idx < 4; idx = idx + 1) begin
        stage_sum4_reg[idx] <= '0;
      end
      for (idx = 0; idx < 2; idx = idx + 1) begin
        stage_sum2_reg[idx] <= '0;
      end
      stage_final_reg <= '0;
    end else begin
      // Accept input sample when s_valid & s_ready
      if (s_valid && s_ready) begin
        // shift register: newest at samples[0]
        for (idx = NUM_TAPS-1; idx >= 1; idx = idx - 1) begin
          samples[idx] <= samples[idx-1];
        end
        samples[0] <= s_data;
      end

      // Valid pipeline shift (inject s_valid at stage 0)
      valid_pipe[0] <= s_valid & s_ready;
      for (idx = 0; idx < PIPELINE_DEPTH; idx = idx + 1) begin
        valid_pipe[idx+1] <= valid_pipe[idx];
      end

      // Pipeline stage 0 -> register extended products
      for (idx = 0; idx < NUM_TAPS; idx = idx + 1) begin
        stage_prod_reg[idx] <= prod_ext_comb[idx];
      end

      // Stage 1: sum pairs of stage_prod_reg (8 results)
      for (idx = 0; idx < 8; idx = idx + 1) begin
        stage_sum8_reg[idx] <= stage_prod_reg[2*idx] + stage_prod_reg[2*idx + 1];
      end

      // Stage 2: sum pairs of stage_sum8_reg (4 results)
      for (idx = 0; idx < 4; idx = idx + 1) begin
        stage_sum4_reg[idx] <= stage_sum8_reg[2*idx] + stage_sum8_reg[2*idx + 1];
      end

      // Stage 3: sum pairs of stage_sum4_reg (2 results)
      for (idx = 0; idx < 2; idx = idx + 1) begin
        stage_sum2_reg[idx] <= stage_sum4_reg[2*idx] + stage_sum4_reg[2*idx + 1];
      end

      // Stage 4: final sum
      stage_final_reg <= stage_sum2_reg[0] + stage_sum2_reg[1];

    end
  end

  // Output formatting: shift to align fractional bits and narrow to OUTPUT_WIDTH
  // Use arithmetic right shift to maintain sign. Then truncate to OUTPUT_WIDTH bits.
  logic signed [ACC_WIDTH-1:0] shifted_acc;
  always_comb begin
    // Arithmetic right shift by SHIFT_RIGHT (>=0 expected)
    if (SHIFT_RIGHT >= 0)
      shifted_acc = $signed(stage_final_reg) >>> SHIFT_RIGHT;
    else
      shifted_acc = $signed(stage_final_reg) <<< (-SHIFT_RIGHT);
  end

  // Drive output data and valid
  assign m_data  = params_pkg::out_t'(shifted_acc[OUTPUT_WIDTH-1:0]);
  assign m_valid = valid_pipe[PIPELINE_DEPTH];

endmodule : algorithm_core
