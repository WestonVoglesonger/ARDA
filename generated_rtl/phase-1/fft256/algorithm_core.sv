/* algorithm_core.sv - Streaming complex multiply + twiddle lookup core
   - Accepts interleaved signed real/imag input (DATA_WIDTH each)
   - Multiplies by twiddle factors from TWIDDLE_ROM (coeff Q1.13)
   - Produces signed output real/imag (OUTPUT_WIDTH each)
   - Ready/valid handshake (streaming)
   - Internally uses a small FIFO of depth PIPELINE_DEPTH for elastic buffering
*/

`timescale 1ns/1ps
`include "params.svh"

module algorithm_core (
  input  logic            clk,
  input  logic            rst_n,

  // ready/valid streaming input: interleaved signed real/imag
  input  logic            in_valid,
  output logic            in_ready,
  input  logic signed [DATA_WIDTH*2-1:0] in_data, // {real[DATA_WIDTH-1:0], imag[DATA_WIDTH-1:0]}

  // ready/valid streaming output: interleaved signed real/imag
  output logic            out_valid,
  input  logic            out_ready,
  output logic signed [OUTPUT_WIDTH*2-1:0] out_data
);

  // FIFO pointers and counters (power-of-two depth assumed for simplicity)
  localparam int DEPTH = PIPELINE_DEPTH;
  localparam int PTR_BITS = $clog2(DEPTH);

  logic [PTR_BITS-1:0] wr_ptr, rd_ptr;
  logic [PTR_BITS:0]   count; // up to DEPTH

  // FIFO storage for outputs (post-multiply/butterfly results)
  logic signed [OUTPUT_WIDTH-1:0] fifo_real [0:DEPTH-1];
  logic signed [OUTPUT_WIDTH-1:0] fifo_imag [0:DEPTH-1];

  // Twiddle selection counter (simple rotating index modulo COEFF_COUNT)
  logic [PTR_BITS-1:0] sample_counter;

  // Input unpack
  logic signed [DATA_WIDTH-1:0] in_real_s, in_imag_s;
  assign in_real_s = in_data[DATA_WIDTH*2-1 -: DATA_WIDTH];
  assign in_imag_s = in_data[DATA_WIDTH-1 -: DATA_WIDTH];

  // Select twiddle from ROM
  coeff_complex_t current_twiddle;
  always_comb begin
    // sample_counter mod COEFF_COUNT: sample_counter is larger width but we index TWIDDLE_ROM with low bits
    current_twiddle = TWIDDLE_ROM[sample_counter % COEFF_COUNT];
  end

  // Complex multiply (combinational) with fixed-point scaling
  // a + j b  (input)  Q1.DATA_FRAC
  // c + j d  (coeff)  Q1.COEFF_FRAC
  // product fractional bits = DATA_FRAC + COEFF_FRAC
  // We will shift right by COEFF_FRAC to bring result back to DATA_FRAC alignment, then
  // convert to OUTPUT_FRAC by left-shifting (OUTPUT_FRAC - DATA_FRAC) if positive.

  logic signed [ACC_WIDTH-1:0] mul_ar; // a * c
  logic signed [ACC_WIDTH-1:0] mul_bi; // b * d
  logic signed [ACC_WIDTH-1:0] mul_ai; // a * d
  logic signed [ACC_WIDTH-1:0] mul_br; // b * c

  // product after shifting by COEFF_FRAC -> aligned to DATA_FRAC
  logic signed [ACC_WIDTH-1:0] prod_real_aligned;
  logic signed [ACC_WIDTH-1:0] prod_imag_aligned;

  always_comb begin
    // sign-extend operands to matched widths implicitly by $signed
    mul_ar = $signed(in_real_s) * $signed(current_twiddle.real);
    mul_bi = $signed(in_imag_s) * $signed(current_twiddle.imag);
    mul_ai = $signed(in_real_s) * $signed(current_twiddle.imag);
    mul_br = $signed(in_imag_s) * $signed(current_twiddle.real);

    // (a*c - b*d), (a*d + b*c)
    // shift right arithmetically by COEFF_FRAC to move from (DATA_FRAC+COEFF_FRAC) to DATA_FRAC
    prod_real_aligned = (mul_ar - mul_bi) >>> COEFF_FRAC;
    prod_imag_aligned = (mul_ai + mul_br) >>> COEFF_FRAC;
  end

  // Convert aligned products (Q with DATA_FRAC fractional bits) to OUTPUT_FRAC
  // If OUTPUT_FRAC >= DATA_FRAC, left-shift; otherwise right-shift
  localparam int FRAC_DIFF = OUTPUT_FRAC - DATA_FRAC;
  logic signed [ACC_WIDTH-1:0] prod_real_out_scaled;
  logic signed [ACC_WIDTH-1:0] prod_imag_out_scaled;

  always_comb begin
    if (FRAC_DIFF >= 0) begin
      prod_real_out_scaled = prod_real_aligned <<< FRAC_DIFF;
      prod_imag_out_scaled = prod_imag_aligned <<< FRAC_DIFF;
    end else begin
      prod_real_out_scaled = prod_real_aligned >>> (-FRAC_DIFF);
      prod_imag_out_scaled = prod_imag_aligned >>> (-FRAC_DIFF);
    end
  end

  // Saturate to OUTPUT_WIDTH when storing into FIFO
  function automatic logic signed [OUTPUT_WIDTH-1:0] saturate_out(input logic signed [ACC_WIDTH-1:0] val);
    logic signed [OUTPUT_WIDTH-1:0] max_v;
    logic signed [OUTPUT_WIDTH-1:0] min_v;
    max_v = $signed({1'b0, {(OUTPUT_WIDTH-1){1'b1}}});
    min_v = $signed({1'b1, {(OUTPUT_WIDTH-1){1'b0}}});
    if (val > $signed({{(ACC_WIDTH-OUTPUT_WIDTH){val[ACC_WIDTH-1]}}, max_v}))
      saturate_out = max_v;
    else if (val < $signed({{(ACC_WIDTH-OUTPUT_WIDTH){val[ACC_WIDTH-1]}}, min_v}))
      saturate_out = min_v;
    else
      saturate_out = val[OUTPUT_WIDTH-1:0];
  endfunction

  // FIFO logic: simple circular buffer
  logic push, pop;
  assign push = in_valid && in_ready;
  assign pop  = out_valid && out_ready;

  // in_ready asserted when FIFO not full
  assign in_ready  = (count < DEPTH);
  assign out_valid = (count > 0);

  // combinational output from FIFO at rd_ptr
  always_comb begin
    if (count > 0) begin
      out_data = { fifo_real[rd_ptr], fifo_imag[rd_ptr] };
    end else begin
      out_data = '0;
    end
  end

  integer i;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_ptr <= '0;
      rd_ptr <= '0;
      count  <= '0;
      sample_counter <= '0;
      for (i=0;i<DEPTH;i=i+1) begin
        fifo_real[i] <= '0;
        fifo_imag[i] <= '0;
      end
    end else begin
      // Advance sample counter on accepted inputs
      if (push) begin
        sample_counter <= sample_counter + 1;
      end

      // Write into FIFO when push && space
      if (push && (count < DEPTH)) begin
        fifo_real[wr_ptr] <= saturate_out(prod_real_out_scaled);
        fifo_imag[wr_ptr] <= saturate_out(prod_imag_out_scaled);
        wr_ptr <= wr_ptr + 1;
        count <= count + 1;
      end

      // Pop when downstream accepts
      if (pop && (count > 0)) begin
        rd_ptr <= rd_ptr + 1;
        count <= count - 1;
      end

      // If push and pop simultaneously, pointers have both advanced and count net unchanged
    end
  end

endmodule
