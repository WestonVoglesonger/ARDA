`timescale 1ns/1ps

module fft_twiddle_rom (
  input  logic        clk,
  input  logic [2:0]  stage_idx,
  input  logic [7:0]  addr,
  output logic [31:0] twiddle
);
  import fft_params_pkg::*;

  // Simple ROM implementation: for resource friendliness we store a limited set
  // and repeat. In real implementation this would be precomputed full twiddle table.
  logic signed [15:0] rom_re [0:255];
  logic signed [15:0] rom_im [0:255];

  // Initialize ROM with a simple pattern derived from stage and addr so simulation
  // and synthesis have deterministic content. In practice, replace with precomputed
  // twiddle factors (COEFF_WIDTH/COEFF_FRAC fixed-point values).
  integer i;
  initial begin
    for (i = 0; i < N; i = i + 1) begin
      // very coarse approximation: cosine-like index mapping
      rom_re[i] = (16'sd8192) - ((i * 3) & 16'h7FFF); // placeholder
      rom_im[i] = ((i * 7) & 16'h7FFF) - 16'sd4096;   // placeholder
    end
  end

  // Registered output for timing
  logic [15:0] out_re_r, out_im_r;
  always_ff @(posedge clk) begin
    out_re_r <= rom_re[addr];
    out_im_r <= rom_im[addr];
  end

  assign twiddle = {out_re_r, out_im_r};

endmodule
