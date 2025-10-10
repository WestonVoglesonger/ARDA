package fft_params_pkg;
  // Global FFT parameters shared across modules
  parameter int N = 256;
  parameter int STAGES = 8;
  parameter int DATA_WIDTH = 16;
  parameter int FRAC_BITS = 12;
  parameter int COEFF_WIDTH = 16;
  parameter int COEFF_FRAC = 13;
  parameter int OUTPUT_WIDTH = 18;
  parameter int OUTPUT_FRAC = 14;
  parameter int ACCUM_WIDTH = 32;
  parameter int PIPELINE_DEPTH = 8;
endpackage : fft_params_pkg
