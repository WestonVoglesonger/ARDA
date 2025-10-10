// fir_params.svh
// Global FIR parameters and fixed-point formats
`ifndef FIR_PARAMS_SVH
`define FIR_PARAMS_SVH

// Number of taps
parameter int N_TAPS     = 16;
// Coefficient fixed-point width and fractional bits
parameter int COEFF_WIDTH = 16;
parameter int COEFF_FRAC  = 14;
// Input sample fixed-point width and fractional bits
parameter int IN_WIDTH    = 12;
parameter int IN_FRAC     = 11;
// Output fixed-point width and fractional bits
parameter int OUT_WIDTH   = 16;
parameter int OUT_FRAC    = 14;
// Accumulator width (internal)
parameter int ACC_WIDTH   = 32;

`endif // FIR_PARAMS_SVH
