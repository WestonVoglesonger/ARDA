package params_pkg;

  // Fixed-point / structural parameters (tweak as needed)
  parameter int COEFF_WIDTH    = 16; // coefficient bit width (signed)
  parameter int DATA_WIDTH     = 12; // input sample width (signed)
  parameter int OUTPUT_WIDTH   = 16; // output sample width (signed)
  parameter int PIPELINE_DEPTH = 5;  // pipeline stages through the compute tree
  parameter int NUM_TAPS       = 16; // number of FIR taps

  // Fractional / accumulator sizing (from quantized config)
  parameter int INPUT_FRAC     = 11; // input fractional bits
  parameter int COEFF_FRAC     = 14; // coefficient fractional bits
  parameter int OUTPUT_FRAC    = 14; // output fractional bits
  parameter int ACC_WIDTH      = 32; // accumulator width

  // Derived widths
  localparam int PROD_WIDTH = DATA_WIDTH + COEFF_WIDTH; // product width
  localparam int SHIFT_RIGHT = (INPUT_FRAC + COEFF_FRAC) - OUTPUT_FRAC; // bits to shift after accumulation

  // Typedefs for clarity
  typedef logic signed [DATA_WIDTH-1:0]    in_t;
  typedef logic signed [COEFF_WIDTH-1:0]  coeff_t;
  typedef logic signed [PROD_WIDTH-1:0]   prod_t;
  typedef logic signed [ACC_WIDTH-1:0]    acc_t;
  typedef logic signed [OUTPUT_WIDTH-1:0] out_t;

  // Quantized coefficient ROM (values taken from quantized_coefficients)
  // These are assumed to be integer representations in Q14 (coeff_frac = 14)
  localparam coeff_t COEFF_ROM [0:NUM_TAPS-1] = '{
    16'sd-116,
    16'sd-226,
    16'sd-179,
    16'sd184,
    16'sd845,
    16'sd1594,
    16'sd2110,
    16'sd2177,
    16'sd1710,
    16'sd790,
    16'sd-261,
    16'sd-1153,
    16'sd-1638,
    16'sd-1589,
    16'sd-1014,
    16'sd-28
  };

endpackage : params_pkg
