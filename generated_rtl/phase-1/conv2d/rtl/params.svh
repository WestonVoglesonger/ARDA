package params_pkg;

  // Fixed-point and architecture parameters
  parameter int COEFF_COUNT    = 16;
  parameter int COEFF_WIDTH    = 8;   // bits for coefficients (signed)
  parameter int DATA_WIDTH     = 8;   // input/output data width (signed)
  parameter int ACC_WIDTH      = 16;  // accumulator width
  parameter int COEFF_FRAC     = 6;   // fractional bits for coefficients
  parameter int DATA_FRAC      = 6;   // fractional bits for input/output
  parameter int PIPELINE_DEPTH = 8;   // pipeline depth from microarch

  // Typedefs for convenience
  typedef logic signed [DATA_WIDTH-1:0] fxp_t;
  typedef logic signed [COEFF_WIDTH-1:0] coeff_t;
  typedef logic signed [ACC_WIDTH-1:0] acc_t;

  // Quantized coefficients (Q6 fixed-point). These are the provided quantized
  // coefficients scaled by 2^6 = 64 and represented as signed 8-bit values.
  // Index 0 => earliest tap.
  parameter logic signed [COEFF_WIDTH-1:0] COEFFS [0:COEFF_COUNT-1] = '{
    8'sd-32, // -0.5  * 64
    8'sd-16, // -0.25 * 64
    8'sd-8,  // -0.125* 64
    8'sd-4,  // -0.0625*64
    8'sd0,   // 0.0
    8'sd4,   // 0.0625*64
    8'sd8,   // 0.125*64
    8'sd16,  // 0.25*64
    8'sd32,  // 0.5*64
    8'sd0,   // 0.0
    8'sd-8,  // -0.125*64
    8'sd8,   // 0.125*64
    8'sd-16, // -0.25*64
    8'sd0,
    8'sd0,
    8'sd0
  };

endpackage : params_pkg
