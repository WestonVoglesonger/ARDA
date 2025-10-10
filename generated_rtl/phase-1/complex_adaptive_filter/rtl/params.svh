package params_pkg;

// Fixed-point / width parameters (from quant.fixed_point_config)
parameter int COEFF_WIDTH      = 16; // width of coefficients (signed)
parameter int COEFF_FRAC       = 15; // fractional bits for coefficients
parameter int DATA_WIDTH       = 16; // input / output data width (signed)
parameter int DATA_FRAC        = 15; // fractional bits for data
parameter int ACC_WIDTH        = 32; // accumulator width for MAC
parameter int PIPELINE_DEPTH   = 8;  // pipeline depth from microarch
parameter int TAP_COUNT        = 16; // filter taps (based on quantized coeffs)

// Adaptation / algorithmic parameters
// Learning rate and adaptation threshold encoded in Q1.DATA_FRAC fixed point
parameter signed [DATA_WIDTH-1:0] LEARNING_RATE_Q = 16'sd164;   // ~0.005 (0.005*32768 ~= 164)
parameter signed [DATA_WIDTH-1:0] ADAPT_THRESHOLD_Q = 16'sd1638; // ~0.05 (0.05*32768 ~= 1638)

// Coefficient ROM initialization (quantized coefficients scaled to Q1.COEFF_FRAC)
// Original float coefficients (from quant.quantized_coefficients):
// [0.12, -0.08, 0.05, 0.0, -0.03, 0.07, -0.02, 0.0, 0.01, -0.04, 0.09, -0.01, 0.0, 0.0, -0.05, 0.03]
// Converted to fixed-point Q1.15 (rounded)
localparam signed [COEFF_WIDTH-1:0] COEFFS [0:TAP_COUNT-1] = '{
  16'sd3932,   // 0.12
  -16'sd2621,  // -0.08
  16'sd1638,   // 0.05
  16'sd0,      // 0.0
  -16'sd983,   // -0.03
  16'sd2294,   // 0.07
  -16'sd655,   // -0.02
  16'sd0,      // 0.0
  16'sd328,    // 0.01
  -16'sd1311,  // -0.04
  16'sd2949,   // 0.09
  -16'sd328,   // -0.01
  16'sd0,      // 0.0
  16'sd0,      // 0.0
  -16'sd1638,  // -0.05
  16'sd983     // 0.03
};

// Typedefs for clarity
typedef logic signed [DATA_WIDTH-1:0] fxp_t;     // Q1.DATA_FRAC fixed-point sample/data type
typedef logic signed [COEFF_WIDTH-1:0] coeff_t; // Q1.COEFF_FRAC fixed-point coefficient type
typedef logic signed [ACC_WIDTH-1:0] acc_t;     // Accumulator type for MAC (wide)

endpackage : params_pkg
