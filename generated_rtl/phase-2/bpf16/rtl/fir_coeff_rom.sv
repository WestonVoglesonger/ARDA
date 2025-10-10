// fir_coeff_rom.sv
// Simple combinational coefficient ROM. Address -> coeff
// Coeff array initialized from quantized_coefficients in architecture JSON.

`include "fir_params.svh"

module fir_coeff_rom (
    input  logic [3:0]                       addr,
    output logic signed [COEFF_WIDTH-1:0]    coeff
);

    // Coefficient memory (indexed 0..15)
    // Values are the quantized coefficients (integers) provided in architecture JSON.
    // They are interpreted as signed fixed-point with COEFF_FRAC fractional bits.
    logic signed [COEFF_WIDTH-1:0] rom [0:N_TAPS-1];

    initial begin
        rom[0]  = $signed({{(COEFF_WIDTH-16){1'b0}}, 16'sd-116});
        rom[1]  = $signed({{(COEFF_WIDTH-16){1'b0}}, 16'sd-226});
        rom[2]  = $signed({{(COEFF_WIDTH-16){1'b0}}, 16'sd-179});
        rom[3]  = $signed({{(COEFF_WIDTH-16){1'b0}}, 16'sd184});
        rom[4]  = $signed({{(COEFF_WIDTH-16){1'b0}}, 16'sd845});
        rom[5]  = $signed({{(COEFF_WIDTH-16){1'b0}}, 16'sd1594});
        rom[6]  = $signed({{(COEFF_WIDTH-16){1'b0}}, 16'sd2110});
        rom[7]  = $signed({{(COEFF_WIDTH-16){1'b0}}, 16'sd2177});
        rom[8]  = $signed({{(COEFF_WIDTH-16){1'b0}}, 16'sd1710});
        rom[9]  = $signed({{(COEFF_WIDTH-16){1'b0}}, 16'sd790});
        rom[10] = $signed({{(COEFF_WIDTH-16){1'b0}}, 16'sd-261});
        rom[11] = $signed({{(COEFF_WIDTH-16){1'b0}}, 16'sd-1153});
        rom[12] = $signed({{(COEFF_WIDTH-16){1'b0}}, 16'sd-1638});
        rom[13] = $signed({{(COEFF_WIDTH-16){1'b0}}, 16'sd-1589});
        rom[14] = $signed({{(COEFF_WIDTH-16){1'b0}}, 16'sd-1014});
        rom[15] = $signed({{(COEFF_WIDTH-16){1'b0}}, 16'sd-28});
    end

    always_comb begin
        coeff = rom[addr];
    end

endmodule
