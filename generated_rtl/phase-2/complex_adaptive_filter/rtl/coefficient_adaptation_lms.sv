`timescale 1ns/1ps
`include "complex_adaptive_kalman_params.svh"

module coefficient_adaptation_lms (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic [TAP_BUS_WIDTH-1:0] input_buffer, // taps
    input  logic [TAP_BUS_WIDTH-1:0] coeff_in,
    input  logic signed [FXP_WIDTH-1:0] error_in,
    input  logic                     adapt_en,
    input  logic signed [FXP_WIDTH-1:0] learning_rate, // Q8.8
    input  logic signed [FXP_WIDTH-1:0] momentum,
    input  logic signed [FXP_WIDTH-1:0] threshold,

    output logic [TAP_BUS_WIDTH-1:0] coeff_out,
    output logic                     adaptation_active
);

// Unpack taps and coeffs
logic signed [FXP_WIDTH-1:0] taps [0:FILTER_LENGTH-1];
logic signed [FXP_WIDTH-1:0] coeffs [0:FILTER_LENGTH-1];
integer i;
always_comb begin
    for (i=0;i<FILTER_LENGTH;i=i+1) begin
        taps[i]   = input_buffer[(i+1)*FXP_WIDTH-1 -: FXP_WIDTH];
        coeffs[i] = coeff_in[(i+1)*FXP_WIDTH-1 -: FXP_WIDTH];
    end
end

// Simple memory for previous coefficients for momentum
logic signed [FXP_WIDTH-1:0] prev_coeffs [0:FILTER_LENGTH-1];

// Adaptation: coeff += lr * error * tap + momentum*(coeff - prev)
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i=0;i<FILTER_LENGTH;i=i+1) begin
            coeff_out[(i+1)*FXP_WIDTH-1 -: FXP_WIDTH] <= DEFAULT_COEFF;
            prev_coeffs[i] <= DEFAULT_COEFF;
        end
        adaptation_active <= 1'b0;
    end else begin
        // determine if active
        adaptation_active <= (adapt_en & ((error_in > threshold) | (error_in < -threshold)));
        if (adaptation_active) begin
            for (i=0;i<FILTER_LENGTH;i=i+1) begin
                // gradient = -error * tap
                logic signed [ACC_WIDTH-1:0] grad;
                grad = -($signed({{(ACC_WIDTH-FXP_WIDTH){error_in[FXP_WIDTH-1]}}, error_in}) *
                         $signed({{(ACC_WIDTH-FXP_WIDTH){taps[i][FXP_WIDTH-1]}}, taps[i]}));
                // scale by learning rate (Q8.8) -> shift back by FXP_FRAC
                logic signed [ACC_WIDTH-1:0] delta;
                delta = (grad * $signed({{(ACC_WIDTH-FXP_WIDTH){learning_rate[FXP_WIDTH-1]}}, learning_rate})) >>> FXP_FRAC;
                // momentum term
                logic signed [ACC_WIDTH-1:0] mom;
                mom = ($signed({{(ACC_WIDTH-FXP_WIDTH){momentum[FXP_WIDTH-1]}}, momentum}) *
                       ($signed({{(ACC_WIDTH-FXP_WIDTH){coeffs[i][FXP_WIDTH-1]}}, coeffs[i]}) -
                        $signed({{(ACC_WIDTH-FXP_WIDTH){prev_coeffs[i][FXP_WIDTH-1]}}, prev_coeffs[i]}))) >>> FXP_FRAC;

                logic signed [ACC_WIDTH-1:0] new_coeff_ext;
                new_coeff_ext = $signed({{(ACC_WIDTH-FXP_WIDTH){coeffs[i][FXP_WIDTH-1]}}, coeffs[i]}) + delta + mom;
                // clip to FXP width
                logic signed [FXP_WIDTH-1:0] new_coeff_q;
                if (new_coeff_ext > $signed({1'b0,{FXP_WIDTH-1{1'b1}}})) new_coeff_q = $signed({1'b0,{FXP_WIDTH-1{1'b1}}});
                else if (new_coeff_ext < -($signed({1'b0,{FXP_WIDTH-1{1'b1}}}))) new_coeff_q = -$signed({1'b0,{FXP_WIDTH-1{1'b1}}});
                else new_coeff_q = new_coeff_ext[FXP_WIDTH-1:0];

                coeff_out[(i+1)*FXP_WIDTH-1 -: FXP_WIDTH] <= new_coeff_q;
                prev_coeffs[i] <= coeffs[i];
            end
        end else begin
            // passthrough coefficients
            for (i=0;i<FILTER_LENGTH;i=i+1) coeff_out[(i+1)*FXP_WIDTH-1 -: FXP_WIDTH] <= coeffs[i];
        end
    end
end

endmodule
