`timescale 1ns/1ps
`include "complex_adaptive_kalman_params.svh"

module nonlinear_transform (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     in_valid,
    input  logic                     in_ready,
    input  logic signed [ACC_WIDTH-1:0] in_sample, // Q16.16
    input  logic [1:0]               transform_sel,

    output logic signed [FXP_WIDTH-1:0] out_sample, // Q8.8
    output logic                     out_valid,
    output logic                     out_ready
);

// Simple piecewise approximations for sigmoid/tanh/relu using shifts and saturations
logic signed [ACC_WIDTH-1:0] stage_reg;
logic                        stage_v;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        stage_reg <= '0;
        stage_v <= 1'b0;
        out_valid <= 1'b0;
        out_ready <= 1'b1;
        out_sample <= '0;
    end else begin
        if (in_valid & in_ready) begin
            stage_reg <= in_sample;
            stage_v <= 1'b1;
        end else begin
            stage_v <= 1'b0;
        end

        if (stage_v) begin
            // normalize in_sample Q16.16 to Q8.8 by shifting right by 8 bits
            logic signed [FXP_WIDTH+16-1:0] scaled;
            scaled = stage_reg >>> FXP_FRAC; // approx scale
            case (transform_sel)
                2'b00: begin // none
                    out_sample <= scaled[FXP_WIDTH-1:0];
                end
                2'b01: begin // sigmoid approximate: y = x/(1+|x|) mapped to Q8.8
                    logic signed [ACC_WIDTH-1:0] ax;
                    ax = (scaled[FXP_WIDTH-1]) ? -scaled : scaled;
                    logic signed [ACC_WIDTH-1:0] denom;
                    denom = ({{(ACC_WIDTH-FXP_WIDTH){1'b0}}, ax}) + (1 <<< FXP_FRAC);
                    if (denom == 0) out_sample <= scaled[FXP_WIDTH-1:0];
                    else out_sample <= ($signed({{(ACC_WIDTH-FXP_WIDTH){scaled[FXP_WIDTH-1]}}, scaled}) / denom)[FXP_WIDTH-1:0];
                end
                2'b10: begin // tanh approximate: clip and scale
                    // simple piecewise: tanh(x) ~ x for small, +/-1 for large
                    if (scaled > (16'sd4 << FXP_FRAC)) out_sample <= 16'sd(1 << FXP_FRAC);
                    else if (scaled < -(16'sd4 << FXP_FRAC)) out_sample <= -16'sd(1 << FXP_FRAC);
                    else out_sample <= scaled[FXP_WIDTH-1:0];
                end
                2'b11: begin // relu
                    if (scaled[FXP_WIDTH-1]) out_sample <= '0; else out_sample <= scaled[FXP_WIDTH-1:0];
                end
                default: out_sample <= scaled[FXP_WIDTH-1:0];
            endcase
            out_valid <= 1'b1;
        end else begin
            if (out_valid & out_ready) out_valid <= 1'b0;
        end

        // simple ready flow
        if (!out_valid) out_ready <= 1'b1; else out_ready <= 1'b0;
    end
end

endmodule
