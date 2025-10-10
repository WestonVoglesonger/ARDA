`timescale 1ns/1ps
`include "complex_adaptive_kalman_params.svh"

module performance_metrics_unit (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic signed [FXP_WIDTH-1:0] current_output,
    input  logic signed [FXP_WIDTH-1:0] error_in,
    input  logic [TAP_BUS_WIDTH-1:0] coeff_in,

    output logic signed [FXP_WIDTH-1:0] snr_estimate,
    output logic signed [FXP_WIDTH-1:0] convergence_rate,
    output logic signed [FXP_WIDTH-1:0] stability_factor
);

// Simple running statistics: EMA for signal power and noise power
logic [FXP_WIDTH+8-1:0] signal_pow;
logic [FXP_WIDTH+8-1:0] noise_pow;
logic [15:0] sample_count;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        signal_pow <= '0;
        noise_pow <= '0;
        sample_count <= 0;
        snr_estimate <= '0;
        convergence_rate <= '0;
        stability_factor <= '0;
    end else begin
        // update EMAs
        logic signed [FXP_WIDTH-1:0] so;
        logic signed [FXP_WIDTH-1:0] ei;
        so = current_output;
        ei = error_in;
        // square approximations (extend and multiply)
        logic [ACC_WIDTH-1:0] s2, n2;
        s2 = $signed({{(ACC_WIDTH-FXP_WIDTH){so[FXP_WIDTH-1]}}, so}) * $signed({{(ACC_WIDTH-FXP_WIDTH){so[FXP_WIDTH-1]}}, so});
        n2 = $signed({{(ACC_WIDTH-FXP_WIDTH){ei[FXP_WIDTH-1]}}, ei}) * $signed({{(ACC_WIDTH-FXP_WIDTH){ei[FXP_WIDTH-1]}}, ei});
        // EMA alpha=1/64
        signal_pow <= ((signal_pow * 63) + (s2 >>> (2*FXP_FRAC - 8))) >>> 6; // scale down to fit
        noise_pow  <= ((noise_pow * 63) + (n2 >>> (2*FXP_FRAC - 8))) >>> 6;

        sample_count <= sample_count + 1;

        // snr estimate = 10*log10(signal/noise) approximated as ratio in Q8.8
        if (noise_pow != 0) begin
            logic [31:0] ratio;
            ratio = (signal_pow << 8) / noise_pow; // Q8.8 ratio
            snr_estimate <= (ratio > 16'h7FFF) ? 16'h7FFF : ratio[15:0];
        end else begin
            snr_estimate <= 16'sd(0);
        end

        // convergence_rate: difference of recent error magnitudes (approx)
        static logic signed [FXP_WIDTH-1:0] prev_error_abs;
        logic signed [FXP_WIDTH-1:0] err_abs;
        err_abs = error_in[FXP_WIDTH-1] ? -error_in : error_in;
        convergence_rate <= prev_error_abs - err_abs;
        prev_error_abs <= err_abs;

        // stability_factor: inverse of coefficient variation; simple approximation using first two coeffs
        logic signed [FXP_WIDTH-1:0] c0, c1;
        c0 = coeff_in[FXP_WIDTH-1 -: FXP_WIDTH];
        c1 = coeff_in[2*FXP_WIDTH-1 -: FXP_WIDTH];
        logic signed [FXP_WIDTH-1:0] diff;
        diff = c0 - c1;
        logic [FXP_WIDTH-1:0] absdiff;
        absdiff = diff[FXP_WIDTH-1] ? -diff : diff;
        if (absdiff == 0) stability_factor <= 16'sd(1 << FXP_FRAC);
        else stability_factor <= (16'sd(1 << FXP_FRAC) <<< FXP_FRAC) / absdiff;
    end
end

endmodule
