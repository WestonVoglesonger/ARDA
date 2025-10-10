`timescale 1ns/1ps
`include "complex_adaptive_kalman_params.svh"

module input_normalizer (
    input  logic                 clk,
    input  logic                 rst_n,
    input  logic                 in_valid,
    input  logic                 in_ready, // external ready input (per architecture)
    input  logic signed [FXP_WIDTH-1:0] in_sample,
    input  logic signed [FXP_WIDTH-1:0] error_history_in,

    output logic                 out_valid,
    output logic                 out_ready,
    output logic signed [FXP_WIDTH-1:0] normalized_sample
);

// Simple adaptive normalizer using running mean/variance (exponential)
logic signed [FXP_WIDTH-1:0] mean_q;
logic [FXP_WIDTH-1:0] var_q; // unsigned magnitude estimate
logic [7:0]                adapt_count;

// Internal handshake: if upstream presents valid and external in_ready asserted, capture
logic capture_en;
assign capture_en = in_valid & in_ready;

// Normalize pipeline register
logic signed [FXP_WIDTH-1:0] sample_reg;
logic                        sample_reg_v;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        mean_q <= '0;
        var_q  <= '0;
        adapt_count <= 0;
        sample_reg <= '0;
        sample_reg_v <= 1'b0;
        out_valid <= 1'b0;
        out_ready <= 1'b1; // default downstream ready
        normalized_sample <= '0;
    end else begin
        // update statistics when capturing
        if (capture_en) begin
            // simple exponential moving average for mean (alpha = 1/16)
            mean_q <= ((mean_q * 15) + in_sample) >>> 4;
            // update variance approximation = EMA of abs(sample-mean)
            logic signed [FXP_WIDTH-1:0] diff;
            logic [FXP_WIDTH-1:0] absdiff;
            diff = in_sample - mean_q;
            absdiff = diff[FXP_WIDTH-1] ? -diff : diff;
            var_q <= ((var_q * 15) + absdiff) >>> 4;
            adapt_count <= adapt_count + 1;
            sample_reg <= in_sample;
            sample_reg_v <= 1'b1;
        end else begin
            sample_reg_v <= 1'b0;
        end

        // Produce normalized sample one cycle after capture
        if (sample_reg_v) begin
            // If variance small, bypass normalization
            if (var_q == 0) begin
                normalized_sample <= sample_reg;
            end else begin
                // Normalize: out = (in - mean) / (var*2) approx by shift
                logic signed [FXP_WIDTH+8-1:0] centered;
                centered = (sample_reg - mean_q);
                // scale by 1/(var*2) approximate: divide by (var_q << 1)
                // implement as (centered << FXP_FRAC) / (var_q << 1)
                logic signed [ACC_WIDTH-1:0] numer;
                logic signed [ACC_WIDTH-1:0] denom;
                numer = $signed(centered) <<< FXP_FRAC;
                denom = $signed({8'b0,var_q}) <<< 1;
                if (denom == 0) begin
                    normalized_sample <= sample_reg;
                end else begin
                    // simple division using signed division (synthesizable by vendor tools)
                    logic signed [ACC_WIDTH-1:0] divv;
                    divv = numer / denom;
                    // saturate back to FXP_WIDTH
                    if (divv > $signed({1'b0, {FXP_WIDTH-1{1'b1}}})) begin
                        normalized_sample <= $signed({1'b0, {FXP_WIDTH-1{1'b1}}});
                    end else if (divv < -($signed({1'b0,{FXP_WIDTH-1{1'b1}}}))) begin
                        normalized_sample <= -$signed({1'b0,{FXP_WIDTH-1{1'b1}}});
                    end else begin
                        normalized_sample <= divv[FXP_WIDTH-1:0];
                    end
                end
            end
            out_valid <= 1'b1;
        end else begin
            // default valid handshake: remain asserted until downstream ready
            if (out_valid && out_ready) out_valid <= 1'b0;
        end

        // pass through downstream ready (simple flow control)
        // out_ready is an output; by default we assert it when not busy
        if (!out_valid) out_ready <= 1'b1; // ready to accept new
        else out_ready <= 1'b0;
    end
end

endmodule
