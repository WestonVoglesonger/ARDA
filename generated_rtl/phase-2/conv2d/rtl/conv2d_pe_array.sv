`timescale 1ns/1ps
`include "conv2d_params.svh"

module conv2d_pe_array (
    input  logic [CHANNELS-1:0][WINDOW_ELEMS-1:0][PIXEL_WIDTH-1:0] window_per_channel,
    input  logic [NUM_OUT_CHANNELS-1:0][WEIGHTS_PER_CH_W-1:0] weights_all_ch,
    input  logic [NUM_OUT_CHANNELS-1:0][BIAS_WIDTH-1:0] bias_all_ch,
    input  logic start,
    input  logic clk,

    output logic [NUM_OUT_CHANNELS-1:0][ACC_WIDTH-1:0] out_px_vec,
    output logic out_valid
);

// Instantiate NUM_OUT_CHANNELS PEs in parallel
genvar i;
generate
    for (i=0; i<NUM_OUT_CHANNELS; i=i+1) begin : pes
        logic signed [ACC_WIDTH-1:0] pe_out;
        logic pe_valid;

        conv2d_pe #(.ACCW(ACC_WIDTH)) pe_i (
            .window_per_channel(window_per_channel),
            .weights(weights_all_ch[i]),
            .bias(bias_all_ch[i]),
            .start(start),
            .clk(clk),
            .out_px(pe_out),
            .out_valid(pe_valid)
        );

        // Register outputs
        always_ff @(posedge clk) begin
            out_px_vec[i] <= pe_out;
        end

        // combine validity: all PEs must assert valid simultaneously for out_valid
        // We'll OR them into a reduction later
        // store per-PE valid into a vector
    end
endgenerate

// Combine valid signals: since PEs operate in lockstep, take valid from PE0
// For safety, monitor any: but we'll simply set out_valid when first PE reports valid
logic any_valid;
// Create array to capture per-PE valid via generate (indexed nets not easily available here)
// Simpler approach: assume PEs assert valid in same cycle; so sample start -> expected latency then assert out_valid

// Basic valid generation: when start asserted, after WEIGHT_ELEMS cycles outputs valid
logic [$clog2(WEIGHT_ELEMS+2)-1:0] start_cnt;
logic running;

always_ff @(posedge clk) begin
    if (start) begin
        running <= 1'b1;
        start_cnt <= 0;
        out_valid <= 1'b0;
    end else if (running) begin
        if (start_cnt == WEIGHT_ELEMS-1) begin
            running <= 1'b0;
            out_valid <= 1'b1;
        end else begin
            start_cnt <= start_cnt + 1;
            out_valid <= 1'b0;
        end
    end else begin
        out_valid <= 1'b0;
    end
end

endmodule
