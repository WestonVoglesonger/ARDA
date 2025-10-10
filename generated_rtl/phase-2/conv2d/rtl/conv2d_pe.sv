`timescale 1ns/1ps
`include "conv2d_params.svh"

module conv2d_pe #(
    parameter int ACCW = ACC_WIDTH
) (
    input  logic [CHANNELS-1:0][WINDOW_ELEMS-1:0][PIXEL_WIDTH-1:0] window_per_channel,
    input  logic [WEIGHTS_PER_CH_W-1:0] weights, // 27 * 8
    input  logic [BIAS_WIDTH-1:0]       bias,
    input  logic                        start,
    input  logic                        clk,

    output logic signed [ACCW-1:0]      out_px,
    output logic                        out_valid
);

// Sequential MAC across WEIGHT_ELEMS (27 multiplies). Start triggers computation which takes WEIGHT_ELEMS cycles.
logic [$clog2(WEIGHT_ELEMS+1)-1:0] elem_cnt;
logic signed [ACCW-1:0] accumulator;
logic computing;

// Unpack weights into array of signed bytes
logic signed [7:0] w_arr [WEIGHT_ELEMS-1:0];

always_comb begin
    for (int i=0;i<WEIGHT_ELEMS;i++) begin
        int offset = (WEIGHT_ELEMS - 1 - i)*8;
        w_arr[i] = weights[offset +: 8];
    end
end

// Provide flat access to input pixels (signed)
logic signed [7:0] in_arr [WEIGHT_ELEMS-1:0];
always_comb begin
    int idx = 0;
    for (int ch=0; ch<CHANNELS; ch++) begin
        for (int e=0; e<WINDOW_ELEMS; e++) begin
            in_arr[idx] = window_per_channel[ch][e];
            idx++;
        end
    end
end

// Compute pipeline
always_ff @(posedge clk) begin
    if (!clk) begin end // keep synthesis happy
end

always_ff @(posedge clk) begin
    if (start) begin
        computing <= 1'b1;
        elem_cnt <= 0;
        accumulator <= $signed(bias); // bias assumed signed 16-bit
        out_valid <= 1'b0;
    end else if (computing) begin
        // multiply-accumulate
        // extend operands to accumulator width
        logic signed [ACCW-1:0] mul_tmp;
        mul_tmp = $signed(in_arr[elem_cnt]) * $signed(w_arr[elem_cnt]);
        accumulator <= accumulator + mul_tmp;

        if (elem_cnt == WEIGHT_ELEMS-1) begin
            computing <= 1'b0;
            out_px <= accumulator;
            out_valid <= 1'b1;
        end else begin
            elem_cnt <= elem_cnt + 1;
            out_valid <= 1'b0;
        end
    end else begin
        out_valid <= 1'b0;
    end
end

endmodule
