`timescale 1ns/1ps
`include "conv2d_params.svh"

module conv2d_activation (
    input  logic [NUM_OUT_CHANNELS-1:0][ACC_WIDTH-1:0] in_px_vec,
    input  logic                      in_valid,
    input  logic                      clk,

    output logic [NUM_OUT_CHANNELS-1:0][7:0] out_px_vec_q,
    output logic                      out_valid
);

// Apply ReLU and clamp to INT8. Simple one-cycle pipeline for activation.
always_ff @(posedge clk) begin
    if (in_valid) begin
        for (int i=0;i<NUM_OUT_CHANNELS;i++) begin
            logic signed [ACC_WIDTH-1:0] val = in_px_vec[i];
            if (val < 0)
                out_px_vec_q[i] <= 8'sd0;
            else begin
                // clamp to int8 range
                if (val > 127)
                    out_px_vec_q[i] <= 8'sd127;
                else
                    out_px_vec_q[i] <= val[7:0];
            end
        end
        out_valid <= 1'b1;
    end else begin
        out_valid <= 1'b0;
    end
end

endmodule
