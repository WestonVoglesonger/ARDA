// fir_adder_tree.sv
// Hierarchical pipelined adder tree to reduce 16 partial products into a 32-bit sum.
// The interface expects 16 products_in each PROD_WIDTH bits (28 here) concatenated.

`include "fir_params.svh"

module fir_adder_tree (
    input  logic                 clk,
    input  logic                 rst,
    input  logic signed [16*(IN_WIDTH+COEFF_WIDTH)-1:0] products_in, // 16 x PROD_WIDTH
    input  logic                 in_valid,

    output logic signed [ACC_WIDTH-1:0] sum_out,
    output logic                 out_valid
);

    localparam int PROD_WIDTH = IN_WIDTH + COEFF_WIDTH; // 28
    // Unpack
    logic signed [PROD_WIDTH-1:0] p [0:15];
    integer i;
    always_comb begin
        for (i = 0; i < 16; i = i + 1) begin
            p[i] = products_in[(16-i)*PROD_WIDTH-1 -: PROD_WIDTH];
        end
    end

    // Simple 4-stage adder tree with registers between stages (pipelined)
    logic signed [PROD_WIDTH:0] s1 [0:7]; // stage1 sums (one extra bit for carry)
    logic signed [PROD_WIDTH+1:0] s2 [0:3];
    logic signed [PROD_WIDTH+2:0] s3 [0:1];
    logic signed [PROD_WIDTH+3:0] s4; // final before sign-extend to ACC_WIDTH

    logic signed [PROD_WIDTH:0] s1_r [0:7];
    logic signed [PROD_WIDTH+1:0] s2_r [0:3];
    logic signed [PROD_WIDTH+2:0] s3_r [0:1];
    logic signed [PROD_WIDTH+3:0] s4_r;

    always_comb begin
        for (i = 0; i < 8; i = i + 1) begin
            s1[i] = $signed(p[2*i]) + $signed(p[2*i+1]);
        end
        for (i = 0; i < 4; i = i + 1) begin
            s2[i] = s1[2*i] + s1[2*i+1];
        end
        for (i = 0; i < 2; i = i + 1) begin
            s3[i] = s2[2*i] + s2[2*i+1];
        end
        s4 = s3[0] + s3[1];
    end

    // Pipeline registers (one cycle each stage)
    always_ff @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < 8; i++) s1_r[i] <= '0;
            for (i = 0; i < 4; i++) s2_r[i] <= '0;
            for (i = 0; i < 2; i++) s3_r[i] <= '0;
            s4_r <= '0;
            sum_out <= '0;
            out_valid <= 1'b0;
        end else begin
            // stage 1 reg
            for (i = 0; i < 8; i++) s1_r[i] <= s1[i];
            // stage 2 reg
            for (i = 0; i < 4; i++) s2_r[i] <= s2[i];
            // stage 3 reg
            for (i = 0; i < 2; i++) s3_r[i] <= s3[i];
            // stage 4 reg
            s4_r <= s4;

            // sign-extend s4_r to accumulator width
            sum_out <= $signed({{(ACC_WIDTH-(PROD_WIDTH+4)){s4_r[PROD_WIDTH+3]}}, s4_r});
            // Valid simply follow in_valid delayed 3 cycles (since we have pipeline regs above)
            out_valid <= in_valid; // note: caller/top should align valid timing if necessary
        end
    end

endmodule
