`timescale 1ns/1ps
`include "complex_adaptive_kalman_params.svh"

module adaptive_fir_mac_pipeline (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic [TAP_BUS_WIDTH-1:0] taps_in,   // packed [tap0,tap1,...]
    input  logic [TAP_BUS_WIDTH-1:0] coeff_in,  // packed
    input  logic                     in_valid,
    input  logic                     in_ready,

    output logic signed [ACC_WIDTH-1:0] fir_out, // Q16.16 accumulator
    output logic                     out_valid,
    output logic                     out_ready
);

// Unpack taps and coeffs into arrays
logic signed [FXP_WIDTH-1:0] taps [0:FILTER_LENGTH-1];
logic signed [FXP_WIDTH-1:0] coeffs [0:FILTER_LENGTH-1];
integer i;

always_comb begin
    for (i=0;i<FILTER_LENGTH;i=i+1) begin
        taps[i]   = taps_in[(i+1)*FXP_WIDTH-1 -: FXP_WIDTH];
        coeffs[i] = coeff_in[(i+1)*FXP_WIDTH-1 -: FXP_WIDTH];
    end
end

// Stage 1: parallel products (width FXP_WIDTH+FXP_WIDTH)
logic signed [ACC_WIDTH-1:0] products [0:FILTER_LENGTH-1];

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i=0;i<FILTER_LENGTH;i=i+1) products[i] <= '0;
        out_valid <= 1'b0;
        out_ready <= 1'b1;
    end else begin
        if (in_valid & in_ready) begin
            for (i=0;i<FILTER_LENGTH;i=i+1) begin
                // extend to accumulator width before multiply
                products[i] <= $signed({{(ACC_WIDTH-FXP_WIDTH){taps[i][FXP_WIDTH-1]}}, taps[i]}) *
                               $signed({{(ACC_WIDTH-FXP_WIDTH){coeffs[i][FXP_WIDTH-1]}}, coeffs[i]});
            end
        end
        // Pipeline the adder tree across several cycles to maintain 1-sample/cycle throughput.
        // Stage sums: we implement a simple balanced tree with two pipeline stages.
    end
end

// Adder tree stage 1 registers
logic signed [ACC_WIDTH-1:0] s1 [0:(FILTER_LENGTH/2)-1];
logic signed [ACC_WIDTH-1:0] s2 [0:(FILTER_LENGTH/4)-1];
logic signed [ACC_WIDTH-1:0] s3 [0:(FILTER_LENGTH/8)-1];
logic signed [ACC_WIDTH-1:0] s4;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i=0;i<FILTER_LENGTH/2;i=i+1) s1[i] <= '0;
        for (i=0;i<FILTER_LENGTH/4;i=i+1) s2[i] <= '0;
        for (i=0;i<FILTER_LENGTH/8;i=i+1) s3[i] <= '0;
        s4 <= '0;
        fir_out <= '0;
        out_valid <= 1'b0;
    end else begin
        // pairing products into s1
        for (i=0;i<FILTER_LENGTH/2;i=i+1) begin
            s1[i] <= products[2*i] + products[2*i+1];
        end
        // next stage
        for (i=0;i<FILTER_LENGTH/4;i=i+1) begin
            s2[i] <= s1[2*i] + s1[2*i+1];
        end
        // next
        for (i=0;i<FILTER_LENGTH/8;i=i+1) begin
            s3[i] <= s2[2*i] + s2[2*i+1];
        end
        // final accumulation (works for FILTER_LENGTH=32)
        s4 <= s3[0] + s3[1] + s3[2] + s3[3];

        // Output valid delayed by pipeline depth (here 3 cycles)
        // Use simple shift register for valid
        static logic [3:0] valid_shift;
        valid_shift <= {valid_shift[2:0], in_valid & in_ready};
        out_valid <= valid_shift[3];
        if (valid_shift[3]) fir_out <= s4;

        // simple out_ready behavior
        if (!out_valid) out_ready <= 1'b1; else out_ready <= 1'b0;
    end
end

endmodule
