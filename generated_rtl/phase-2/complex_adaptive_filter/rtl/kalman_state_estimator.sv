`timescale 1ns/1ps
`include "complex_adaptive_kalman_params.svh"

module kalman_state_estimator (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic signed [FXP_WIDTH-1:0] measurement_in,
    input  logic signed [FXP_WIDTH-1:0] innovation_in,
    input  logic [STATE_BUS_WIDTH-1:0] state_vector_in,
    input  logic [STATE_BUS_WIDTH-1:0] state_cov_in,
    input  logic                     in_valid,
    input  logic                     in_ready,

    output logic [STATE_BUS_WIDTH-1:0] state_vector_out,
    output logic [STATE_BUS_WIDTH-1:0] state_cov_out,
    output logic                     out_valid,
    output logic                     out_ready
);

// Unpack state vectors
logic signed [FXP_WIDTH-1:0] state_in [0:STATE_DIM-1];
logic signed [FXP_WIDTH-1:0] cov_in [0:STATE_DIM-1];
integer j;
always_comb begin
    for (j=0;j<STATE_DIM;j=j+1) begin
        state_in[j] = state_vector_in[(j+1)*FXP_WIDTH-1 -: FXP_WIDTH];
        cov_in[j]   = state_cov_in[(j+1)*FXP_WIDTH-1 -: FXP_WIDTH];
    end
end

// Local registers for outputs
logic signed [FXP_WIDTH-1:0] state_reg [0:STATE_DIM-1];
logic signed [FXP_WIDTH-1:0] cov_reg [0:STATE_DIM-1];

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (j=0;j<STATE_DIM;j=j+1) begin
            state_reg[j] <= '0;
            cov_reg[j] <= '0;
        end
        out_valid <= 1'b0;
        out_ready <= 1'b1;
        state_vector_out <= '0;
        state_cov_out <= '0;
    end else begin
        if (in_valid & in_ready) begin
            // simplified Kalman-like update per element
            // compute scalar kalman_gain = cov / (cov + innovation_var)
            logic [FXP_WIDTH-1:0] innovation_var;
            logic signed [FXP_WIDTH-1:0] gain_q;
            // innovation_var approx = abs(innovation_in) + 1
            innovation_var = (innovation_in[FXP_WIDTH-1] ? -innovation_in : innovation_in) + (1 << (FXP_FRAC-1));
            // gain = cov_in[0] / (cov_in[0] + innovation_var)
            // Use Q8.8 division producing Q8.8
            if ((cov_in[0] + innovation_var) == 0) gain_q = 0;
            else gain_q = ($signed({{(ACC_WIDTH-FXP_WIDTH){cov_in[0][FXP_WIDTH-1]}}, cov_in[0]}) /
                              $signed({{(ACC_WIDTH-FXP_WIDTH){1'b0}}, (cov_in[0] + innovation_var)})) [FXP_WIDTH-1:0];

            // update state 0
            state_reg[0] <= state_in[0] + $signed((gain_q * innovation_in) >>> FXP_FRAC);
            // update covariance 0: cov*(1-gain)
            cov_reg[0] <= $signed((cov_in[0] * ($signed({1'b0,{FXP_WIDTH-1{1'b1}}}) - gain_q)) >>> FXP_FRAC);

            // propagate to other dimensions with simple coupling
            integer k;
            for (k=1;k<STATE_DIM;k=k+1) begin
                // coupling_factor approx fixed: 0.1, scaled to Q8.8
                logic signed [FXP_WIDTH-1:0] coupling_q;
                coupling_q = $signed(16'sd26); // approx 0.1 in Q8.8 (0.1016*256=26)
                state_reg[k] <= state_in[k] + $signed((coupling_q * (state_reg[0] - state_in[0])) >>> FXP_FRAC);
                cov_reg[k] <= cov_in[k] - ((cov_in[k] * coupling_q) >>> FXP_FRAC);
            end

            // pack outputs
            for (j=0;j<STATE_DIM;j=j+1) begin
                state_vector_out[(j+1)*FXP_WIDTH-1 -: FXP_WIDTH] <= state_reg[j];
                state_cov_out[(j+1)*FXP_WIDTH-1 -: FXP_WIDTH] <= cov_reg[j];
            end

            out_valid <= 1'b1;
        end else begin
            if (out_valid & out_ready) out_valid <= 1'b0;
        end

        if (!out_valid) out_ready <= 1'b1; else out_ready <= 1'b0;
    end
end

endmodule
