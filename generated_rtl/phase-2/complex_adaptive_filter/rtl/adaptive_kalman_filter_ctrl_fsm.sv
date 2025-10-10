`timescale 1ns/1ps
`include "complex_adaptive_kalman_params.svh"

module adaptive_kalman_filter_ctrl_fsm (
    input  logic             clk,
    input  logic             rst_n,
    input  logic             in_valid,
    input  logic             out_ready,
    input  logic signed [FXP_WIDTH-1:0] error_in,
    input  logic [15:0]      adaptation_counter,

    output logic [7:0]       stage_en,
    output logic             adapt_en,
    output logic             reset_pipeline
);

typedef enum logic [1:0] {IDLE=2'b00, RUN=2'b01, PAUSE=2'b10} fsm_t;
fsm_t state;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        stage_en <= 8'hFF; // enable all by default
        adapt_en <= 1'b0;
        reset_pipeline <= 1'b0;
    end else begin
        case (state)
            IDLE: begin
                reset_pipeline <= 1'b1;
                stage_en <= 8'h00;
                adapt_en <= 1'b0;
                if (in_valid) begin
                    state <= RUN;
                    reset_pipeline <= 1'b0;
                end
            end
            RUN: begin
                reset_pipeline <= 1'b0;
                stage_en <= 8'hFF;
                // trigger adaptation when error magnitude exceeds threshold
                if ((error_in > 16'sd(1 << (FXP_FRAC-1))) || (error_in < -(16'sd(1 << (FXP_FRAC-1))))) adapt_en <= 1'b1; else adapt_en <= 1'b0;
                if (!out_ready) state <= PAUSE;
            end
            PAUSE: begin
                // pause pipeline when downstream can't accept
                stage_en <= 8'h00;
                adapt_en <= 1'b0;
                if (out_ready) state <= RUN;
            end
            default: state <= IDLE;
        endcase
    end
end

endmodule
