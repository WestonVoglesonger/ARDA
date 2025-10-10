`timescale 1ns/1ps
`include "conv2d_params.svh"

module conv2d_control_fsm (
    input  logic clk,
    input  logic rst_n,
    input  logic in_valid,
    input  logic out_ready,

    output logic lb_en,
    output logic [7:0] weight_rd_addr,
    output logic pe_start,
    output logic activation_en
);

// Very small FSM to orchestrate read of weights and start PEs when window available.
// Behavior: when in_valid (window ready), assert lb_en; after that generate pe_start for each output channel index sequentially.

typedef enum logic [1:0] {IDLE, LOAD_WINDOW, RUN_PES, WAIT_OUT} state_t;
state_t state, next_state;

logic [7:0] ch_idx;
logic [$clog2(WEIGHT_ELEMS+2)-1:0] pebias_wait;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        ch_idx <= 0;
        lb_en <= 1'b0;
        weight_rd_addr <= '0;
        pe_start <= 1'b0;
        activation_en <= 1'b0;
    end else begin
        state <= next_state;
        case (state)
            IDLE: begin
                lb_en <= 1'b0;
                pe_start <= 1'b0;
                activation_en <= 1'b0;
                ch_idx <= 0;
                weight_rd_addr <= 0;
            end
            LOAD_WINDOW: begin
                lb_en <= 1'b1;
            end
            RUN_PES: begin
                lb_en <= 1'b0;
                // present current channel weights and start
                weight_rd_addr <= ch_idx;
                pe_start <= 1'b1;
                activation_en <= 1'b0;
            end
            WAIT_OUT: begin
                pe_start <= 1'b0;
                activation_en <= 1'b1;
            end
        endcase
    end
end

// Next state logic
always_comb begin
    next_state = state;
    case (state)
        IDLE: if (in_valid) next_state = LOAD_WINDOW;
        LOAD_WINDOW: next_state = RUN_PES;
        RUN_PES: next_state = WAIT_OUT;
        WAIT_OUT: begin
            if (out_ready) begin
                if (ch_idx == NUM_OUT_CHANNELS-1) next_state = IDLE; else next_state = RUN_PES;
            end
        end
    endcase
end

// Channel index update
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) ch_idx <= 0;
    else begin
        if (state == WAIT_OUT && out_ready) begin
            if (ch_idx == NUM_OUT_CHANNELS-1) ch_idx <= 0; else ch_idx <= ch_idx + 1;
        end
    end
end

endmodule
