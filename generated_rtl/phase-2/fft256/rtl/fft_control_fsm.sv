`timescale 1ns/1ps

module fft_control_fsm (
  input  logic clk,
  input  logic rst_n,
  input  logic in_valid,
  input  logic in_ready,
  input  logic stage_done,
  input  logic mem_ready,
  input  logic start,

  output logic [2:0] stage_idx,
  output logic       stage_start,
  output logic       read_en,
  output logic       write_en,
  output logic       out_valid,
  output logic       done
);
  import fft_params_pkg::*;

  typedef enum logic [2:0] {IDLE=3'd0, LOAD=3'd1, PROCESS=3'd2, WAIT_STAGE=3'd3, OUTPUT=3'd4, FINISH=3'd5} state_t;
  state_t state, next_state;

  logic [2:0] stage_counter;

  // Simple FSM: on start, move through STAGES processing, then assert done
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      stage_counter <= 3'd0;
    end else begin
      state <= next_state;
      if (state == PROCESS && next_state == WAIT_STAGE) begin
        // when PROCESS triggers, we'll wait for stage_done
      end
      if (state == WAIT_STAGE && stage_done) begin
        stage_counter <= stage_counter + 1'b1;
      end
      if (state == IDLE && start) begin
        stage_counter <= 0;
      end
    end
  end

  always_comb begin
    // defaults
    next_state = state;
    stage_start = 1'b0;
    read_en = 1'b0;
    write_en = 1'b0;
    out_valid = 1'b0;
    done = 1'b0;

    case (state)
      IDLE: begin
        if (start) begin
          next_state = LOAD;
        end
      end
      LOAD: begin
        // allow memory to be written from bit reversal stage
        if (mem_ready) begin
          next_state = PROCESS;
        end
      end
      PROCESS: begin
        // start a stage
        stage_start = 1'b1;
        read_en = 1'b1;
        write_en = 1'b1;
        next_state = WAIT_STAGE;
      end
      WAIT_STAGE: begin
        if (stage_done) begin
          if (stage_counter + 1 >= STAGES) begin
            next_state = OUTPUT;
          end else begin
            next_state = PROCESS;
          end
        end
      end
      OUTPUT: begin
        out_valid = 1'b1;
        // signal done when consumer ready and we've provided output
        if (in_valid && in_ready) begin
          done = 1'b1;
          next_state = FINISH;
        end
      end
      FINISH: begin
        // return to idle
        next_state = IDLE;
      end
      default: next_state = IDLE;
    endcase
  end

  assign stage_idx = stage_counter;

endmodule
