`timescale 1ns/1ps

module fft_top (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [31:0] in_data,
  input  logic        in_valid,
  input  logic        in_ready,
  input  logic        start,

  output logic [35:0] out_data, // 18b real | 18b imag (36 bits)
  output logic        out_valid,
  output logic        out_ready,
  output logic        done
);
  import fft_params_pkg::*;

  // Internal wires between modules
  logic [31:0] br_out_data;
  logic        br_out_valid;
  logic        br_out_ready;
  logic        br_done;

  // Control FSM
  logic [2:0] stage_idx;
  logic       stage_start;
  logic       mem_read_en;
  logic       mem_write_en;
  logic       ctrl_out_valid;
  logic       ctrl_done;

  // Memory interface
  logic [7:0]  mem_wr_addr;
  logic [7:0]  mem_rd_addr;
  logic [31:0] mem_wr_data;
  logic [31:0] mem_rd_data;
  logic        mem_ready;

  // Twiddle ROM
  logic [31:0] twiddle_word;
  logic [7:0]  twiddle_addr;

  // Stage interface
  logic [31:0] stage_out_data;
  logic        stage_out_valid;
  logic        stage_out_ready;

  // Instantiate parameter package implicitly via import

  // Bit reversal instance
  fft_bit_reversal br_inst (
    .clk(clk),
    .rst_n(rst_n),
    .in_data(in_data),
    .in_valid(in_valid),
    .in_ready(in_ready),
    .start(start),
    .out_data(br_out_data),
    .out_valid(br_out_valid),
    .out_ready(br_out_ready),
    .done(br_done)
  );

  // Memory instance
  fft_memory mem_inst (
    .clk(clk),
    .rst_n(rst_n),
    .wr_en(mem_write_en),
    .wr_addr(mem_wr_addr),
    .wr_data(mem_wr_data),
    .rd_en(mem_read_en),
    .rd_addr(mem_rd_addr),
    .rd_data(mem_rd_data),
    .ready(mem_ready)
  );

  // Twiddle ROM
  fft_twiddle_rom tw_rom (
    .clk(clk),
    .stage_idx(stage_idx),
    .addr(twiddle_addr),
    .twiddle(twiddle_word)
  );

  // Single stage instance
  fft_stage stage_inst (
    .clk(clk),
    .rst_n(rst_n),
    .stage_idx(stage_idx),
    .in_data(mem_rd_data),
    .in_valid(mem_read_en),
    .in_ready(mem_ready),
    .twiddle_addr(twiddle_addr),
    .twiddle(twiddle_word),
    .cfg_unused(1'b0),
    .out_data(stage_out_data),
    .out_valid(stage_out_valid),
    .out_ready(stage_out_ready)
  );

  // Control FSM
  fft_control_fsm ctrl (
    .clk(clk),
    .rst_n(rst_n),
    .in_valid(in_valid),
    .in_ready(in_ready),
    .stage_done(br_done),
    .mem_ready(mem_ready),
    .start(start),
    .stage_idx(stage_idx),
    .stage_start(stage_start),
    .read_en(mem_read_en),
    .write_en(mem_write_en),
    .out_valid(ctrl_out_valid),
    .done(ctrl_done)
  );

  // Simple glue logic: write data from bit reversal to memory, then read/process through stage
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      mem_wr_addr <= 0;
      mem_wr_data <= 32'h0;
      mem_rd_addr <= 0;
      mem_read_en <= 1'b0;
      mem_write_en <= 1'b0;
      twiddle_addr <= 0;
      out_valid <= 1'b0;
      out_data <= 36'h0;
      done <= 1'b0;
    end else begin
      // Write from bit reversal into memory when BR provides valid
      if (br_out_valid) begin
        mem_write_en <= 1'b1;
        mem_wr_addr <= mem_wr_addr + 1;
        mem_wr_data <= br_out_data;
      end else begin
        mem_write_en <= 1'b0;
      end

      // When controller signals a stage start, begin reading from mem and driving stage
      if (stage_start) begin
        mem_read_en <= 1'b1;
        mem_rd_addr <= mem_rd_addr + 1;
        // choose a twiddle address (simple mapping: use read addr)
        twiddle_addr <= mem_rd_addr[7:0];
      end else begin
        mem_read_en <= 1'b0;
      end

      // When stage produces output, widen and present on top outputs
      if (stage_out_valid) begin
        // extend 16-bit pieces to 18 bits (simple sign extension)
        logic signed [15:0] s_re, s_im;
        s_re = stage_out_data[31:16];
        s_im = stage_out_data[15:0];
        out_data <= { {2{s_re[15]}}, s_re, { {2{s_im[15]}}, s_im } }; // 36-bit: 18b re | 18b im
        out_valid <= 1'b1;
      end else begin
        out_valid <= 1'b0;
      end

      // done propagation
      done <= ctrl_done;
    end
  end

  // Top-level out_ready: passthrough stage_out_ready (stage offers out_ready as output)
  assign out_ready = stage_out_ready;

endmodule
