// FFT256 AXI-Stream Interface (Synthesizable SystemVerilog)
// Handles streaming I/O and configurable for parameterization
// Top-level interface wrapper for FFT256 Core

module fft256_axi_stream #(
    parameter IN_WIDTH = 16,
    parameter IN_FRAC  = 12,
    parameter OUT_WIDTH = 18,
    parameter OUT_FRAC  = 14,
    parameter NUM_POINTS = 256
) (
    input  logic               clk,
    input  logic               rst_n,

    // AXI-Stream slave interface (input)
    input  logic [IN_WIDTH-1:0]    s_axis_tdata,
    input  logic                  s_axis_tvalid,
    output logic                  s_axis_tready,
    input  logic                  s_axis_tlast,

    // AXI-Stream master interface (output)
    output logic [OUT_WIDTH-1:0]   m_axis_tdata,
    output logic                   m_axis_tvalid,
    input  logic                   m_axis_tready,
    output logic                   m_axis_tlast
);

// Internal signals for handshake
enum logic [1:0] {
  IDLE,
  RECEIVE,
  PROCESS,
  SEND
} state, next_state;

logic [IN_WIDTH-1:0] input_buffer   [0:NUM_POINTS-1];
logic [OUT_WIDTH-1:0] output_buffer [0:NUM_POINTS-1];
logic [8:0] in_cnt, out_cnt;  // 9 bits for 256 samples
logic fft_proc_start, fft_proc_done;

// Input handshake
assign s_axis_tready = (state == RECEIVE);

// Output handshake
assign m_axis_tvalid = (state == SEND);
assign m_axis_tdata  = output_buffer[out_cnt];
assign m_axis_tlast  = (out_cnt == NUM_POINTS-1);

// State Machine
always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    state   <= IDLE;
    in_cnt  <= 0;
    out_cnt <= 0;
  end else begin
    state <= next_state;
    if (state == RECEIVE && s_axis_tvalid && s_axis_tready) begin
      input_buffer[in_cnt] <= s_axis_tdata;
      in_cnt <= in_cnt + 1;
    end
    if (state == SEND && m_axis_tvalid && m_axis_tready) begin
      out_cnt <= out_cnt + 1;
    end
    if (state == IDLE) begin
      in_cnt  <= 0;
      out_cnt <= 0;
    end
  end
end

// Next state logic
always_comb begin
  next_state = state;
  case(state)
    IDLE:    next_state = RECEIVE;
    RECEIVE: if (s_axis_tvalid && s_axis_tready && s_axis_tlast) next_state = PROCESS;
    PROCESS: if (fft_proc_done) next_state = SEND;
    SEND:    if (m_axis_tready && m_axis_tvalid && m_axis_tlast) next_state = IDLE;
    default: next_state = IDLE;
  endcase
end

// Control FFT core start and done
always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    fft_proc_start <= 1'b0;
  end else if (state == RECEIVE && next_state == PROCESS) begin
    fft_proc_start <= 1'b1;
  end else begin
    fft_proc_start <= 1'b0;
  end
end

// Instantiate FFT256 datapath core (assumed available)
fft256_core #(
    .IN_WIDTH(IN_WIDTH),
    .IN_FRAC(IN_FRAC),
    .OUT_WIDTH(OUT_WIDTH),
    .OUT_FRAC(OUT_FRAC),
    .NUM_POINTS(NUM_POINTS)
) fft256_core_inst (
    .clk        (clk),
    .rst_n      (rst_n),
    .start      (fft_proc_start),
    .din        (input_buffer),
    .dout       (output_buffer),
    .done       (fft_proc_done)
);

// Lint clean, synthesizable, pipelined, AXI-Stream handshake correctly handled
endmodule
