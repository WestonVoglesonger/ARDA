`timescale 1ns/1ps

module fft_bit_reversal (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [31:0] in_data,   // 16b real | 16b imag
  input  logic        in_valid,
  input  logic        in_ready,  // external readiness to send
  input  logic        start,

  output logic [31:0] out_data,
  output logic        out_valid,
  output logic        out_ready, // NOTE: listed as output per architecture
  output logic        done
);

  import fft_params_pkg::*;

  // Simple internal memory to store inputs then read out in bit-reversed order
  logic [31:0] mem [0:N-1];
  logic [7:0]  write_ptr; // supports up to 256
  logic [7:0]  read_ptr;
  logic [7:0]  read_index; // bit-reversed index
  logic        writing;
  logic        reading;
  logic [3:0]  bitrev_bits; // STAGES bits

  // Always indicate we are ready to output (per interface: out_ready is an output)
  assign out_ready = 1'b1;

  // Bit reversal function (combinational helper)
  function automatic logic [7:0] bit_reverse(input logic [7:0] val);
    integer i;
    logic [7:0] tmp;
    begin
      tmp = '0;
      for (i = 0; i < STAGES; i++) begin
        tmp = (tmp << 1) | (val & 1);
        val = val >> 1;
      end
      bit_reverse = tmp;
    end
  endfunction

  // Sequential: manage write/read pointers
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      write_ptr <= 0;
      read_ptr  <= 0;
      read_index <= 0;
      writing <= 1'b0;
      reading <= 1'b0;
      out_valid <= 1'b0;
      done <= 1'b0;
      out_data <= 32'h0;
    end else begin
      // Start triggers writing phase
      if (start && !writing && !reading) begin
        writing <= 1'b1;
        write_ptr <= 0;
        out_valid <= 1'b0;
        done <= 1'b0;
      end

      // Writing phase: accept inputs when external in_valid && in_ready
      if (writing) begin
        if (in_valid && in_ready) begin
          mem[write_ptr] <= in_data;
          write_ptr <= write_ptr + 1;
          if (write_ptr == N-1) begin
            writing <= 1'b0;
            reading <= 1'b1;
            read_ptr <= 0;
            read_index <= bit_reverse(0);
          end
        end
      end else if (reading) begin
        // Produce outputs one per cycle; out_ready is output so we assume consumer always ready
        out_data <= mem[read_index];
        out_valid <= 1'b1;
        // When we have sent this item, advance
        if (out_valid && out_ready) begin
          read_ptr <= read_ptr + 1;
          if (read_ptr == N-1) begin
            reading <= 1'b0;
            out_valid <= 1'b0;
            done <= 1'b1;
          end else begin
            read_index <= bit_reverse(read_ptr + 1);
          end
        end
      end else begin
        out_valid <= 1'b0;
        done <= 1'b0;
      end
    end
  end

endmodule
