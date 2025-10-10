`timescale 1ns/1ps
`include "conv2d_params.svh"

module conv2d_line_buffer (
    input  logic                 clk,
    input  logic                 rst_n,
    input  logic [PACKED_PIXEL_W-1:0] in_pixel, // 24-bit packed 3x8
    input  logic                 in_valid,
    input  logic                 in_ready,

    output logic [WINDOW_PACKED_W-1:0] window, // 216-bit packed 3x3x3
    output logic                 window_valid,
    output logic                 out_ready
);

// Simple line-buffer implementation for 8x8 image with 3 channels
// Uses two row buffers (shift registers) to provide 3-row sliding window.

// Internal storage for two previous rows (each WIDTH entries of packed pixels)
logic [PACKED_PIXEL_W-1:0] rowbuf0 [WIDTH-1:0];
logic [PACKED_PIXEL_W-1:0] rowbuf1 [WIDTH-1:0];

// write pointer and valid counters
logic [$clog2(WIDTH)-1:0] write_col;
logic [$clog2(HEIGHT)-1:0] write_row;
logic pixels_received;

// Output registers
logic [WINDOW_PACKED_W-1:0] window_r;
logic window_valid_r;

assign window = window_r;
assign window_valid = window_valid_r;

// Backpressure: we present out_ready when downstream isn't applying backpressure
assign out_ready = in_ready;

// Simple control: push incoming pixels into row buffers; when enough rows/cols exist, form window
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        write_col <= 0;
        write_row <= 0;
        pixels_received <= 1'b0;
        window_r <= '0;
        window_valid_r <= 1'b0;
        // initialize buffers
        for (int i=0;i<WIDTH;i++) begin
            rowbuf0[i] <= '0;
            rowbuf1[i] <= '0;
        end
    end else begin
        window_valid_r <= 1'b0; // default

        if (in_valid && out_ready) begin
            // shift rows when starting a new row
            rowbuf1[write_col] <= rowbuf0[write_col];
            rowbuf0[write_col] <= in_pixel;

            // update column pointer
            if (write_col == WIDTH-1) begin
                write_col <= 0;
                if (write_row == HEIGHT-1)
                    write_row <= 0;
                else
                    write_row <= write_row + 1;
            end else begin
                write_col <= write_col + 1;
            end

            // track when we have at least 3 rows written and 3 columns available
            if (!pixels_received)
                pixels_received <= 1'b1;

            // produce window when we have three full rows and at least col >=2
            // For simplicity, generate a window once write_col >=2 and we had at least 2 previous rows
            if ((write_row >= 2 || (pixels_received && write_row != 0)) && write_col >= 2) begin
                // form 3x3 window centered at current write_col-1 position
                int c0 = (write_col >= 2) ? write_col-2 : (write_col + WIDTH - 2);
                int c1 = (write_col >= 1) ? write_col-1 : (write_col + WIDTH - 1);
                int c2 = write_col;

                // Order: row-2,row-1,row (top to bottom), each row: left->right (c0,c1,c2)
                // pack as 9 pixels * CHANNELS * 8bits
                logic [WINDOW_PACKED_W-1:0] wtmp;
                int idx = 0;
                // top row -> rowbuf1 (older), middle -> rowbuf0 (previous), bottom -> in_pixel (current)
                logic [PACKED_PIXEL_W-1:0] top0 = rowbuf1[c0];
                logic [PACKED_PIXEL_W-1:0] top1 = rowbuf1[c1];
                logic [PACKED_PIXEL_W-1:0] top2 = rowbuf1[c2];
                logic [PACKED_PIXEL_W-1:0] mid0 = rowbuf0[c0];
                logic [PACKED_PIXEL_W-1:0] mid1 = rowbuf0[c1];
                logic [PACKED_PIXEL_W-1:0] mid2 = rowbuf0[c2];
                logic [PACKED_PIXEL_W-1:0] bot0 = rowbuf0[c0]; // conservative fallback
                logic [PACKED_PIXEL_W-1:0] bot1 = rowbuf0[c1];
                logic [PACKED_PIXEL_W-1:0] bot2 = in_pixel;

                // packing sequence
                wtmp = {top0, top1, top2, mid0, mid1, mid2, bot0, bot1, bot2};
                window_r <= wtmp;
                window_valid_r <= 1'b1;
            end
        end
    end
end

endmodule
