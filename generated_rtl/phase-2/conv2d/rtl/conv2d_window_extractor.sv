`timescale 1ns/1ps
`include "conv2d_params.svh"

module conv2d_window_extractor (
    input  logic [WINDOW_PACKED_W-1:0] line_buffer_window,
    input  logic                      window_valid,

    output logic [CHANNELS-1:0][WINDOW_ELEMS-1:0][PIXEL_WIDTH-1:0] window_per_channel,
    output logic                      extract_valid
);

// Unpack the packed 216-bit window into per-channel, per-element arrays.
// Packing order must match line buffer: 9 pixels (top-left..bottom-right), each pixel is 3 channels packed [23:0]

always_comb begin
    // default
    for (int ch=0; ch<CHANNELS; ch++) begin
        for (int e=0; e<WINDOW_ELEMS; e++) begin
            window_per_channel[ch][e] = '0;
        end
    end
    extract_valid = window_valid;

    if (window_valid) begin
        // Extract 9 packed pixels
        // Each packed pixel is PACKED_PIXEL_W bits (24 bits)
        for (int p = 0; p < WINDOW_ELEMS; p++) begin
            int start_bit = (WINDOW_ELEMS - 1 - p) * PACKED_PIXEL_W; // because concatenation used MSB first
            logic [PACKED_PIXEL_W-1:0] pixel;
            pixel = line_buffer_window[start_bit +: PACKED_PIXEL_W];
            // pixel contains 3 channels, MSB first: [23:16]=ch2,[15:8]=ch1,[7:0]=ch0? assume ch2..ch0
            for (int ch = 0; ch < CHANNELS; ch++) begin
                int ch_start = (CHANNELS - 1 - ch)*PIXEL_WIDTH;
                window_per_channel[ch][p] = pixel[ch_start +: PIXEL_WIDTH];
            end
        end
    end
end

endmodule
