`timescale 1ns/1ps
`include "conv2d_params.svh"

module conv2d_top (
    input  logic                 clk,
    input  logic                 rst_n,
    input  logic [64-1:0]        axi_in_data,
    input  logic                 axi_in_valid,
    input  logic                 axi_out_ready,

    output logic [128-1:0]       axi_out_data,
    output logic                 axi_out_valid
);

// Instantiate AXI interface
logic [PACKED_PIXEL_W-1:0] lb_in_pixel;
logic lb_in_valid;

conv2d_axi_interface #(.DATA_WIDTH(64), .ADDR_WIDTH(10)) axi_if (
    .clk(clk),
    .rst_n(rst_n),
    .in_data(axi_in_data),
    .in_valid(axi_in_valid),
    .out_ready(axi_out_ready),
    .out_px_vec_q(axi_out_data),
    .out_data(axi_out_data),
    .out_valid(axi_out_valid),
    .out_pixel(lb_in_pixel),
    .out_pixel_valid(lb_in_valid)
);

// Control FSM
logic lb_en;
logic [7:0] weight_rd_addr;
logic pe_start;
logic activation_en;

conv2d_control_fsm ctrl (
    .clk(clk),
    .rst_n(rst_n),
    .in_valid(lb_in_valid),
    .out_ready(axi_out_ready),
    .lb_en(lb_en),
    .weight_rd_addr(weight_rd_addr),
    .pe_start(pe_start),
    .activation_en(activation_en)
);

// Line buffer
logic [WINDOW_PACKED_W-1:0] lb_window;
logic lb_window_valid;
logic lb_out_ready = 1'b1;

conv2d_line_buffer lb (
    .clk(clk),
    .rst_n(rst_n),
    .in_pixel(lb_in_pixel),
    .in_valid(lb_in_valid),
    .in_ready(lb_out_ready),
    .window(lb_window),
    .window_valid(lb_window_valid),
    .out_ready(lb_out_ready)
);

// Window extractor
logic [CHANNELS-1:0][WINDOW_ELEMS-1:0][PIXEL_WIDTH-1:0] window_per_channel;
logic window_extract_valid;
conv2d_window_extractor we (
    .line_buffer_window(lb_window),
    .window_valid(lb_window_valid),
    .window_per_channel(window_per_channel),
    .extract_valid(window_extract_valid)
);

// Weight BRAM -> provides weights and bias for addressed output channel
logic [WEIGHTS_PER_CH_W-1:0] weights_single;
logic [BIAS_WIDTH-1:0] bias_single;
conv2d_weight_bram wbram (
    .clk(clk),
    .rst_n(rst_n),
    .addr(weight_rd_addr),
    .enable(1'b1),
    .weights(weights_single),
    .bias(bias_single)
);

// Broadcast weights & biases to PE array by fabricating arrays (simple replication for single-channel read)
logic [NUM_OUT_CHANNELS-1:0][WEIGHTS_PER_CH_W-1:0] weights_all;
logic [NUM_OUT_CHANNELS-1:0][BIAS_WIDTH-1:0] bias_all;

always_comb begin
    for (int i=0;i<NUM_OUT_CHANNELS;i++) begin
        // when control samples addr, weight_bram provides the corresponding channel. For simplicity replicate the same loaded word
        weights_all[i] = weights_single;
        bias_all[i] = bias_single;
    end
end

// PE array
logic [NUM_OUT_CHANNELS-1:0][ACC_WIDTH-1:0] pe_out_vec;
logic pe_out_valid;
conv2d_pe_array pe_arr (
    .window_per_channel(window_per_channel),
    .weights_all_ch(weights_all),
    .bias_all_ch(bias_all),
    .start(pe_start),
    .clk(clk),
    .out_px_vec(pe_out_vec),
    .out_valid(pe_out_valid)
);

// Activation
logic [NUM_OUT_CHANNELS-1:0][7:0] activated_vec;
logic act_valid;
conv2d_activation act (
    .in_px_vec(pe_out_vec),
    .in_valid(pe_out_valid),
    .clk(clk),
    .out_px_vec_q(activated_vec),
    .out_valid(act_valid)
);

// Output buffer
logic [127:0] out_data_buf;
logic outbuf_valid;
conv2d_output_buffer #(.DEPTH(32)) outbuf (
    .in_px_vec_q(activated_vec),
    .in_valid(act_valid),
    .clk(clk),
    .out_ready(axi_out_ready),
    .out_data(out_data_buf),
    .out_valid(outbuf_valid)
);

// Connect to top-level AXI outputs
assign axi_out_data = out_data_buf;
assign axi_out_valid = outbuf_valid;

endmodule
