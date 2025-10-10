`ifndef CONV2D_PARAMS_SVH
`define CONV2D_PARAMS_SVH

// Global parameters for Conv2D RTL
parameter int WIDTH           = 8;
parameter int HEIGHT          = 8;
parameter int CHANNELS        = 3;
parameter int WINDOW_SIZE     = 3;
parameter int WINDOW_ELEMS    = WINDOW_SIZE*WINDOW_SIZE; // 9

parameter int PIXEL_WIDTH     = 8;   // per-channel input pixel width (INT8)
parameter int PACKED_PIXEL_W   = PIXEL_WIDTH*CHANNELS; // 24
parameter int WINDOW_PACKED_W  = WINDOW_ELEMS*CHANNELS*PIXEL_WIDTH; // 9*3*8 = 216

parameter int NUM_OUT_CHANNELS = 16;
parameter int WEIGHT_ELEMS      = WINDOW_ELEMS*CHANNELS; // 27
parameter int WEIGHTS_PER_CH_W  = WEIGHT_ELEMS*8; // 27*8 = 216 bits per out-channel
parameter int BIAS_WIDTH        = 16;
parameter int ACC_WIDTH         = 16;

`endif // CONV2D_PARAMS_SVH
