// Parameter include for AXI4-Stream interface
// The data width for the Conv2D input/output is quantized as INT8/INT16

`ifndef AXI_STREAM_IF_PARAMS_SV
`define AXI_STREAM_IF_PARAMS_SV

parameter AXIS_DATA_WIDTH = 24; // 8b x 3 channels per clock (8x3 = 24b - for input channels)

`endif // AXI_STREAM_IF_PARAMS_SV
