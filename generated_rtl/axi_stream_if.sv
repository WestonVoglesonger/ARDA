// AXI4-Stream Interface Definition
interface axi_stream_if #(parameter DATA_WIDTH = 24) (input logic ACLK, ARESETN);
    logic [DATA_WIDTH-1:0]    TDATA;
    logic                     TVALID;
    logic                     TREADY;
    logic                     TLAST;
    logic                     TKEEP;

    modport master (
        input  TREADY,
        output TDATA, TVALID, TLAST, TKEEP
    );
    modport slave (
        input  TDATA, TVALID, TLAST, TKEEP,
        output TREADY
    );
endinterface
