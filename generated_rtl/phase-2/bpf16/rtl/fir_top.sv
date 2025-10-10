// fir_top.sv
// Top-level FIR integrating tap buffer, coeff ROM, MAC pipeline, adder tree and control FSM.
// Exposes a ready/valid streaming interface: in_valid/in_ready and out_valid/out_ready.

`include "fir_params.svh"

module fir_top (
    input  logic                      clk,
    input  logic                      rst,
    input  logic                      in_valid,   // input sample valid
    input  logic                      in_ready,   // upstream provides ready permission (handshake)
    input  logic signed [IN_WIDTH-1:0] sample_in,
    input  logic                      out_ready,  // downstream ready

    output logic signed [OUT_WIDTH-1:0] sample_out,
    output logic                      out_valid
);

    // Internal wires
    logic sample_accept;
    logic ctrl_out_valid;

    // Tap buffer outputs
    logic signed [N_TAPS*IN_WIDTH-1:0] taps;
    logic tap_out_valid;

    // Instantiate control FSM
    fir_control_fsm ctrl_inst (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in_ready(in_ready), .out_ready(out_ready),
        .sample_accept(sample_accept), .output_valid(ctrl_out_valid)
    );

    // Tap buffer: sample_accept drives in_valid; in_ready tied high internally (we accept when controller asks)
    fir_tap_buffer tapbuf_inst (
        .clk(clk), .rst(rst),
        .in_valid(sample_accept), .in_ready(1'b1), .sample_in(sample_in),
        .taps_out(taps), .out_valid(tap_out_valid)
    );

    // Instantiate coefficient ROMs in a generate to produce a wide coefficient vector
    logic signed [N_TAPS*COEFF_WIDTH-1:0] coeffs_wide;
    genvar gi;
    generate
        for (gi = 0; gi < N_TAPS; gi = gi + 1) begin : gen_roms
            logic signed [COEFF_WIDTH-1:0] coeff_i;
            fir_coeff_rom rom_i (.addr(gi[3:0]), .coeff(coeff_i));
            assign coeffs_wide[(N_TAPS-gi)*COEFF_WIDTH-1 -: COEFF_WIDTH] = coeff_i;
        end
    endgenerate

    // MAC pipeline: compute accumulator (pipelined)
    logic signed [ACC_WIDTH-1:0] mac_acc;
    logic mac_valid;
    fir_mac_pipeline #(.PIPELINE_DEPTH(4)) mac_inst (
        .clk(clk), .rst(rst), .in_valid(tap_out_valid),
        .samples(taps), .coeffs(coeffs_wide),
        .mac_result(mac_acc), .out_valid(mac_valid)
    );

    // Adder tree instantiation (kept for hierarchy completeness). We will feed it a packed products vector.
    // Build a dummy products vector by sign-extending the mac_acc into the first product lane and zeroing others.
    localparam int PROD_WIDTH = IN_WIDTH + COEFF_WIDTH;
    logic signed [16*PROD_WIDTH-1:0] products_packed;
    integer jj;
    always_comb begin
        // default zeros
        products_packed = '0;
        // sign-extend mac_acc to PROD_WIDTH and place into first product slot
        // mac_acc is ACC_WIDTH wide; extend/truncate to PROD_WIDTH
        logic signed [PROD_WIDTH-1:0] mac_as_prod;
        mac_as_prod = mac_acc[PROD_WIDTH-1:0];
        products_packed[(16-0)*PROD_WIDTH-1 -: PROD_WIDTH] = mac_as_prod;
    end

    logic signed [ACC_WIDTH-1:0] adder_sum;
    logic adder_valid;
    fir_adder_tree adder_inst (
        .clk(clk), .rst(rst), .products_in(products_packed), .in_valid(mac_valid),
        .sum_out(adder_sum), .out_valid(adder_valid)
    );

    // Final output formatting: convert accumulator (mac_acc) from product fractional format to OUT_FRAC
    // product fractional bits = IN_FRAC + COEFF_FRAC
    localparam int PROD_FRAC = IN_FRAC + COEFF_FRAC; // 11 + 14 = 25
    localparam int SHIFT = PROD_FRAC - OUT_FRAC; // 25 - 14 = 11

    // Shift with rounding/truncation and saturate to OUT_WIDTH
    logic signed [ACC_WIDTH-1:0] acc_to_shift;
    logic signed [ACC_WIDTH-1:0] acc_shifted;
    logic signed [OUT_WIDTH-1:0] out_saturated;

    // Choose the accumulator coming from mac pipeline (mac_acc) or adder tree (adder_sum). They should match.
    always_comb begin
        acc_to_shift = mac_acc; // prefer mac_acc (it is the true accumulator)
        // arithmetic right shift
        acc_shifted = acc_to_shift >>> SHIFT;
        // saturation bounds for OUT_WIDTH
        logic signed [ACC_WIDTH-1:0] max_val;
        logic signed [ACC_WIDTH-1:0] min_val;
        max_val = $signed({{(ACC_WIDTH-OUT_WIDTH){1'b0}}, {1'b0}, {(OUT_WIDTH-1){1'b1}}});
        min_val = - (1 << (OUT_WIDTH-1));
        // simple saturation
        if (acc_shifted > max_val) out_saturated = max_val[OUT_WIDTH-1:0];
        else if (acc_shifted < min_val) out_saturated = min_val[OUT_WIDTH-1:0];
        else out_saturated = acc_shifted[OUT_WIDTH-1:0];
    end

    // Drive outputs aligned to mac_valid timing
    always_ff @(posedge clk) begin
        if (rst) begin
            sample_out <= '0;
            out_valid <= 1'b0;
        end else begin
            // We present valid when the MAC pipeline produces a valid result and controller indicates corresponding output_valid
            // The controller tracks accepted samples; we combine with mac_valid for safety
            out_valid <= mac_valid & ctrl_out_valid;
            if (mac_valid & ctrl_out_valid) begin
                sample_out <= out_saturated;
            end
        end
    end

endmodule
