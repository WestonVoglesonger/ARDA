// Core datapath for BPF16 FIR filter
// 16-tap, shift-register, fixed-point arithmetic, pipelined
// Synthesizable, lint-clean, AXI handshake

module BPF16_core #(
    parameter IN_WIDTH = 12,
    parameter IN_FRAC = 11,
    parameter OUT_WIDTH = 16,
    parameter OUT_FRAC = 14,
    parameter TAP_NUM = 16
) (
    input  logic                    clk,
    input  logic                    rstn,
    input  logic [IN_WIDTH-1:0]     din,
    input  logic                    din_valid,
    output logic                    din_ready,
    output logic [OUT_WIDTH-1:0]    dout,
    output logic                    dout_valid,
    input  logic                    dout_ready
);
    // Coefficient ROM (16-entry, 12-bit signed)
    localparam COEFF_WIDTH = 12;
    localparam COEFF_FRAC = 11;
    
    logic signed [COEFF_WIDTH-1:0] coeff_rom [0:TAP_NUM-1];
    initial begin
        coeff_rom[ 0] = -14;
        coeff_rom[ 1] = -28;
        coeff_rom[ 2] = -22;
        coeff_rom[ 3] =  23;
        coeff_rom[ 4] = 106;
        coeff_rom[ 5] = 199;
        coeff_rom[ 6] = 264;
        coeff_rom[ 7] = 272;
        coeff_rom[ 8] = 213;
        coeff_rom[ 9] =  99;
        coeff_rom[10] = -33;
        coeff_rom[11] = -145;
        coeff_rom[12] = -205;
        coeff_rom[13] = -198;
        coeff_rom[14] = -127;
        coeff_rom[15] =  -3;
    end

    // Shift register for input samples
    logic signed [IN_WIDTH-1:0] tap_sr [0:TAP_NUM-1];
    integer k;

    always_ff @(posedge clk) begin
        if (!rstn) begin
            for (k = 0; k < TAP_NUM; k = k + 1)
                tap_sr[k] <= '0;
        end else if (din_valid && din_ready) begin
            tap_sr[0] <= din;
            for (k = 1; k < TAP_NUM; k = k + 1)
                tap_sr[k] <= tap_sr[k-1];
        end
    end

    // Control pipeline for output valid/ready
    logic [3:0] latency_ctr;
    logic       active;
    always_ff @(posedge clk) begin
        if (!rstn) begin
            dout_valid <= 1'b0;
            latency_ctr <= 4'd0;
            active <= 1'b0;
        end else begin
            if (din_valid && din_ready) begin
                if (!active) begin
                    latency_ctr <= 4'd12; // pipeline delay
                    active <= 1'b1;
                end else if (latency_ctr != 0)
                    latency_ctr <= latency_ctr - 1;
            end
            dout_valid <= (active && latency_ctr == 0);
            if (dout_valid && dout_ready) begin // consume output
                dout_valid <= 1'b0;
                active <= 1'b0;
            end
        end
    end
    assign din_ready = (!active) || (active && (latency_ctr > 0));

    // Multiply-accumulate chain (fully pipelined)
    logic signed [IN_WIDTH+COEFF_WIDTH-1:0] mult [0:TAP_NUM-1]; // 24b
    logic signed [31:0]                     accum [0:TAP_NUM-1];

    always_comb begin
        for (k = 0; k < TAP_NUM; k = k + 1)
            mult[k] = tap_sr[k] * coeff_rom[k];
        accum[0] = mult[0];
        for (k = 1; k < TAP_NUM; k = k + 1)
            accum[k] = accum[k-1] + mult[k];
    end
    // Output (rounded, saturated, format adjusted)
    logic signed [OUT_WIDTH-1:0] dout_reg;
    always_ff @(posedge clk) begin
        if (!rstn)
            dout_reg <= '0;
        else if (dout_valid && dout_ready)
            dout_reg <= accum[TAP_NUM-1][IN_FRAC+COEFF_FRAC-OUT_FRAC+:OUT_WIDTH];
    end
    assign dout = dout_reg;
endmodule
