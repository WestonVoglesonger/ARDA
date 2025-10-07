// Coefficient ROM for BPF16, 16 x 12b, synthesizable
module coeff_rom_bpf16 (
    input  logic [$clog2(16)-1:0] addr,
    output logic signed [11:0] data
);
    always_comb begin
        case (addr)
            4'd0:  data = -12'd15;
            4'd1:  data = -12'd28;
            4'd2:  data = -12'd22;
            4'd3:  data =  12'd23;
            4'd4:  data =  12'd106;
            4'd5:  data =  12'd199;
            4'd6:  data =  12'd263;
            4'd7:  data =  12'd272;
            4'd8:  data =  12'd213;
            4'd9:  data =  12'd99;
            4'd10: data = -12'd33;
            4'd11: data = -12'd144;
            4'd12: data = -12'd205;
            4'd13: data = -12'd198;
            4'd14: data = -12'd127;
            4'd15: data = -12'd3;
            default: data = '0;
        endcase
    end
endmodule