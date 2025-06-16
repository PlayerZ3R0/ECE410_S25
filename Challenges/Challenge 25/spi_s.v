// spi_s.v

module spi_s #(
    parameter WIDTH = 8
)(
    input  wire        sclk,      // SPI clock
    input  wire        cs_n,      // Active‐low chip‐select
    input  wire        mosi,      // Master → Slave data
    output reg         miso,      // Slave → Master data
    input  wire        clk,       // System clock (for rx_data_valid staging)
    input  wire        rst_n,     // Active‐low reset
    output reg [7:0]   rx_data,   // Received byte
    output reg         rx_data_valid,
    input  wire [7:0]   tx_data    // Byte to send back
);

    // internal shift register and bit counter
    reg [WIDTH-1:0] shift_reg;
    reg [3:0]       bit_cnt;

    // Sample MOSI on rising‐edge of SCLK
    always @(posedge sclk or posedge cs_n) begin
        if (cs_n) begin
            bit_cnt   <= 0;
            shift_reg <= 0;
        end else begin
            shift_reg <= { shift_reg[WIDTH-2:0], mosi };
            bit_cnt   <= bit_cnt + 1;
        end
    end

    // Latch rx_data and flag when one byte has arrived
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_data_valid <= 0;
            rx_data       <= 0;
        end else if (bit_cnt == WIDTH) begin
            rx_data       <= shift_reg;
            rx_data_valid <= 1;
        end else begin
            rx_data_valid <= 0;
        end
    end

    // Drive MISO on falling‐edge of SCLK
    always @(negedge sclk or posedge cs_n) begin
        if (cs_n) begin
            miso <= 0;
        end else begin
            miso <= tx_data[WIDTH-1-bit_cnt];
        end
    end

endmodule

//Shifts in MOSI into shift_reg on every rising SCLK
//When bit_cnt == 8, it latches rx_data and pulses rx_data_valid for one clk cycle
//On each falling SCLK, it drives the next bit of tx_data onto MISO