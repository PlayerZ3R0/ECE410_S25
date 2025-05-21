# cocotb testbench

import cocotb
from cocotb.triggers import RisingEdge
from spi_master import SPIMaster

@cocotb.test()
async def spi_loopback_test(dut):
    """Send a byte and expect the slave to echo it back"""
    # assume tx_data register is tied to rx_data for echo behavior
    master = SPIMaster(dut)

    # Reset logic (if any)
    dut.rst_n <= 0
    await RisingEdge(dut.clk)
    dut.rst_n <= 1
    await RisingEdge(dut.clk)

    # Tell the DUT to echo rx_data back on tx_data
    # (You’d wire rx_data → tx_data in your top‐level)
    dut.tx_data <= 0x00

    # First transfer: send 0xA5, expect echo of previous tx_data (0x00)
    rx = await master.transfer([0xA5])
    assert rx == [0x00], f"Expected [0x00], got {rx}"

    # Hook up echo: when rx_data_valid, route rx_data back to tx_data
    @cocotb.coroutine
    async def echo_proc():
        while True:
            await RisingEdge(dut.clk)
            if dut.rx_data_valid.value:
                dut.tx_data <= dut.rx_data.value

    cocotb.start_soon(echo_proc())

    # Second transfer: now tx_data was set to 0xA5, so we should get it back
    rx2 = await master.transfer([0x00])
    assert rx2 == [0xA5], f"Expected [0xA5], got {rx2}"
