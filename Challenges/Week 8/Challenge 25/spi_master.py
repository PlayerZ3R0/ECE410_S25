# for cocotb

import cocotb
from cocotb.triggers import Timer
from cocotb.handle import SimHandleBase

class SPIMaster:
    def __init__(self, dut, sclk_period_ns=100):
        self.dut = dut
        self.sclk_period = sclk_period_ns

    async def transfer(self, data_out):
        """Assert CS, shift out bytes, return list of bytes read back."""
        dut = self.dut
        recv = []

        # Pick up signals
        dut.cs_n <= 1
        dut.sclk <= 0
        await Timer(self.sclk_period, units='ns')

        # Begin transaction
        dut.cs_n <= 0
        await Timer(self.sclk_period, units='ns')

        # Byte-by-byte
        for byte in data_out:
            in_byte = 0
            for bit in range(8):
                # put next bit on MOSI (MSB first)
                bitval = (byte >> (7 - bit)) & 1
                dut.mosi <= bitval

                # half-period low → high
                dut.sclk <= 1
                await Timer(self.sclk_period/2, units='ns')

                # sample MISO
                in_byte = (in_byte << 1) | int(dut.miso.value)

                # finish clock
                dut.sclk <= 0
                await Timer(self.sclk_period/2, units='ns')
            recv.append(in_byte)

        # End transaction
        dut.cs_n <= 1
        await Timer(self.sclk_period, units='ns')
        return recv
