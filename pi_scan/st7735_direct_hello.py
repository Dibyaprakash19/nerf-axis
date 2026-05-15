import signal
import sys
import time

import gpiod
import spidev
from gpiod.line import Direction, Value
from PIL import Image, ImageDraw


WIDTH = 128
HEIGHT = 160
SPI_SPEED = 24_000_000

SWRESET = 0x01
SLPOUT = 0x11
NORON = 0x13
INVOFF = 0x20
INVON = 0x21
DISPOFF = 0x28
DISPON = 0x29
CASET = 0x2A
RASET = 0x2B
RAMWR = 0x2C
MADCTL = 0x36
COLMOD = 0x3A
FRMCTR1 = 0xB1
FRMCTR2 = 0xB2
FRMCTR3 = 0xB3
INVCTR = 0xB4
PWCTR1 = 0xC0
PWCTR2 = 0xC1
PWCTR3 = 0xC2
PWCTR4 = 0xC3
PWCTR5 = 0xC4
VMCTR1 = 0xC5
GMCTRP1 = 0xE0
GMCTRN1 = 0xE1


class DirectST7735:
    def __init__(self, dc_pin=24, rst_pin=25, cs=0, xoff=0, yoff=0, bgr=True, invert=False):
        self.dc_pin = dc_pin
        self.rst_pin = rst_pin
        self.xoff = xoff
        self.yoff = yoff
        self.bgr = bgr
        self.invert = invert
        self.spi = spidev.SpiDev(0, cs)
        self.spi.mode = 0
        self.spi.lsbfirst = False
        self.spi.max_speed_hz = SPI_SPEED

        settings = gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE)
        self.chip = gpiod.Chip("/dev/gpiochip0")
        self.lines = self.chip.request_lines(
            consumer="st7735-direct",
            config={dc_pin: settings, rst_pin: settings},
        )
        self.reset()
        self.init()

    def set_pin(self, pin, state):
        self.lines.set_value(pin, Value.ACTIVE if state else Value.INACTIVE)

    def reset(self):
        self.set_pin(self.rst_pin, True)
        time.sleep(0.15)
        self.set_pin(self.rst_pin, False)
        time.sleep(0.15)
        self.set_pin(self.rst_pin, True)
        time.sleep(0.15)

    def write(self, data, is_data):
        self.set_pin(self.dc_pin, is_data)
        if isinstance(data, int):
            data = [data]
        self.spi.xfer3(list(data))

    def cmd(self, command, data=None, delay=0):
        self.write(command, False)
        if data:
            self.write(data, True)
        if delay:
            time.sleep(delay)

    def init(self):
        self.cmd(SWRESET, delay=0.15)
        self.cmd(SLPOUT, delay=0.5)
        self.cmd(FRMCTR1, [0x01, 0x2C, 0x2D])
        self.cmd(FRMCTR2, [0x01, 0x2C, 0x2D])
        self.cmd(FRMCTR3, [0x01, 0x2C, 0x2D, 0x01, 0x2C, 0x2D])
        self.cmd(INVCTR, [0x07])
        self.cmd(PWCTR1, [0xA2, 0x02, 0x84])
        self.cmd(PWCTR2, [0xC5])
        self.cmd(PWCTR3, [0x0A, 0x00])
        self.cmd(PWCTR4, [0x8A, 0x2A])
        self.cmd(PWCTR5, [0x8A, 0xEE])
        self.cmd(VMCTR1, [0x0E])
        self.cmd(INVON if self.invert else INVOFF)
        self.cmd(MADCTL, [0xC8 if self.bgr else 0xC0])
        self.cmd(COLMOD, [0x05])
        self.cmd(NORON, delay=0.1)
        self.cmd(DISPON, delay=0.1)

    def set_window(self, x0, y0, x1, y1):
        x0 += self.xoff
        x1 += self.xoff
        y0 += self.yoff
        y1 += self.yoff
        self.cmd(CASET, [0, x0, 0, x1])
        self.cmd(RASET, [0, y0, 0, y1])
        self.cmd(RAMWR)

    def display(self, image):
        self.set_window(0, 0, WIDTH - 1, HEIGHT - 1)
        rgb = image.convert("RGB")
        out = bytearray()
        for r, g, b in rgb.getdata():
            value = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
            out.append((value >> 8) & 0xFF)
            out.append(value & 0xFF)
        self.write(out, True)

    def off(self):
        self.cmd(DISPOFF)

    def close(self):
        self.spi.close()
        self.lines.release()
        self.chip.close()


def image(label, color):
    img = Image.new("RGB", (WIDTH, HEIGHT), color)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, WIDTH - 1, HEIGHT - 1), outline="yellow")
    draw.text((8, 58), "HELLO WORLD", fill="white")
    draw.text((8, 78), label, fill="black")
    return img


def run_once(config, seconds):
    print("TEST", config, flush=True)
    display = DirectST7735(**config)
    display.off()
    time.sleep(0.5)
    display.cmd(DISPON)
    display.display(image(f"dc{config['dc_pin']} rst{config['rst_pin']}", "red"))
    time.sleep(seconds)
    display.close()


def main():
    tests = [
        {"dc_pin": 24, "rst_pin": 25, "xoff": 0, "yoff": 0, "bgr": True, "invert": False},
        {"dc_pin": 24, "rst_pin": 25, "xoff": 2, "yoff": 1, "bgr": True, "invert": False},
        {"dc_pin": 24, "rst_pin": 25, "xoff": 0, "yoff": 0, "bgr": False, "invert": False},
        {"dc_pin": 24, "rst_pin": 25, "xoff": 0, "yoff": 0, "bgr": True, "invert": True},
        {"dc_pin": 25, "rst_pin": 24, "xoff": 0, "yoff": 0, "bgr": True, "invert": False},
    ]
    for config in tests:
        run_once(config, 5)
    print("Holding final normal config. Ctrl+C to clear/exit.", flush=True)
    display = DirectST7735(dc_pin=24, rst_pin=25, xoff=0, yoff=0, bgr=True, invert=False)
    display.display(image("holding", "blue"))

    def stop(*_args):
        display.display(Image.new("RGB", (WIDTH, HEIGHT), "black"))
        display.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
