# Nerf-Axis: 3D Scanner Context

## Quick Start
- **Sync**: `/Users/spass/ws/soa/nerf-axis/pi_scan/sync.sh`
- **Remote Run**: `ssh gp@gp5.local "cd ~/pi_scan && source venv/bin/activate && python3 full_scan.py"`
- **Full Scan**: `python3 full_scan.py` (Captures 4 views, builds mesh, rotates on TFT)
- **Gallery**: `python3 app.py` (View at `http://gp5.local:5000/gallery`)

## Commands
- **Kill**: `pkill -f "python3 full_scan.py"` or `pkill -f "libcamera-still"`
- **Restart Loop**: `python3 full_scan.py --no-capture` (Re-renders last scan)
- **Run Args**:
  - `--no-capture` (Optional): Skip camera/motor, use existing images.
  - `--step [4,6,8]` (Optional): Mesh density (lower = clearer/slower).

## Hardware Wiring (Direct to Pi - No Breadboard)
### TFT Display (ST7735)
- **VCC/BL**: 3.3V | **GND**: GND
- **SCK**: GPIO 11 (Pin 23)
- **MOSI**: GPIO 10 (Pin 19)
- **CS**: GPIO 8 (Pin 24)
- **DC**: GPIO 24 (Pin 18)
- **RST**: GPIO 25 (Pin 22)

### Stepper Motor (ULN2003)
- **VCC**: 5V (Pin 2/4) | **GND**: GND (Pin 6/9/etc)
- **IN1**: GPIO 17 (Pin 11)
- **IN2**: GPIO 18 (Pin 12)
- **IN3**: GPIO 27 (Pin 13)
- **IN4**: GPIO 22 (Pin 15)

## Important
- `st7735_direct_hello.py` is the **Holy Bible**. DO NOT MODIFY.
- If TFT shows "strikes/dots", ensure `spidev` buffer is sufficient (`sudo nano /boot/cmdline.txt` add `spidev.bufsiz=65536`).
- Always run `sync.sh` after local edits.
