# 🌀 Nerf-Axis: 360° Scan

## 🚀 Quick Start
- **Sync**: `./sync.sh` (Sync local -> Pi `gp@gp5.local`)
- **Run**: `ssh gp@gp5.local "cd ~/pi_scan && source venv/bin/activate && python3 full_scan.py"`
- **Re-visualize**: `python3 full_scan.py --no-capture`

## 🔌 Wiring (Direct-to-Pi)
| Component | Pinout (BCM / Physical) |
| :--- | :--- |
| **TFT (ST7735)** | VCC: 3.3V(1), GND: (6), SCK: 11(23), MOSI: 10(19), CS: 8(24), DC: 24(18), RST: 25(22) |
| **Motor (ULN2003)**| VCC: 5V(2), GND: (9), IN1: 17(11), IN2: 18(12), IN3: 27(13), IN4: 22(15) |

## 📜 Holy Bible 
- **TFT Driver**: Use `DirectST7735` from `st7735_direct_hello.py` strictly.
- **Scanning**: 4 stills -> 90° ACW (128 steps) -> Rotate back -> Fused Mesh -> Render.
- **Optimization**: Avoided `scikit-image`. Using lightweight `cv2` and `trimesh(process=False)`.
- **Object Extraction**: Background pixels are filtered during mesh building to isolate the object.
- **Display Fix**: If dots appear, `sudo nano /boot/cmdline.txt` add `spidev.bufsiz=65536`.

---

 Performance Fix (Buttery Smooth Screen)
Earlier, the screen was lagging because it was limited by a 1MHz hardware SPI speed and a slow Python "pixel-by-pixel" loop. I have:

Boosted hardware SPI speed to 24MHz.
Rewrote the renderer to use NumPy vectorization, processing all 20,480 pixels in a single mathematical operation.