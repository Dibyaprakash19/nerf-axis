import math
import time
import signal
import sys
from PIL import Image, ImageDraw, ImageFont
from st7735_direct_hello import DirectST7735, WIDTH, HEIGHT

def hsv_to_rgb(h, s, v):
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i %= 6
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    return (v, p, q)

def render_donut_frame(angle_a, angle_b):
    cols, rows = 40, 40
    z_buffer = [0.0] * (cols * rows)
    output = [" "] * (cols * rows)
    
    cos_a, sin_a = math.cos(angle_a), math.sin(angle_a)
    cos_b, sin_b = math.cos(angle_b), math.sin(angle_b)
    
    R1, R2, K2 = 1, 2, 5
    K1 = cols * K2 * 3 / (8 * (R1 + R2))
    chars = ".,-~:;=!*#$@"
    
    for theta in [t * 0.06 for t in range(105)]:
        ct, st = math.cos(theta), math.sin(theta)
        for phi in [p * 0.02 for p in range(314)]:
            cp, sp = math.cos(phi), math.sin(phi)
            
            cx = R2 + R1 * ct
            cy = R1 * st
            
            x = cx * (cos_b * cp + sin_a * sin_b * sp) - cy * cos_a * sin_b
            y = cx * (sin_b * cp - sin_a * cos_b * sp) + cy * cos_a * cos_b
            z = K2 + cos_a * cx * sp + cy * sin_a
            ooz = 1 / z
            
            xp = int(cols / 2 + K1 * ooz * x)
            yp = int(rows / 2 + K1 * ooz * y)
            
            L = (cp * ct * sin_b - cos_a * ct * sp - sin_a * st + 
                 cos_b * (cos_a * st - ct * sin_a * sp))
            
            if L > 0:
                idx = xp + cols * yp
                if 0 <= xp < cols and 0 <= yp < rows:
                    if ooz > z_buffer[idx]:
                        z_buffer[idx] = ooz
                        output[idx] = chars[min(len(chars)-1, int(L * 8))]

    img = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(img)
    cw, ch = WIDTH // cols, HEIGHT // rows
    
    for y in range(rows):
        for x in range(cols):
            char = output[x + cols * y]
            if char != " ":
                hue = (angle_a + x/cols + y/rows) % 1.0
                color = tuple(int(c * 255) for c in hsv_to_rgb(hue, 0.8, 1.0))
                draw.text((x * cw, y * ch), char, fill=color)
    return img

def main():
    display = DirectST7735()
    
    def stop(s, f):
        display.display(Image.new("RGB", (WIDTH, HEIGHT), "black"))
        display.close()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    
    # 1. Display "Hello"
    img = Image.new("RGB", (WIDTH, HEIGHT), "blue")
    draw = ImageDraw.Draw(img)
    draw.text((30, 70), "HELLO!", fill="white")
    display.display(img)
    time.sleep(3)
    
    # 2. Display Donut
    a, b = 0.0, 0.0
    while True:
        display.display(render_donut_frame(a, b))
        a += 0.15
        b += 0.08

if __name__ == "__main__":
    main()
